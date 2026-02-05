# -*- coding: utf-8 -*-
"""
Grounding Training with Cached YOLO Features - Enhanced Version

Train ONLY the grounding components (adapter + scorer + MIRL) using
precomputed YOLO features from cache. YOLO models are never loaded.

FEATURES:
- ✅ Full validation loop after each epoch
- ✅ Comprehensive metrics (Loss, Margin Success Rate, Accuracy@1, Mean GT Rank, PCK@50, etc.)
- ✅ Best model selection based on VAL Margin Success Rate
- ✅ CSV logging (train_metrics.csv, val_metrics.csv)
- ✅ No leakage between splits
- ✅ Deterministic and reproducible

CONSTRAINTS:
- No YOLO inference
- No YOLO unfreezing
- No architecture modifications
- Use cached .pt files only
- Test split NOT used during training (only val)

USAGE:
    python training/grounding_train_cached.py --config config/config.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
from collections import defaultdict
from typing import Dict, List, Optional, Literal, TYPE_CHECKING, Tuple

from core.datatypes import D_TOKEN, D_QUERY
from core.metrics import MetricsComputer, GroundingMetrics, format_metrics_table
from core.logging import CSVLogger
from adapter.cross_attention_adapter import CrossAttentionAdapter, create_grounding_adapter

# Phase-2: Hard Negative Mining
from training.hard_negative_mining import HardNegativeMiner, WeightedMIRLLoss

if TYPE_CHECKING:
    from core.config import Config


# =============================================================================
# TRAINABLE COMPONENTS (unchanged from original)
# =============================================================================

class TrainableAdapter(nn.Module):
    """
    Trainable adapter for grounding (no torch.no_grad).
    Uses query-conditioned feature modulation (FiLM-style).
    """
    
    def __init__(self, token_dim: int = 256, query_dim: int = 256):
        super().__init__()
        self.token_dim = token_dim
        self.query_dim = query_dim
        
        # Query-conditioned modulation (FiLM)
        self.gamma_generator = nn.Linear(query_dim, token_dim, bias=True)
        self.beta_generator = nn.Linear(query_dim, token_dim, bias=True)
        
        # Output projection
        self.output_proj = nn.Linear(token_dim, token_dim, bias=False)
        
        # Initialize
        nn.init.zeros_(self.gamma_generator.bias)
        nn.init.zeros_(self.beta_generator.bias)
        nn.init.orthogonal_(self.output_proj.weight)
        
    def forward(self, tokens: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, N, token_dim] or [N, token_dim]
            query: [B, query_dim] or [query_dim]
        Returns:
            grounded_tokens: same shape as tokens
        """
        unbatch = False
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
            unbatch = True
        
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        B, N, D = tokens.shape
        
        gamma = self.gamma_generator(query)
        beta = self.beta_generator(query)
        
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        
        modulated = tokens * (1.0 + gamma) + beta
        output = self.output_proj(modulated)
        
        if unbatch:
            output = output.squeeze(0)
        
        return output


class TrainableScorer(nn.Module):
    """
    Trainable scorer for grounding.
    Uses query-aware scoring with MLP.
    """
    
    def __init__(self, token_dim: int = 256, query_dim: int = 256):
        super().__init__()
        self.token_dim = token_dim
        self.query_dim = query_dim
        
        self.query_proj = nn.Linear(query_dim, token_dim, bias=False)
        
        self.scorer = nn.Sequential(
            nn.Linear(token_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
        
        nn.init.orthogonal_(self.query_proj.weight)
    
    def forward(self, tokens: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        unbatch = False
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
            unbatch = True
        
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        B, N, D = tokens.shape
        
        query_proj = self.query_proj(query)
        query_expanded = query_proj.unsqueeze(1).expand(B, N, -1)
        
        combined = torch.cat([tokens, query_expanded], dim=-1)
        scores = self.scorer(combined).squeeze(-1)
        
        if unbatch:
            scores = scores.squeeze(0)
        
        return scores


class SimpleQueryEncoder(nn.Module):
    """
    Query encoder using sentence-transformers/all-MiniLM-L6-v2.
    Outputs 256D embeddings (projected from 384D).
    
    Supports two modes:
    - Sentence-level: forward/forward_batch → [B, 256]
    - Token-level (Phase-3): forward_tokens_batch → [B, T, 256]
    """
    
    def __init__(self, max_length: int = 64):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel
        
        self.max_length = max_length  # Configurable token length
        
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model.eval()
        
        # Sentence-level projection (384 → 256)
        if D_QUERY != 384:
            self.projection = nn.Linear(384, D_QUERY, bias=False)
            torch.manual_seed(42)
            nn.init.orthogonal_(self.projection.weight)
            self.projection.eval()
            for p in self.projection.parameters():
                p.requires_grad = False
        else:
            self.projection = None
        
        # Token-level projection (Phase-3: 384 → 256 per token)
        # Separate from sentence projection for clarity
        self.token_projection = nn.Linear(384, D_QUERY, bias=False)
        torch.manual_seed(43)  # Different seed for token projection
        nn.init.orthogonal_(self.token_projection.weight)
        self.token_projection.eval()
        for p in self.token_projection.parameters():
            p.requires_grad = False
    
    def _get_device(self) -> torch.device:
        return next(self.model.parameters()).device
    
    def _mean_pooling(self, model_output, attention_mask) -> torch.Tensor:
        token_embeddings = model_output[0]
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def forward(self, text: str) -> torch.Tensor:
        device = self._get_device()
        
        with torch.no_grad():
            encoded = self.tokenizer(
                text, 
                padding=True, 
                truncation=True, 
                max_length=self.max_length, 
                return_tensors='pt'
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            model_output = self.model(**encoded)
            embedding = self._mean_pooling(model_output, encoded['attention_mask'])
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1).squeeze(0)
            
            if self.projection is not None:
                embedding = self.projection(embedding)
        
        return embedding
    
    def forward_batch(self, texts: List[str]) -> torch.Tensor:
        device = self._get_device()
        
        with torch.no_grad():
            encoded = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=self.max_length, 
                return_tensors='pt'
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            model_output = self.model(**encoded)
            embeddings = self._mean_pooling(model_output, encoded['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            if self.projection is not None:
                embeddings = self.projection(embeddings)
        
        return embeddings
    
    def forward_tokens_batch(
        self, 
        texts: List[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Phase-3: Extract token-level embeddings instead of sentence pooling.
        
        Args:
            texts: List of caption strings [B]
        
        Returns:
            token_embeddings: [B, T, D_QUERY] - Token embeddings projected to 256D
            attention_mask: [B, T] - Boolean mask (True = valid token, False = padding)
        """
        device = self._get_device()
        
        with torch.no_grad():
            # Tokenize with padding (uses configurable max_length)
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            # Get last hidden states from transformer
            # model_output[0] = last_hidden_state [B, T, 384]
            model_output = self.model(**encoded)
            token_embeddings = model_output[0]  # [B, T, 384]
            
            # Project each token to D_QUERY (256)
            token_embeddings = self.token_projection(token_embeddings)  # [B, T, 256]
            
            # Get attention mask (True = valid, False = padding)
            attention_mask = encoded['attention_mask'].bool()  # [B, T]
        
        return token_embeddings, attention_mask


class MIRLLoss(nn.Module):
    """
    MIRL (Margin-based Instance Ranking Loss) for referring expression grounding.
    """
    
    def __init__(self, margin: float = 0.2, lambda_reject: float = 0.1):
        super().__init__()
        self.margin = margin
        self.lambda_reject = lambda_reject
    
    def forward(
        self,
        scores: torch.Tensor,
        gt_indices: torch.Tensor,
        valid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, N = scores.shape
        device = scores.device
        
        total_ranking_loss = torch.tensor(0.0, device=device)
        total_rejection_loss = torch.tensor(0.0, device=device)
        valid_samples = 0
        
        for b in range(B):
            gt_idx = gt_indices[b].item()
            valid_mask = valid[b]
            sample_scores = scores[b]
            
            if gt_idx < 0 or gt_idx >= N or not valid_mask[gt_idx]:
                continue
            
            valid_samples += 1
            
            pos_score = sample_scores[gt_idx]
            
            neg_mask = valid_mask.clone()
            neg_mask[gt_idx] = False
            
            if neg_mask.any():
                neg_scores = sample_scores[neg_mask]
                margins = pos_score - neg_scores
                ranking_loss = torch.relu(self.margin - margins).mean()
                total_ranking_loss = total_ranking_loss + ranking_loss
            
            valid_scores = sample_scores[valid_mask]
            rejection_loss = torch.relu(-valid_scores).mean()
            total_rejection_loss = total_rejection_loss + rejection_loss
        
        if valid_samples > 0:
            total_ranking_loss = total_ranking_loss / valid_samples
            total_rejection_loss = total_rejection_loss / valid_samples
        
        total = total_ranking_loss + self.lambda_reject * total_rejection_loss
        
        return {
            "total": total,
            "ranking": total_ranking_loss,
            "rejection": total_rejection_loss,
        }


# =============================================================================
# COLLATE FUNCTION
# =============================================================================

def collate_variable_humans(batch: List[Dict]) -> Optional[Dict]:
    """Custom collate function for variable number of humans per image."""
    batch = [s for s in batch if s is not None]
    
    if len(batch) == 0:
        return None
    
    max_humans = max(sample['visual_embeddings'].shape[0] for sample in batch)
    
    if max_humans == 0:
        return None
    
    batch_size = len(batch)
    embed_dim = batch[0]['visual_embeddings'].shape[1]
    
    first_masks = batch[0]['masks']
    if first_masks.dim() == 3:
        mask_h, mask_w = first_masks.shape[1], first_masks.shape[2]
    else:
        mask_h, mask_w = 160, 160
    
    visual_embeddings = torch.zeros(batch_size, max_humans, embed_dim)
    boxes = torch.zeros(batch_size, max_humans, 4)
    masks = torch.zeros(batch_size, max_humans, mask_h, mask_w)
    keypoints = torch.zeros(batch_size, max_humans, 17, 3)
    valid = torch.zeros(batch_size, max_humans, dtype=torch.bool)
    
    captions = []
    gt_indices = []
    image_ids = []
    
    for i, sample in enumerate(batch):
        n_humans = sample['visual_embeddings'].shape[0]
        
        if n_humans > 0:
            visual_embeddings[i, :n_humans] = sample['visual_embeddings']
            boxes[i, :n_humans] = sample['boxes']
            keypoints[i, :n_humans] = sample['keypoints']
            
            sample_masks = sample['masks']
            if sample_masks.dim() == 3 and sample_masks.shape[0] == n_humans:
                if sample_masks.shape[1] != mask_h or sample_masks.shape[2] != mask_w:
                    sample_masks = torch.nn.functional.interpolate(
                        sample_masks.unsqueeze(1).float(),
                        size=(mask_h, mask_w),
                        mode='nearest'
                    ).squeeze(1)
                masks[i, :n_humans] = sample_masks
            
            sample_valid = sample['valid']
            if sample_valid.shape[0] == n_humans:
                valid[i, :n_humans] = sample_valid
            else:
                valid[i, :n_humans] = True
        
        captions.append(sample['caption'])
        
        gt_idx = sample['gt_index']
        if not isinstance(gt_idx, int):
            gt_idx = int(gt_idx)
        if gt_idx >= n_humans or gt_idx < 0:
            gt_idx = -1
        gt_indices.append(gt_idx)
        
        image_ids.append(sample.get('image_id', i))
    
    return {
        'visual_embeddings': visual_embeddings,
        'boxes': boxes,
        'masks': masks,
        'keypoints': keypoints,
        'valid': valid,
        'caption': captions,
        'gt_index': torch.tensor(gt_indices, dtype=torch.long),
        'image_id': image_ids,
    }


# =============================================================================
# CACHED FEATURE DATASET
# =============================================================================

_CACHED_SPLIT_INDICES: Dict[str, Dict[str, List[int]]] = {}


def compute_sample_id_for_cache(image_id: str, ann_id: int, caption: str) -> str:
    """Compute unique sample ID for split assignment."""
    import hashlib
    import string
    text = caption.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = ' '.join(text.split()).strip()
    cap_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
    return f"{image_id}_{ann_id}_{cap_hash}"


class CachedFeatureDataset(Dataset):
    """Dataset that loads precomputed YOLO features from cache."""
    
    def __init__(
        self, 
        cache_dir: Path, 
        coco_json_path: Path, 
        split: Optional[Literal["train", "val", "test"]] = None,
        split_config: Optional[Dict] = None,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.cache_dir = cache_dir
        self.seed = seed
        self.split = split
        self.split_config = split_config or {'train': 0.8, 'val': 0.1, 'test': 0.1, 'seed': 42}
        
        total_ratio = self.split_config['train'] + self.split_config['val'] + self.split_config['test']
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
        
        all_cache_files = sorted(list(cache_dir.glob("*.pt")))
        
        print(f"\nCachedFeatureDataset initializing...")
        print(f"  Cache directory: {cache_dir}")
        print(f"  Total cache files found: {len(all_cache_files)}")
        print(f"  Split: {split if split else 'ALL'}")
        
        print(f"  Loading COCO annotations from: {coco_json_path}")
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        self.image_id_to_anns = defaultdict(list)
        for ann in coco_data['annotations']:
            image_id = str(ann['image_id'])
            self.image_id_to_anns[image_id].append(ann)
        
        print(f"  COCO annotations loaded: {len(coco_data['annotations'])}")
        print(f"  Unique images with annotations: {len(self.image_id_to_anns)}")
        
        self._all_samples = []
        
        for cache_file in tqdm(all_cache_files, desc="  Validating cache files"):
            image_id = cache_file.stem
            
            if image_id not in self.image_id_to_anns:
                continue
            
            anns = self.image_id_to_anns[image_id]
            
            for ann_idx, ann in enumerate(anns):
                if 'caption' not in ann or not ann['caption']:
                    continue
                
                sample_id = compute_sample_id_for_cache(
                    image_id, 
                    ann.get('id', ann_idx),
                    ann['caption']
                )
                
                self._all_samples.append({
                    'cache_file': cache_file,
                    'image_id': image_id,
                    'ann_idx': ann_idx,
                    'ann_id': ann.get('id', ann_idx),
                    'caption': ann['caption'],
                    'bbox': ann.get('bbox', None),
                    'sample_id': sample_id,
                })
        
        print(f"  Total valid samples: {len(self._all_samples)}")
        
        cache_key = str(coco_json_path.resolve())
        
        if cache_key not in _CACHED_SPLIT_INDICES:
            self._compute_splits()
            _CACHED_SPLIT_INDICES[cache_key] = self._split_indices
        else:
            self._split_indices = _CACHED_SPLIT_INDICES[cache_key]
        
        if self.split is None:
            self.samples = self._all_samples
        else:
            indices = self._split_indices[self.split]
            self.samples = [self._all_samples[i] for i in indices]
        
        print(f"  Samples in '{split if split else 'ALL'}' split: {len(self.samples)}")
        
        if max_samples is not None and len(self.samples) > max_samples:
            import random
            random.seed(seed)
            shuffled_indices = list(range(len(self.samples)))
            random.shuffle(shuffled_indices)
            self.samples = [self.samples[i] for i in shuffled_indices[:max_samples]]
            print(f"  Limited to: {len(self.samples)} samples")
        
        self._print_split_stats()
        
        self._cache = {}
        self._cache_max_size = 100
    
    def _compute_splits(self):
        n_total = len(self._all_samples)
        
        if n_total == 0:
            self._split_indices = {"train": [], "val": [], "test": []}
            return
        
        indices = list(range(n_total))
        rng = random.Random(self.split_config['seed'])
        rng.shuffle(indices)
        
        n_train = int(n_total * self.split_config['train'])
        n_val = int(n_total * self.split_config['val'])
        
        self._split_indices = {
            "train": indices[:n_train],
            "val": indices[n_train:n_train + n_val],
            "test": indices[n_train + n_val:],
        }
        
        train_ids = {self._all_samples[i]['sample_id'] for i in self._split_indices['train']}
        val_ids = {self._all_samples[i]['sample_id'] for i in self._split_indices['val']}
        test_ids = {self._all_samples[i]['sample_id'] for i in self._split_indices['test']}
        
        if train_ids & val_ids:
            raise RuntimeError("[LEAKAGE] Samples in both train and val!")
        if train_ids & test_ids:
            raise RuntimeError("[LEAKAGE] Samples in both train and test!")
        if val_ids & test_ids:
            raise RuntimeError("[LEAKAGE] Samples in both val and test!")
        
        total_split = len(train_ids) + len(val_ids) + len(test_ids)
        if total_split != n_total:
            raise RuntimeError(f"[SPLIT ERROR] {total_split} != {n_total}")
    
    def _print_split_stats(self):
        total = len(self._all_samples)
        train_n = len(self._split_indices['train'])
        val_n = len(self._split_indices['val'])
        test_n = len(self._split_indices['test'])
        
        print(f"\n  SAMPLE-LEVEL SPLIT STATISTICS:")
        print(f"    Total curated samples: {total:,}")
        print(f"    Train samples: {train_n:,} ({train_n/total*100:.1f}%)")
        print(f"    Val samples:   {val_n:,} ({val_n/total*100:.1f}%)")
        print(f"    Test samples:  {test_n:,} ({test_n/total*100:.1f}%)")
        print(f"    [PASS] Sample-level split complete")
        print(f"    [PASS] No leakage detected")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_cache_file(self, cache_file: Path) -> Dict:
        cache_key = str(cache_file)
        
        if cache_key not in self._cache:
            if len(self._cache) >= self._cache_max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            
            self._cache[cache_key] = torch.load(cache_file, weights_only=True)
        
        return self._cache[cache_key]
    
    def _match_gt_index(
        self, 
        ann_bbox: Optional[List[float]], 
        cached_boxes: torch.Tensor,
    ) -> int:
        N = cached_boxes.shape[0]
        
        if ann_bbox is None or N == 0:
            return 0
        
        x, y, w, h = ann_bbox
        ann_cx = x + w / 2
        ann_cy = y + h / 2
        
        best_idx = 0
        best_dist = float('inf')
        
        for i in range(N):
            box = cached_boxes[i]
            box_cx = (box[0] + box[2]) / 2
            box_cy = (box[1] + box[3]) / 2
            
            ann_cx_norm = ann_cx / 640
            ann_cy_norm = ann_cy / 640
            
            dist = (box_cx - ann_cx_norm) ** 2 + (box_cy - ann_cy_norm) ** 2
            
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        
        return best_idx
    
    def __getitem__(self, idx: int) -> Optional[Dict]:
        sample_info = self.samples[idx]
        cache_file = sample_info['cache_file']
        
        try:
            cache_data = self._load_cache_file(cache_file)
            
            visual_embeddings = cache_data['visual_embeddings']
            boxes = cache_data['boxes']
            masks = cache_data['masks']
            keypoints = cache_data['keypoints']
            valid = cache_data['valid']
            
            N = boxes.shape[0]
            
            if N == 0:
                return None
            
            caption = sample_info['caption']
            gt_index = self._match_gt_index(sample_info.get('bbox'), boxes)
            
            if gt_index >= N:
                gt_index = N - 1
            
            return {
                "visual_embeddings": visual_embeddings,
                "boxes": boxes,
                "masks": masks,
                "keypoints": keypoints,
                "valid": valid,
                "caption": caption,
                "gt_index": gt_index,
                "image_id": sample_info['image_id'],
            }
            
        except Exception as e:
            print(f"Warning: Failed to load {cache_file}: {e}")
            return None


# =============================================================================
# VALIDATION FUNCTION
# =============================================================================

@torch.no_grad()
def validate_epoch(
    adapter: nn.Module,
    scorer: nn.Module,
    query_encoder: nn.Module,
    mirl_loss_fn: nn.Module,
    val_dataloader: DataLoader,
    device: str,
    use_token_level_alignment: bool = False,
) -> GroundingMetrics:
    """
    Run validation loop and compute all metrics.
    
    Args:
        adapter: Grounding adapter module
        scorer: Scoring module
        query_encoder: Query encoder module
        mirl_loss_fn: MIRL loss function
        val_dataloader: Validation data loader
        device: Device string
        use_token_level_alignment: If True, use Phase-3 token-level encoding
    
    Returns:
        GroundingMetrics for validation set
    """
    adapter.eval()
    scorer.eval()
    
    metrics_computer = MetricsComputer()
    
    for batch in tqdm(val_dataloader, desc="Validating", leave=False):
        if batch is None:
            continue
        
        visual_embeddings = batch['visual_embeddings'].to(device)
        boxes = batch['boxes'].to(device)
        keypoints = batch['keypoints'].to(device)
        valid = batch['valid'].to(device)
        captions = batch['caption']
        gt_indices = batch['gt_index'].to(device)
        
        B, N, D = visual_embeddings.shape
        
        if B == 0 or N == 0:
            continue
        
        # Forward pass - encode captions
        with torch.no_grad():
            if use_token_level_alignment:
                # Phase-3: Token-level embeddings
                caption_tokens, caption_mask = query_encoder.forward_tokens_batch(captions)
                query_embeddings = query_encoder.forward_batch(captions)
            else:
                query_embeddings = query_encoder.forward_batch(captions)
        
        # Forward pass - adapter
        with torch.no_grad():
            if use_token_level_alignment:
                grounded_tokens = adapter(visual_embeddings, caption_tokens, caption_mask)
            else:
                grounded_tokens = adapter(visual_embeddings, query_embeddings)
            
            scores = scorer(grounded_tokens, query_embeddings)
        
        # Compute loss
        loss_dict = mirl_loss_fn(scores, gt_indices, valid)
        total_loss = loss_dict["total"]
        
        # Compute batch metrics
        batch_metrics = metrics_computer.compute_batch_metrics(
            scores=scores,
            gt_indices=gt_indices,
            valid=valid,
            loss=total_loss,
            keypoints_pred=keypoints,
            keypoints_gt=keypoints[torch.arange(B), gt_indices.clamp(0, N-1)],
            boxes=boxes,
        )
        
        metrics_computer.accumulate(batch_metrics)
    
    adapter.train()
    scorer.train()
    
    return metrics_computer.get_accumulated_metrics()


# =============================================================================
# TRAINING LOOP
# =============================================================================

def _run_training_loop(
    config: "Config",
    cache_dir: Path,
    checkpoint_dir: Path,
    coco_json: Path,
    device: str,
    num_epochs: int,
    batch_size: int,
    max_steps_per_epoch: Optional[int],
    learning_rate: float,
):
    """Internal training loop with validation."""
    
    # =========================================================================
    # MODEL INITIALIZATION
    # =========================================================================
    
    print("\n" + "-" * 50)
    print("Initializing model components (NO YOLO)")
    print("-" * 50)
    
    # ==========================================================================
    # EXPERIMENT MODE RESOLUTION
    # ==========================================================================
    resolved_adapter_type, resolved_hnm_enabled, mode_description = config.grounding.resolve_experiment_mode()
    
    print(f"\n{'='*50}")
    print(f"EXPERIMENT MODE: {mode_description}")
    print(f"{'='*50}")
    print(f"  Resolved adapter_type: {resolved_adapter_type}")
    print(f"  Resolved HNM enabled: {resolved_hnm_enabled}")
    print(f"  Text encoder max_length: {config.grounding.text_encoder.max_length}")
    
    query_encoder = SimpleQueryEncoder(
        max_length=config.grounding.text_encoder.max_length
    )
    query_encoder.to(device)
    query_encoder.eval()
    for param in query_encoder.parameters():
        param.requires_grad = False
    print(f"✓ SimpleQueryEncoder loaded (frozen, max_length={config.grounding.text_encoder.max_length})")
    
    # ==========================================================================
    # ADAPTER SELECTION (Phase-0/1/3: film, cross_attention, text_visual_alignment)
    # ==========================================================================
    adapter_type = resolved_adapter_type
    print(f"\n  Adapter type: {adapter_type}")
    
    # Track if we're using token-level alignment (affects forward pass)
    use_token_level_alignment = False
    
    if adapter_type == "cross_attention":
        # Phase-1 improvement: Cross-Attention based grounding
        ca_config = config.grounding.cross_attention
        adapter = create_grounding_adapter(
            adapter_type="cross_attention",
            token_dim=D_TOKEN,
            query_dim=D_QUERY,
            num_heads=ca_config.num_heads,
            num_layers=ca_config.num_layers,
            dim_feedforward=ca_config.dim_feedforward,
            dropout=ca_config.dropout,
        )
        print(f"✓ CrossAttentionAdapter initialized (trainable)")
        print(f"    num_heads: {ca_config.num_heads}")
        print(f"    num_layers: {ca_config.num_layers}")
        print(f"    dim_feedforward: {ca_config.dim_feedforward}")
        print(f"    dropout: {ca_config.dropout}")
    elif adapter_type == "text_visual_alignment":
        # Phase-3: Token-level cross-modal alignment
        tva_config = config.grounding.text_visual_alignment
        adapter = create_grounding_adapter(
            adapter_type="text_visual_alignment",
            token_dim=D_TOKEN,
            query_dim=D_QUERY,
            num_heads=tva_config.num_heads,
            num_layers=tva_config.num_layers,
            dim_feedforward=tva_config.dim_feedforward,
            dropout=tva_config.dropout,
            bidirectional=tva_config.bidirectional,
        )
        use_token_level_alignment = True
        print(f"✓ TextVisualAlignmentAdapter initialized (Phase-3, trainable)")
        print(f"    num_heads: {tva_config.num_heads}")
        print(f"    num_layers: {tva_config.num_layers}")
        print(f"    dim_feedforward: {tva_config.dim_feedforward}")
        print(f"    dropout: {tva_config.dropout}")
        print(f"    bidirectional: {tva_config.bidirectional}")
        print(f"✓ Token-level cross-attention active (max_length={config.grounding.text_encoder.max_length})")
    else:
        # Baseline: FiLM-style adapter (Phase-0)
        adapter = TrainableAdapter(token_dim=D_TOKEN, query_dim=D_QUERY)
        print(f"✓ TrainableAdapter (FiLM) initialized (Phase-0 baseline, trainable)")
    
    adapter.to(device)
    adapter.train()
    
    # Count adapter parameters
    adapter_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"    Trainable parameters: {adapter_params:,}")
    
    scorer = TrainableScorer(token_dim=D_TOKEN, query_dim=D_QUERY)
    scorer.to(device)
    scorer.train()
    print(f"✓ TrainableScorer initialized (trainable)")
    
    # ==========================================================================
    # LOSS FUNCTION SELECTION (Phase-2: Hard Negative Mining)
    # ==========================================================================
    hnm_config = config.grounding.hard_negative_mining
    # Use resolved HNM setting (from experiment_mode or direct config)
    hnm_enabled = resolved_hnm_enabled
    
    if hnm_enabled:
        # Phase-2: Weighted MIRL with Hard Negative Mining
        mirl_loss_fn = WeightedMIRLLoss(margin=0.2, lambda_reject=0.1)
        hard_negative_miner = HardNegativeMiner(hnm_config)
        
        print(f"\n✓ Hard Negative Mining: ENABLED")
        print(f"    Difficulty weights: IoU={hnm_config.weight_iou}, pose={hnm_config.weight_pose}, size={hnm_config.weight_size}")
        if hnm_config.curriculum_enabled:
            print(f"    Curriculum: {hnm_config.curriculum_start_ratio:.0%} → {hnm_config.curriculum_end_ratio:.0%} over {hnm_config.curriculum_warmup_epochs} warmup epochs")
        else:
            print(f"    Curriculum: DISABLED (fixed hard ratio)")
        print(f"    Top-K hard negatives: {hnm_config.top_k_hard}")
        print(f"    Hard negative weight: {hnm_config.hard_negative_weight}x")
        print(f"✓ WeightedMIRLLoss initialized (margin=0.2, lambda_reject=0.1)")
    else:
        # Standard MIRL (no hard negative mining)
        mirl_loss_fn = MIRLLoss(margin=0.2, lambda_reject=0.1)
        hard_negative_miner = None
        
        print(f"\n✓ Hard Negative Mining: DISABLED")
        print(f"✓ MIRLLoss initialized (margin=0.2, lambda_reject=0.1)")
    
    # =========================================================================
    # DATASET SETUP
    # =========================================================================
    
    print("\n" + "-" * 50)
    print("Preparing datasets")
    print("-" * 50)
    
    split_config = {
        'train': config.splits.train,
        'val': config.splits.val,
        'test': config.splits.test,
        'seed': config.splits.seed,
    }
    
    print(f"\n  Split configuration (sample-level):")
    print(f"    Train: {split_config['train']:.0%}")
    print(f"    Val:   {split_config['val']:.0%}")
    print(f"    Test:  {split_config['test']:.0%}")
    print(f"    Seed:  {split_config['seed']}")
    
    # Create TRAIN dataset
    print("\n  Creating TRAIN dataset...")
    train_dataset = CachedFeatureDataset(
        cache_dir=cache_dir,
        coco_json_path=coco_json,
        split="train",
        split_config=split_config,
        max_samples=None,
        seed=config.runtime.seed,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_variable_humans,
        drop_last=True,
    )
    
    # Create VAL dataset (for validation after each epoch)
    print("\n  Creating VAL dataset...")
    val_dataset = CachedFeatureDataset(
        cache_dir=cache_dir,
        coco_json_path=coco_json,
        split="val",
        split_config=split_config,
        max_samples=None,
        seed=config.runtime.seed,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_variable_humans,
        drop_last=False,
    )
    
    print(f"\n✓ DataLoaders ready")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # =========================================================================
    # OPTIMIZER & SCHEDULER
    # =========================================================================
    
    trainable_params = list(adapter.parameters()) + list(scorer.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=learning_rate,
        weight_decay=config.training.weight_decay,
    )
    print(f"✓ Optimizer initialized (AdamW, lr={learning_rate})")
    
    total_steps = len(train_dataloader) * num_epochs
    if max_steps_per_epoch is not None:
        total_steps = min(total_steps, max_steps_per_epoch * num_epochs)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=total_steps,
        eta_min=learning_rate * 0.01,
    )
    print(f"✓ Scheduler initialized (CosineAnnealing, T_max={total_steps})")
    
    # =========================================================================
    # CSV LOGGERS
    # =========================================================================
    
    logs_dir = config.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    train_logger = CSVLogger(
        logs_dir / "train_metrics.csv",
        columns=CSVLogger.GROUNDING_COLUMNS,
        overwrite=True,  # Start fresh each training run
    )
    
    val_logger = CSVLogger(
        logs_dir / "val_metrics.csv",
        columns=CSVLogger.GROUNDING_COLUMNS,
        overwrite=True,
    )
    
    print(f"✓ CSV loggers initialized")
    print(f"  Train log: {logs_dir / 'train_metrics.csv'}")
    print(f"  Val log: {logs_dir / 'val_metrics.csv'}")
    
    # =========================================================================
    # METRICS & SELECTION CRITERIA
    # =========================================================================
    
    print("\n" + "-" * 50)
    print("Metrics enabled:")
    print("-" * 50)
    print("  - Margin Success Rate (PRIMARY)")
    print("  - Accuracy@1")
    print("  - Mean GT Rank")
    print("  - PCK@50 (Train / Val / Test)")
    print("  - Avg GT Score / Avg Max Neg Score")
    print("\n  Best model selection criterion: VAL Margin Success Rate")
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    
    print("\n" + "-" * 50)
    print("Starting training with validation")
    print("-" * 50)
    
    best_val_margin_rate = 0.0
    best_checkpoint_path = None
    step_counter = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch + 1}/{num_epochs}")
        print(f"{'='*70}")
        
        # ---------------------------------------------------------------------
        # TRAINING PHASE
        # ---------------------------------------------------------------------
        adapter.train()
        scorer.train()
        
        train_metrics_computer = MetricsComputer()
        epoch_losses = []
        epoch_hardness_scores = []  # Phase-2: Track average hardness for debugging
        epoch_caption_lengths = []  # Phase-3: Track caption token lengths
        
        # Phase-2: Get current hard negative ratio from curriculum
        if hard_negative_miner is not None:
            current_hard_ratio = hard_negative_miner.get_hard_ratio(epoch, num_epochs)
            print(f"  [Phase-2] Current hard negative ratio: {current_hard_ratio:.1%}")
        
        pbar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            if max_steps_per_epoch is not None and batch_idx >= max_steps_per_epoch:
                break
            
            if batch is None:
                continue
            
            visual_embeddings = batch['visual_embeddings'].to(device)
            boxes = batch['boxes'].to(device)
            keypoints = batch['keypoints'].to(device)
            valid = batch['valid'].to(device)
            captions = batch['caption']
            gt_indices = batch['gt_index'].to(device)
            
            B, N, D = visual_embeddings.shape
            
            if B == 0 or N == 0:
                continue
            
            # Forward pass - encode captions
            with torch.no_grad():
                if use_token_level_alignment:
                    # Phase-3: Token-level embeddings [B, T, 256] + mask [B, T]
                    caption_tokens, caption_mask = query_encoder.forward_tokens_batch(captions)
                    # Also get sentence-level for scorer (still uses pooled query)
                    query_embeddings = query_encoder.forward_batch(captions)
                    
                    # Phase-3 diagnostic: track actual caption token lengths
                    actual_lengths = caption_mask.sum(dim=1).tolist()  # [B]
                    epoch_caption_lengths.extend(actual_lengths)
                else:
                    # Phase-0/1: Sentence-level embedding [B, 256]
                    query_embeddings = query_encoder.forward_batch(captions)
            
            # Forward pass - adapter
            if use_token_level_alignment:
                # Phase-3: Token-level cross-modal alignment
                grounded_tokens = adapter(visual_embeddings, caption_tokens, caption_mask)
            else:
                # Phase-0/1: Sentence-level grounding
                grounded_tokens = adapter(visual_embeddings, query_embeddings)
            
            # Forward pass - scorer (always uses sentence-level query)
            scores = scorer(grounded_tokens, query_embeddings)
            
            # Compute loss (Phase-2: with optional hard negative weights)
            if hard_negative_miner is not None:
                # Compute difficulty-aware weights for negatives
                negative_weights, weight_stats = hard_negative_miner.compute_negative_weights(
                    boxes=boxes,
                    keypoints=keypoints,
                    gt_indices=gt_indices,
                    valid=valid,
                    epoch=epoch,
                    max_epochs=num_epochs,
                )
                loss_dict = mirl_loss_fn(scores, gt_indices, valid, negative_weights=negative_weights)
                
                # Track hardness for debugging
                if 'avg_hardness' in weight_stats:
                    epoch_hardness_scores.append(weight_stats['avg_hardness'])
            else:
                # Standard MIRL (no hard negative mining)
                loss_dict = mirl_loss_fn(scores, gt_indices, valid)
            
            total_loss = loss_dict["total"]
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"Warning: Invalid loss at step {step_counter}, skipping")
                continue
            
            # Backward & optimize
            optimizer.zero_grad()
            total_loss.backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params, 
                max_norm=config.training.grad_clip_norm
            )
            
            optimizer.step()
            scheduler.step()
            
            # Accumulate metrics (including PCK@50 - requires keypoints/boxes)
            with torch.no_grad():
                # Compute predicted indices for PCK (argmax of scores over valid humans)
                valid_scores = scores.clone()
                valid_scores[~valid] = float('-inf')
                pred_indices = valid_scores.argmax(dim=1)  # [B]
                
                batch_metrics = train_metrics_computer.compute_batch_metrics(
                    scores=scores,
                    gt_indices=gt_indices,
                    valid=valid,
                    loss=total_loss,
                    keypoints_pred=keypoints,
                    keypoints_gt=keypoints[torch.arange(B, device=device), gt_indices.clamp(0, N-1)],
                    boxes=boxes,
                )
                train_metrics_computer.accumulate(batch_metrics)
            
            loss_val = total_loss.item()
            epoch_losses.append(loss_val)
            
            pbar.set_postfix({
                'loss': f'{loss_val:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
            })
            
            step_counter += 1
        
        # Get epoch training metrics
        train_metrics = train_metrics_computer.get_accumulated_metrics()
        
        # Phase-2: Log hardness debug statistic
        if hard_negative_miner is not None and len(epoch_hardness_scores) > 0:
            avg_epoch_hardness = sum(epoch_hardness_scores) / len(epoch_hardness_scores)
            print(f"  [Phase-2] Avg hardness of sampled negatives: {avg_epoch_hardness:.4f}")
        
        # Phase-3: Log caption length statistics
        if use_token_level_alignment and len(epoch_caption_lengths) > 0:
            avg_len = sum(epoch_caption_lengths) / len(epoch_caption_lengths)
            max_len = max(epoch_caption_lengths)
            min_len = min(epoch_caption_lengths)
            print(f"  [Phase-3] Caption token lengths: avg={avg_len:.1f}, min={min_len}, max={max_len}")
            if max_len >= config.grounding.text_encoder.max_length:
                print(f"  [Phase-3] ⚠️ Some captions hit max_length={config.grounding.text_encoder.max_length} (may be truncated)")
        
        # Log training metrics
        # Note: rejection_accuracy removed - dataset contains no rejection samples
        train_logger.log({
            "epoch": epoch + 1,
            "loss": train_metrics.loss,
            "margin_success_rate": train_metrics.margin_success_rate,
            "accuracy_at_1": train_metrics.accuracy_at_1,
            "mean_gt_rank": train_metrics.mean_gt_rank,
            "pck_50": train_metrics.pck_50,
            "avg_gt_score": train_metrics.avg_gt_score,
            "avg_max_neg_score": train_metrics.avg_max_neg_score,
        })
        
        print(format_metrics_table(train_metrics, f"TRAIN Epoch {epoch+1}"))
        
        # ---------------------------------------------------------------------
        # VALIDATION PHASE
        # ---------------------------------------------------------------------
        print("\n  Running validation...")
        val_metrics = validate_epoch(
            adapter=adapter,
            scorer=scorer,
            query_encoder=query_encoder,
            mirl_loss_fn=mirl_loss_fn,
            val_dataloader=val_dataloader,
            device=device,
            use_token_level_alignment=use_token_level_alignment,
        )
        
        # Log validation metrics
        # Note: rejection_accuracy removed - dataset contains no rejection samples
        val_logger.log({
            "epoch": epoch + 1,
            "loss": val_metrics.loss,
            "margin_success_rate": val_metrics.margin_success_rate,
            "accuracy_at_1": val_metrics.accuracy_at_1,
            "mean_gt_rank": val_metrics.mean_gt_rank,
            "pck_50": val_metrics.pck_50,
            "avg_gt_score": val_metrics.avg_gt_score,
            "avg_max_neg_score": val_metrics.avg_max_neg_score,
        })
        
        print(format_metrics_table(val_metrics, f"VAL Epoch {epoch+1}"))
        
        # ---------------------------------------------------------------------
        # CHECKPOINTING
        # ---------------------------------------------------------------------
        
        # Save epoch checkpoint
        checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1}.pt"
        torch.save({
            "epoch": epoch + 1,
            "adapter": adapter.state_dict(),
            "scorer": scorer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step_counter": step_counter,
            "train_metrics": train_metrics.to_dict(),
            "val_metrics": val_metrics.to_dict(),
        }, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")
        
        # Save best model based on VAL margin success rate
        if val_metrics.margin_success_rate > best_val_margin_rate:
            best_val_margin_rate = val_metrics.margin_success_rate
            best_checkpoint_path = checkpoint_dir / "best_model.pt"
            torch.save({
                "epoch": epoch + 1,
                "adapter": adapter.state_dict(),
                "scorer": scorer.state_dict(),
                "val_margin_success_rate": val_metrics.margin_success_rate,
                "val_metrics": val_metrics.to_dict(),
            }, best_checkpoint_path)
            print(f"  ★ New best model saved: {best_checkpoint_path}")
            print(f"    VAL Margin Success Rate: {val_metrics.margin_success_rate*100:.2f}%")
    
    # =========================================================================
    # TRAINING COMPLETE
    # =========================================================================
    
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/max(step_counter,1):.3f}s/step)")
    print(f"Best VAL margin success rate: {best_val_margin_rate*100:.2f}%")
    if best_checkpoint_path:
        print(f"Best model: {best_checkpoint_path}")
    print(f"\nCSV logs saved to: {logs_dir}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def train_grounding(config: "Config"):
    """Main training function."""
    
    cache_dir = config.features_dir
    checkpoint_dir = config.checkpoint_dir
    coco_json = config.annotations_path
    
    device = config.training.device
    num_epochs = config.training.num_epochs
    batch_size = config.training.batch_size
    max_steps_per_epoch = config.training.max_steps_per_epoch
    learning_rate = config.training.learning_rate
    
    print("\n" + "=" * 70)
    print("GROUNDING TRAINING WITH CACHED YOLO FEATURES")
    print("=" * 70)
    
    # Device validation
    print(f"\nDevice configuration: {device}")
    if device.startswith("cuda"):
        assert torch.cuda.is_available(), \
            f"Config specifies device='{device}' but CUDA is not available!"
        print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
    
    # Verify cache
    if not cache_dir.exists():
        print(f"\n❌ ABORT: Cache directory not found: {cache_dir}")
        sys.exit(1)
    
    cache_files = list(cache_dir.glob("*.pt"))
    if len(cache_files) == 0:
        print(f"\n❌ ABORT: No cached .pt files found in {cache_dir}")
        sys.exit(1)
    
    # Validate cache metadata
    metadata_path = cache_dir / "cache_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            cache_meta = json.load(f)
        
        cached_finetuned = cache_meta.get("yolo_fine_tuned", False)
        config_finetuned = config.yolo.fine_tune
        
        if cached_finetuned != config_finetuned:
            print(f"\n❌ ABORT: YOLO weight mismatch!")
            print(f"  Cache was built with: {'FINE-TUNED' if cached_finetuned else 'PRETRAINED'} weights")
            print(f"  Config expects:       {'FINE-TUNED' if config_finetuned else 'PRETRAINED'} weights")
            print(f"\n  To fix: Re-run cache_yolo_features.py with matching config")
            sys.exit(1)
        
        print(f"\n✓ Cache metadata validated:")
        print(f"  Created: {cache_meta.get('created_at', 'unknown')}")
        print(f"  YOLO weights: {'FINE-TUNED' if cached_finetuned else 'PRETRAINED'}")
    else:
        print(f"\n⚠ WARNING: No cache metadata found at {metadata_path}")
    
    print(f"\nCache found: YES")
    print(f"Cached images: {len(cache_files)}")
    
    # Create directories
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    config.logs_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Logs directory: {config.logs_dir}")
    
    # Print config
    print(f"\nConfiguration (from config file):")
    print(f"  num_epochs: {num_epochs}")
    print(f"  batch_size: {batch_size}")
    print(f"  max_steps_per_epoch: {max_steps_per_epoch}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  device: {device}")
    
    # Run training
    _run_training_loop(
        config=config,
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        coco_json=coco_json,
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        max_steps_per_epoch=max_steps_per_epoch,
        learning_rate=learning_rate,
    )


if __name__ == "__main__":
    import argparse
    from core.config import load_config, add_config_argument
    
    parser = argparse.ArgumentParser(description="Train grounding with cached YOLO features")
    add_config_argument(parser)
    args = parser.parse_args()
    
    config = load_config(args.config)
    train_grounding(config)
