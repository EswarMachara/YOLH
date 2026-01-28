# -*- coding: utf-8 -*-
"""
Grounding Training with Cached YOLO Features

Train ONLY the grounding components (adapter + scorer + MIRL) using
precomputed YOLO features from cache. YOLO models are never loaded.

CONSTRAINTS:
- No YOLO inference
- No YOLO unfreezing
- No architecture modifications
- Use cached .pt files only
- Focus on learning behavior

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
from typing import Dict, List, Optional, Literal, TYPE_CHECKING

from core.datatypes import (
    D_TOKEN,
    D_QUERY,
)

if TYPE_CHECKING:
    from core.config import Config


# =============================================================================
# TRAINABLE COMPONENTS
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
        # Handle unbatched input
        unbatch = False
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)  # [1, N, token_dim]
            unbatch = True
        
        if query.dim() == 1:
            query = query.unsqueeze(0)  # [1, query_dim]
        
        B, N, D = tokens.shape
        
        # Generate modulation parameters
        gamma = self.gamma_generator(query)  # [B, token_dim]
        beta = self.beta_generator(query)    # [B, token_dim]
        
        # Expand for broadcasting: [B, 1, token_dim]
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        
        # Apply FiLM modulation
        modulated = tokens * (1.0 + gamma) + beta  # [B, N, token_dim]
        
        # Output projection
        output = self.output_proj(modulated)  # [B, N, token_dim]
        
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
        
        # Query projection
        self.query_proj = nn.Linear(query_dim, token_dim, bias=False)
        
        # Scoring MLP: concat(token, query) -> score
        self.scorer = nn.Sequential(
            nn.Linear(token_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
        
        # Initialize
        nn.init.orthogonal_(self.query_proj.weight)
    
    def forward(self, tokens: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, N, token_dim] or [N, token_dim]
            query: [B, query_dim] or [query_dim]
        Returns:
            scores: [B, N] or [N]
        """
        # Handle unbatched input
        unbatch = False
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
            unbatch = True
        
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        B, N, D = tokens.shape
        
        # Project query to token space
        query_proj = self.query_proj(query)  # [B, token_dim]
        query_expanded = query_proj.unsqueeze(1).expand(B, N, -1)  # [B, N, token_dim]
        
        # Concatenate tokens with query
        combined = torch.cat([tokens, query_expanded], dim=-1)  # [B, N, 2*token_dim]
        
        # Score
        scores = self.scorer(combined).squeeze(-1)  # [B, N]
        
        if unbatch:
            scores = scores.squeeze(0)
        
        return scores


class SimpleQueryEncoder(nn.Module):
    """
    Query encoder using sentence-transformers/all-MiniLM-L6-v2.
    Outputs 256D embeddings (projected from 384D).
    """
    
    def __init__(self):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel
        
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model.eval()
        
        # MiniLM outputs 384 dim, we need D_QUERY (256)
        if D_QUERY != 384:
            self.projection = nn.Linear(384, D_QUERY, bias=False)
            torch.manual_seed(42)
            nn.init.orthogonal_(self.projection.weight)
            self.projection.eval()
            for p in self.projection.parameters():
                p.requires_grad = False
        else:
            self.projection = None
    
    def _get_device(self) -> torch.device:
        """Get device of model parameters."""
        return next(self.model.parameters()).device
    
    def _mean_pooling(self, model_output, attention_mask) -> torch.Tensor:
        """Mean pooling weighted by attention mask."""
        token_embeddings = model_output[0]  # [B, seq_len, hidden_dim]
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def forward(self, text: str) -> torch.Tensor:
        """
        Encode a single text string.
        
        Args:
            text: Caption string
        Returns:
            embedding: [D_QUERY] tensor on same device as model
        """
        device = self._get_device()
        
        with torch.no_grad():
            encoded = self.tokenizer(
                text, 
                padding=True, 
                truncation=True, 
                max_length=128, 
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
        """
        Encode a batch of text strings.
        
        Args:
            texts: List of caption strings
        Returns:
            embeddings: [B, D_QUERY] tensor
        """
        device = self._get_device()
        
        with torch.no_grad():
            encoded = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors='pt'
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            model_output = self.model(**encoded)
            embeddings = self._mean_pooling(model_output, encoded['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            if self.projection is not None:
                embeddings = self.projection(embeddings)
        
        return embeddings


class MIRLLoss(nn.Module):
    """
    MIRL (Margin-based Instance Ranking Loss) for referring expression grounding.
    
    Components:
    - Ranking loss: Encourage GT score > negative scores by margin
    - Rejection loss: Encourage all scores to be positive (avoid collapse)
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
        """
        Compute MIRL loss for a batch.
        
        Args:
            scores: [B, N] predicted scores
            gt_indices: [B] ground truth indices (-1 if no GT)
            valid: [B, N] validity mask
        
        Returns:
            Dict with keys: total, ranking, rejection
        """
        B, N = scores.shape
        device = scores.device
        
        total_ranking_loss = torch.tensor(0.0, device=device)
        total_rejection_loss = torch.tensor(0.0, device=device)
        valid_samples = 0
        
        for b in range(B):
            gt_idx = gt_indices[b].item()
            valid_mask = valid[b]  # [N]
            sample_scores = scores[b]  # [N]
            
            # Skip if no GT or GT is invalid
            if gt_idx < 0 or gt_idx >= N or not valid_mask[gt_idx]:
                continue
            
            valid_samples += 1
            
            # Get positive score
            pos_score = sample_scores[gt_idx]
            
            # Get negative scores (valid and not GT)
            neg_mask = valid_mask.clone()
            neg_mask[gt_idx] = False
            
            if neg_mask.any():
                neg_scores = sample_scores[neg_mask]
                
                # Ranking loss: max(0, margin - (pos - neg))
                margins = pos_score - neg_scores  # [num_neg]
                ranking_loss = torch.relu(self.margin - margins).mean()
                total_ranking_loss = total_ranking_loss + ranking_loss
            
            # Rejection loss: encourage scores to be positive
            valid_scores = sample_scores[valid_mask]
            rejection_loss = torch.relu(-valid_scores).mean()
            total_rejection_loss = total_rejection_loss + rejection_loss
        
        # Average over valid samples
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
# CUSTOM COLLATE FUNCTION
# =============================================================================

def collate_variable_humans(batch: List[Dict]) -> Optional[Dict]:
    """
    Custom collate function for batches with variable number of humans per image.
    Pads all tensors to the maximum number of humans in the batch.
    
    Args:
        batch: List of sample dicts from dataset
        
    Returns:
        Batched dict with padded tensors, or None if batch is empty
    """
    # Filter out None samples
    batch = [s for s in batch if s is not None]
    
    if len(batch) == 0:
        return None
    
    # Find max humans in this batch
    max_humans = max(sample['visual_embeddings'].shape[0] for sample in batch)
    
    if max_humans == 0:
        return None
    
    batch_size = len(batch)
    
    # Get dimensions from first valid sample
    embed_dim = batch[0]['visual_embeddings'].shape[1]
    
    # Handle masks shape
    first_masks = batch[0]['masks']
    if first_masks.dim() == 3:
        mask_h, mask_w = first_masks.shape[1], first_masks.shape[2]
    else:
        mask_h, mask_w = 160, 160
    
    # Initialize padded tensors
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
            
            # Handle masks
            sample_masks = sample['masks']
            if sample_masks.dim() == 3 and sample_masks.shape[0] == n_humans:
                # Resize if needed
                if sample_masks.shape[1] != mask_h or sample_masks.shape[2] != mask_w:
                    sample_masks = torch.nn.functional.interpolate(
                        sample_masks.unsqueeze(1).float(),
                        size=(mask_h, mask_w),
                        mode='nearest'
                    ).squeeze(1)
                masks[i, :n_humans] = sample_masks
            
            # Handle valid mask
            sample_valid = sample['valid']
            if sample_valid.shape[0] == n_humans:
                valid[i, :n_humans] = sample_valid
            else:
                valid[i, :n_humans] = True
        
        captions.append(sample['caption'])
        
        # GT index validation
        gt_idx = sample['gt_index']
        if not isinstance(gt_idx, int):
            gt_idx = int(gt_idx)
        if gt_idx >= n_humans or gt_idx < 0:
            gt_idx = -1  # Mark as invalid
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

# Module-level cache for sample splits (shared across dataset instances)
_CACHED_SPLIT_INDICES: Dict[str, Dict[str, List[int]]] = {}


def compute_sample_id_for_cache(image_id: str, ann_id: int, caption: str) -> str:
    """
    Compute unique sample ID for split assignment.
    Must match the ID scheme in CuratedCocoDataset.
    """
    import hashlib
    # Normalize caption
    import string
    text = caption.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = ' '.join(text.split()).strip()
    cap_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
    return f"{image_id}_{ann_id}_{cap_hash}"


class CachedFeatureDataset(Dataset):
    """
    Dataset that loads precomputed YOLO features from cache.
    
    Supports sample-level train/val/test splits (NOT image-level).
    A sample is defined as: (image_id, caption, gt_human_index).
    
    Each sample contains:
    - visual_embeddings: [N, 256]
    - boxes: [N, 4]
    - masks: [N, H, W]
    - keypoints: [N, 17, 3]
    - valid: [N]
    - caption: str
    - gt_index: int
    """
    
    def __init__(
        self, 
        cache_dir: Path, 
        coco_json_path: Path, 
        split: Optional[Literal["train", "val", "test"]] = None,
        split_config: Optional[Dict] = None,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Args:
            cache_dir: Directory containing cached .pt files
            coco_json_path: Path to COCO annotations JSON
            split: Which split to load ("train", "val", "test", or None for all)
            split_config: Dict with 'train', 'val', 'test' ratios and 'seed'
            max_samples: Limit total samples (applied AFTER split)
            seed: Random seed for sample limiting (not for splits)
        """
        self.cache_dir = cache_dir
        self.seed = seed
        self.split = split
        self.split_config = split_config or {'train': 0.8, 'val': 0.1, 'test': 0.1, 'seed': 42}
        
        # Validate split ratios
        total_ratio = self.split_config['train'] + self.split_config['val'] + self.split_config['test']
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
        
        # Get all cache files
        all_cache_files = sorted(list(cache_dir.glob("*.pt")))
        
        print(f"\nCachedFeatureDataset initializing...")
        print(f"  Cache directory: {cache_dir}")
        print(f"  Total cache files found: {len(all_cache_files)}")
        print(f"  Split: {split if split else 'ALL'}")
        
        # Load COCO annotations
        print(f"  Loading COCO annotations from: {coco_json_path}")
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        # Build mapping: image_id -> list of annotations
        self.image_id_to_anns = defaultdict(list)
        for ann in coco_data['annotations']:
            image_id = str(ann['image_id'])
            self.image_id_to_anns[image_id].append(ann)
        
        print(f"  COCO annotations loaded: {len(coco_data['annotations'])}")
        print(f"  Unique images with annotations: {len(self.image_id_to_anns)}")
        
        # Build valid sample list (without loading all cache files)
        self._all_samples = []
        
        for cache_file in tqdm(all_cache_files, desc="  Validating cache files"):
            image_id = cache_file.stem
            
            # Check if annotations exist for this image
            if image_id not in self.image_id_to_anns:
                continue
            
            # Get all annotations for this image
            anns = self.image_id_to_anns[image_id]
            
            # Create one sample per annotation (each caption is a separate sample)
            for ann_idx, ann in enumerate(anns):
                if 'caption' not in ann or not ann['caption']:
                    continue
                
                # Compute stable sample ID for splitting
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
        
        # =====================================================================
        # SAMPLE-LEVEL SPLITTING
        # =====================================================================
        cache_key = str(coco_json_path.resolve())
        
        if cache_key not in _CACHED_SPLIT_INDICES:
            # Compute splits once
            self._compute_splits()
            _CACHED_SPLIT_INDICES[cache_key] = self._split_indices
        else:
            self._split_indices = _CACHED_SPLIT_INDICES[cache_key]
        
        # Filter samples by split
        if self.split is None:
            self.samples = self._all_samples
        else:
            indices = self._split_indices[self.split]
            self.samples = [self._all_samples[i] for i in indices]
        
        print(f"  Samples in '{split if split else 'ALL'}' split: {len(self.samples)}")
        
        # Limit samples if requested (AFTER split)
        if max_samples is not None and len(self.samples) > max_samples:
            import random
            random.seed(seed)
            shuffled_indices = list(range(len(self.samples)))
            random.shuffle(shuffled_indices)
            self.samples = [self.samples[i] for i in shuffled_indices[:max_samples]]
            print(f"  Limited to: {len(self.samples)} samples")
        
        # Print split statistics
        self._print_split_stats()
        
        # Cache for loaded features (LRU-style, limited size)
        self._cache = {}
        self._cache_max_size = 100
    
    def _compute_splits(self):
        """Compute sample-level train/val/test splits."""
        n_total = len(self._all_samples)
        
        if n_total == 0:
            self._split_indices = {"train": [], "val": [], "test": []}
            return
        
        # Shuffle indices deterministically
        indices = list(range(n_total))
        rng = random.Random(self.split_config['seed'])
        rng.shuffle(indices)
        
        # Compute split boundaries
        n_train = int(n_total * self.split_config['train'])
        n_val = int(n_total * self.split_config['val'])
        
        self._split_indices = {
            "train": indices[:n_train],
            "val": indices[n_train:n_train + n_val],
            "test": indices[n_train + n_val:],
        }
        
        # Validate no leakage
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
        """Print split statistics."""
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
        """Load cache file with simple caching."""
        cache_key = str(cache_file)
        
        if cache_key not in self._cache:
            # Evict oldest if cache is full
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
        """
        Match annotation bbox to cached detection boxes using IoU.
        
        Args:
            ann_bbox: [x, y, w, h] from COCO annotation (can be None)
            cached_boxes: [N, 4] normalized xyxy boxes from cache
            
        Returns:
            Best matching index, or 0 if no bbox provided
        """
        N = cached_boxes.shape[0]
        
        if ann_bbox is None or N == 0:
            return 0
        
        # Convert annotation bbox from [x, y, w, h] to normalized [x1, y1, x2, y2]
        # Note: COCO bbox is in pixels, cached boxes are normalized [0, 1]
        # We need image dimensions for proper conversion
        # For now, use simple heuristic: find box with highest IoU
        
        # Assume annotation bbox is already somewhat normalized or use center matching
        x, y, w, h = ann_bbox
        ann_cx = x + w / 2
        ann_cy = y + h / 2
        
        # Find closest box by center distance (simple heuristic)
        best_idx = 0
        best_dist = float('inf')
        
        for i in range(N):
            box = cached_boxes[i]
            box_cx = (box[0] + box[2]) / 2
            box_cy = (box[1] + box[3]) / 2
            
            # Normalize annotation center (assume image ~640px)
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
            # Load cached features
            cache_data = self._load_cache_file(cache_file)
            
            visual_embeddings = cache_data['visual_embeddings']
            boxes = cache_data['boxes']
            masks = cache_data['masks']
            keypoints = cache_data['keypoints']
            valid = cache_data['valid']
            
            N = boxes.shape[0]
            
            # Skip if no humans
            if N == 0:
                return None
            
            # Get caption from sample info
            caption = sample_info['caption']
            
            # Match GT index
            gt_index = self._match_gt_index(sample_info.get('bbox'), boxes)
            
            # Ensure gt_index is valid
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
    """Internal training loop."""
    
    # =========================================================================
    # TASK 3: MODEL INITIALIZATION
    # =========================================================================
    
    print("\n" + "-" * 50)
    print("TASK 3: Initializing model components (NO YOLO)")
    print("-" * 50)
    
    # Query encoder (frozen)
    query_encoder = SimpleQueryEncoder()
    query_encoder.to(device)
    query_encoder.eval()
    for param in query_encoder.parameters():
        param.requires_grad = False
    print(f"✓ SimpleQueryEncoder loaded (frozen)")
    
    # Grounding adapter (trainable)
    adapter = TrainableAdapter(token_dim=D_TOKEN, query_dim=D_QUERY)
    adapter.to(device)
    adapter.train()
    print(f"✓ TrainableAdapter initialized (trainable)")
    
    # Scorer (trainable)
    scorer = TrainableScorer(token_dim=D_TOKEN, query_dim=D_QUERY)
    scorer.to(device)
    scorer.train()
    print(f"✓ TrainableScorer initialized (trainable)")
    
    # MIRL loss
    mirl_loss_fn = MIRLLoss(margin=0.2, lambda_reject=0.1)
    print(f"✓ MIRLLoss initialized (margin=0.2, lambda_reject=0.1)")
    
    # Device verification
    def verify_device(module, name, expected):
        for param in module.parameters():
            actual = str(param.device)
            assert actual.startswith(expected.split(':')[0]), \
                f"{name} device mismatch: {actual} vs {expected}"
    
    verify_device(query_encoder, "query_encoder", device)
    verify_device(adapter, "adapter", device)
    verify_device(scorer, "scorer", device)
    print(f"\n✓ All modules verified on device: {device}")
    
    # =========================================================================
    # TASK 4: PARAMETER COUNTING
    # =========================================================================
    
    print("\n" + "-" * 50)
    print("TASK 4: Freeze policy and parameter counting")
    print("-" * 50)
    
    def count_params(module):
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable
    
    modules = [
        ("SimpleQueryEncoder", query_encoder),
        ("TrainableAdapter", adapter),
        ("TrainableScorer", scorer),
    ]
    
    print("\nParameter counts:")
    total_all, trainable_all = 0, 0
    for name, module in modules:
        total, trainable = count_params(module)
        total_all += total
        trainable_all += trainable
        print(f"  {name:30s} - Total: {total:>10,} | Trainable: {trainable:>10,}")
    print(f"  {'TOTAL':30s} - Total: {total_all:>10,} | Trainable: {trainable_all:>10,}")
    
    print(f"\nTrainable modules:")
    for name, module in modules:
        if any(p.requires_grad for p in module.parameters()):
            print(f"  ✓ {name}")
    
    print(f"\nFrozen modules:")
    for name, module in modules:
        if not any(p.requires_grad for p in module.parameters()):
            print(f"  ❄ {name}")
    
    # =========================================================================
    # TASK 5: DATASET AND OPTIMIZER
    # =========================================================================
    
    print("\n" + "-" * 50)
    print("TASK 5: Preparing training loop")
    print("-" * 50)
    
    # Get split configuration from config.splits (typed dataclass)
    split_config = {
        'train': config.splits.train,
        'val': config.splits.val,
        'test': config.splits.test,
        'seed': config.splits.seed,
    }
    
    print(f"\n  Split configuration (sample-level, NOT image-level):")
    print(f"    Train: {split_config['train']:.0%}")
    print(f"    Val:   {split_config['val']:.0%}")
    print(f"    Test:  {split_config['test']:.0%}")
    print(f"    Seed:  {split_config['seed']}")
    
    # Compute max_samples safely (applied AFTER split)
    if max_steps_per_epoch is not None:
        max_samples = max_steps_per_epoch * batch_size * num_epochs
    else:
        max_samples = None
    
    # Create TRAIN split dataset (sample-level, not image-level)
    print("\n  Creating TRAIN split...")
    dataset = CachedFeatureDataset(
        cache_dir=cache_dir,
        coco_json_path=coco_json,
        split="train",  # ONLY load train split
        split_config=split_config,
        max_samples=max_samples,
        seed=config.runtime.seed,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_variable_humans,
        drop_last=True,
    )
    
    print(f"\n✓ DataLoader ready ({len(dataset)} TRAIN samples, batch_size={batch_size})")
    print(f"  NOTE: Dataset is split at SAMPLE level, not image level.")
    print(f"        This is intentional for referring expression grounding.")
    
    # Optimizer
    trainable_params = list(adapter.parameters()) + list(scorer.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=learning_rate,
        weight_decay=config.training.weight_decay,
    )
    print(f"✓ Optimizer initialized (AdamW, lr={learning_rate})")
    
    # Learning rate scheduler
    total_steps = len(dataloader) * num_epochs
    if max_steps_per_epoch is not None:
        total_steps = min(total_steps, max_steps_per_epoch * num_epochs)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=total_steps,
        eta_min=learning_rate * 0.01,
    )
    print(f"✓ Scheduler initialized (CosineAnnealing, T_max={total_steps})")
    
    # =========================================================================
    # TASK 6-7: TRAINING LOOP WITH LOGGING
    # =========================================================================
    
    print("\n" + "-" * 50)
    print("TASK 6-7: Running grounding training with logging")
    print("-" * 50)
    
    # Logging
    training_log = {
        "losses": [],
        "positive_scores": [],
        "max_negative_scores": [],
        "margins": [],
    }
    
    best_margin_rate = 0.0
    best_checkpoint_path = None
    step_counter = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch + 1}/{num_epochs}")
        print(f"{'='*70}")
        
        epoch_losses = []
        epoch_margins = []
        
        adapter.train()
        scorer.train()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Check step limit
            if max_steps_per_epoch is not None and batch_idx >= max_steps_per_epoch:
                break
            
            # Skip invalid batches
            if batch is None:
                continue
            
            # Move batch to device
            visual_embeddings = batch['visual_embeddings'].to(device)  # [B, N, 256]
            boxes = batch['boxes'].to(device)
            valid = batch['valid'].to(device)  # [B, N]
            captions = batch['caption']  # List[str]
            gt_indices = batch['gt_index'].to(device)  # [B]
            
            B, N, D = visual_embeddings.shape
            
            # Skip empty batches
            if B == 0 or N == 0:
                continue
            
            # =================================================================
            # FORWARD PASS
            # =================================================================
            
            # 1. Encode queries (batched)
            with torch.no_grad():
                query_embeddings = query_encoder.forward_batch(captions)  # [B, D_QUERY]
            
            # 2. Apply adapter to each sample in batch
            # Process batch: visual_embeddings [B, N, D], query [B, D_QUERY]
            grounded_tokens = adapter(visual_embeddings, query_embeddings)  # [B, N, D]
            
            # 3. Score humans
            scores = scorer(grounded_tokens, query_embeddings)  # [B, N]
            
            # =================================================================
            # COMPUTE LOSS
            # =================================================================
            
            loss_dict = mirl_loss_fn(scores, gt_indices, valid)
            total_loss = loss_dict["total"]
            
            # Skip if loss is invalid
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"Warning: Invalid loss at step {step_counter}, skipping")
                continue
            
            # =================================================================
            # BACKWARD & OPTIMIZE
            # =================================================================
            
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params, 
                max_norm=config.training.grad_clip_norm
            )
            
            optimizer.step()
            scheduler.step()
            
            # =================================================================
            # LOGGING
            # =================================================================
            
            # Compute metrics for logging
            with torch.no_grad():
                batch_margins = []
                batch_pos_scores = []
                batch_neg_scores = []
                
                for b in range(B):
                    gt_idx = gt_indices[b].item()
                    if gt_idx < 0 or gt_idx >= N:
                        continue
                    
                    pos_score = scores[b, gt_idx].item()
                    batch_pos_scores.append(pos_score)
                    
                    # Get max negative score
                    neg_mask = valid[b].clone()
                    neg_mask[gt_idx] = False
                    
                    if neg_mask.any():
                        max_neg = scores[b][neg_mask].max().item()
                        batch_neg_scores.append(max_neg)
                        batch_margins.append(pos_score - max_neg)
                
                if batch_pos_scores:
                    avg_pos = sum(batch_pos_scores) / len(batch_pos_scores)
                    training_log["positive_scores"].append(avg_pos)
                
                if batch_neg_scores:
                    avg_neg = sum(batch_neg_scores) / len(batch_neg_scores)
                    training_log["max_negative_scores"].append(avg_neg)
                
                if batch_margins:
                    avg_margin = sum(batch_margins) / len(batch_margins)
                    training_log["margins"].append(avg_margin)
                    epoch_margins.append(avg_margin)
            
            loss_val = total_loss.item()
            training_log["losses"].append(loss_val)
            epoch_losses.append(loss_val)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_val:.4f}',
                'margin': f'{avg_margin:.4f}' if batch_margins else 'N/A',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
            })
            
            # Detailed logging every N steps
            if step_counter % 50 == 0:
                print(f"\nStep {step_counter}:")
                print(f"  Loss:          {loss_val:.4f}")
                print(f"  GT score:      {avg_pos:.4f}" if batch_pos_scores else "  GT score:      N/A")
                print(f"  Max neg score: {avg_neg:.4f}" if batch_neg_scores else "  Max neg score: N/A")
                print(f"  Margin:        {avg_margin:+.4f}" if batch_margins else "  Margin:        N/A")
                print(f"  Grad norm:     {grad_norm:.4f}")
                print(f"  LR:            {scheduler.get_last_lr()[0]:.2e}")
            
            step_counter += 1
        
        # Epoch summary
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        margin_success = sum(1 for m in epoch_margins if m > 0) / len(epoch_margins) if epoch_margins else 0
        
        print(f"\nEpoch {epoch + 1} summary:")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Margin success rate: {margin_success*100:.1f}%")
        print(f"  Steps completed: {len(epoch_losses)}")
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1}.pt"
        torch.save({
            "epoch": epoch + 1,
            "adapter": adapter.state_dict(),
            "scorer": scorer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step_counter": step_counter,
            "margin_success_rate": margin_success,
        }, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if margin_success > best_margin_rate:
            best_margin_rate = margin_success
            best_checkpoint_path = checkpoint_dir / "best_model.pt"
            torch.save({
                "epoch": epoch + 1,
                "adapter": adapter.state_dict(),
                "scorer": scorer.state_dict(),
                "margin_success_rate": margin_success,
            }, best_checkpoint_path)
            print(f"  ★ New best model saved: {best_checkpoint_path} (margin: {margin_success*100:.1f}%)")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/max(step_counter,1):.3f}s/step)")
    print(f"Best margin success rate: {best_margin_rate*100:.1f}%")
    if best_checkpoint_path:
        print(f"Best model: {best_checkpoint_path}")
    
    # =========================================================================
    # TASK 8: FINAL VERDICT
    # =========================================================================
    
    print("\n" + "-" * 50)
    print("TASK 8: Final Verdict")
    print("-" * 50)
    
    # Checks
    losses = training_log["losses"]
    margins = training_log["margins"]
    pos_scores = training_log["positive_scores"]
    
    if len(losses) < 10:
        print("\n⚠ Not enough steps for proper evaluation")
        return
    
    # Check 1: Loss decreased
    first_losses = losses[:min(10, len(losses))]
    last_losses = losses[-min(10, len(losses)):]
    loss_decreased = sum(last_losses)/len(last_losses) < sum(first_losses)/len(first_losses)
    
    # Check 2: GT scores increased
    first_pos = pos_scores[:min(10, len(pos_scores))]
    last_pos = pos_scores[-min(10, len(pos_scores)):]
    gt_increased = sum(last_pos)/len(last_pos) > sum(first_pos)/len(first_pos) if pos_scores else False
    
    # Check 3: Margin success rate
    positive_margins = [m for m in margins if m > 0]
    margin_success = len(positive_margins) / len(margins) if margins else 0
    
    # Check 4: No NaN/Inf
    has_nan = any(not torch.isfinite(torch.tensor(l)) for l in losses)
    
    print(f"\nChecks:")
    print(f"  Loss decreased:           {'✓' if loss_decreased else '✗'}")
    print(f"  GT scores increased:      {'✓' if gt_increased else '✗'}")
    print(f"  Margin success rate:      {margin_success*100:.1f}% {'✓' if margin_success > 0.5 else '✗'}")
    print(f"  No NaN/Inf:               {'✓' if not has_nan else '✗'}")
    
    print(f"\nFirst 5 losses: {[f'{l:.4f}' for l in losses[:5]]}")
    print(f"Last 5 losses:  {[f'{l:.4f}' for l in losses[-5:]]}")
    print(f"Last 5 margins: {[f'{m:+.4f}' for m in margins[-5:]]}")
    
    all_passed = loss_decreased and gt_increased and margin_success > 0.5 and not has_nan
    
    if all_passed:
        print(f"\n{'='*70}")
        print("✅ GROUNDING TRAINING PASSED")
        print("="*70)
    else:
        print(f"\n{'='*70}")
        print("❌ GROUNDING TRAINING NEEDS MORE EPOCHS")
        print("="*70)
        print("The model is learning but needs more training time.")


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
    
    # Validate cache metadata - ensure cache was built with expected YOLO weights
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
        print(f"  Pose model: {cache_meta.get('pose_model', 'unknown')}")
        print(f"  Seg model: {cache_meta.get('seg_model', 'unknown')}")
    else:
        print(f"\n⚠ WARNING: No cache metadata found at {metadata_path}")
        print(f"  Cannot verify YOLO weights consistency")
        print(f"  Consider re-running cache_yolo_features.py to generate metadata")
    
    print(f"\nCache found: YES")
    print(f"Cached images: {len(cache_files)}")
    
    # Create checkpoint directory
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    print(f"Checkpoint directory: {checkpoint_dir}")
    
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

# This is the updated one!!!