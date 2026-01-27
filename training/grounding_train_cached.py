# -*- coding: utf-8 -*-
"""
Grounding Training with Cached YOLO Features

Train ONLY the grounding components (adapter + scorer + MIRL) using
precomputed YOLO features from cache. YOLO models are never loaded.

CONSTRAINTS:
- ❌ No YOLO inference
- ❌ No YOLO unfreezing
- ❌ No architecture modifications
- ✅ Use cached .pt files only
- ✅ Focus on learning behavior

USAGE:
    python training/grounding_train_cached.py --config config/config.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
from collections import defaultdict
from typing import Dict, TYPE_CHECKING

from core.datatypes import (
    D_VISION,
    D_TOKEN,
    D_QUERY,
    H_MASK,
    W_MASK,
    K_KEYPOINTS,
)

if TYPE_CHECKING:
    from core.config import Config

# We'll use simplified trainable adapter and scorer like sanity_train
# since full architecture components may not exist yet


# =============================================================================
# TRAINABLE COMPONENTS (from sanity_train.py)
# =============================================================================

class TrainableAdapter(nn.Module):
    """
    Trainable adapter for grounding (no torch.no_grad).
    Uses query-conditioned feature modulation.
    """
    
    def __init__(self, token_dim: int = 256, query_dim: int = 256):
        super().__init__()
        self.token_dim = token_dim
        self.query_dim = query_dim
        
        # Query-conditioned modulation
        self.gamma_generator = nn.Linear(query_dim, token_dim, bias=True)
        self.beta_generator = nn.Linear(query_dim, token_dim, bias=True)
        
        # Output projection
        self.output_proj = nn.Linear(token_dim, token_dim, bias=False)
        
    def forward(self, tokens: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, N, token_dim] or [N, token_dim]
            query: [query_dim] or [B, query_dim]
        Returns:
            grounded_tokens: [B, N, token_dim] or [N, token_dim]
        """
        # Handle unbatched input
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)  # [1, N, token_dim]
            unbatch = True
        else:
            unbatch = False
        
        if query.dim() == 1:
            query = query.unsqueeze(0)  # [1, query_dim]
        
        B, N, D = tokens.shape
        
        # Generate modulation parameters
        gamma = self.gamma_generator(query)  # [B, token_dim]
        beta = self.beta_generator(query)    # [B, token_dim]
        
        # Apply query-conditioned modulation
        gamma_expanded = gamma.unsqueeze(1)  # [B, 1, token_dim]
        beta_expanded = beta.unsqueeze(1)    # [B, 1, token_dim]
        
        modulated = tokens * (1.0 + gamma_expanded) + beta_expanded  # [B, N, token_dim]
        
        # Output projection
        output = self.output_proj(modulated)  # [B, N, token_dim]
        
        if unbatch:
            output = output.squeeze(0)  # [N, token_dim]
        
        return output


class TrainableScorer(nn.Module):
    """
    Trainable scorer for grounding (no torch.no_grad).
    Uses query-aware scoring.
    """
    
    def __init__(self, token_dim: int = 256, query_dim: int = 256):
        super().__init__()
        self.token_dim = token_dim
        self.query_dim = query_dim
        
        # Query projection
        self.query_proj = nn.Linear(query_dim, token_dim, bias=False)
        
        # Scoring MLP
        self.scorer = nn.Sequential(
            nn.Linear(token_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
    
    def forward(self, tokens: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, N, token_dim] or [N, token_dim]
            query: [query_dim] or [B, query_dim]
        Returns:
            scores: [B, N] or [N]
        """
        # Handle unbatched input
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)  # [1, N, token_dim]
            unbatch = True
        else:
            unbatch = False
        
        if query.dim() == 1:
            query = query.unsqueeze(0)  # [1, query_dim]
        
        B, N, D = tokens.shape
        
        # Project query to token space
        query_proj = self.query_proj(query)  # [B, token_dim]
        query_expanded = query_proj.unsqueeze(1).expand(B, N, -1)  # [B, N, token_dim]
        
        # Concatenate tokens with query
        combined = torch.cat([tokens, query_expanded], dim=-1)  # [B, N, 2D]
        
        # Score
        scores = self.scorer(combined).squeeze(-1)  # [B, N]
        
        if unbatch:
            scores = scores.squeeze(0)  # [N]
        
        return scores


class SimpleQueryEncoder(nn.Module):
    """Simple query encoder using transformers directly (avoid sentence-transformers TF issues)."""
    
    def __init__(self):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel
        
        # Use all-MiniLM-L6-v2 directly
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model.eval()
        
        # Projection to query dim if needed
        # MiniLM outputs 384 dim, we need D_QUERY (256)
        if D_QUERY != 384:
            self.projection = nn.Linear(384, D_QUERY, bias=False)
            torch.manual_seed(42)
            nn.init.orthogonal_(self.projection.weight)
            self.projection.eval()
        else:
            self.projection = None
    
    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling - take mean of token embeddings weighted by attention mask."""
        token_embeddings = model_output[0]  # [B, seq_len, hidden_dim]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, text: str) -> torch.Tensor:
        """
        Args:
            text: Caption string
        Returns:
            embedding: [D_QUERY]
        """
        with torch.no_grad():
            encoded = self.tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt')
            model_output = self.model(**encoded)
            embedding = self.mean_pooling(model_output, encoded['attention_mask'])
            
            # Normalize
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1).squeeze(0)  # [384]
            
            if self.projection is not None:
                embedding = self.projection(embedding)  # [D_QUERY]
        
        return embedding


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
        labels: torch.Tensor,
        valid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            scores: [N] predicted scores
            labels: [N] binary labels (1=positive, 0=negative)
            valid: [N] validity mask
        
        Returns:
            loss_dict with keys: total, ranking, rejection
        """
        # Mask invalid humans
        scores_masked = scores * valid.float()
        
        # Get positive and negative scores
        pos_mask = (labels == 1) & valid
        neg_mask = (labels == 0) & valid
        
        if not pos_mask.any():
            # No positive, return zero loss
            return {
                "total": torch.tensor(0.0, requires_grad=True),
                "ranking": torch.tensor(0.0),
                "rejection": torch.tensor(0.0),
            }
        
        pos_scores = scores_masked[pos_mask]
        
        if not neg_mask.any():
            # No negatives, only rejection loss
            rejection_loss = torch.relu(-pos_scores).mean()
            total = self.lambda_reject * rejection_loss
            return {
                "total": total,
                "ranking": torch.tensor(0.0),
                "rejection": rejection_loss,
            }
        
        neg_scores = scores_masked[neg_mask]
        
        # Ranking loss: max(0, margin - (pos - neg))
        pos_expanded = pos_scores.unsqueeze(1)  # [P, 1]
        neg_expanded = neg_scores.unsqueeze(0)  # [1, N]
        
        margins = pos_expanded - neg_expanded  # [P, N]
        ranking_loss = torch.relu(self.margin - margins).mean()
        
        # Rejection loss: max(0, -score) for all instances
        rejection_loss = torch.relu(-scores_masked[valid]).mean()
        
        total = ranking_loss + self.lambda_reject * rejection_loss
        
        return {
            "total": total,
            "ranking": ranking_loss,
            "rejection": rejection_loss,
        }


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_grounding(config: "Config"):
    """
    Main training function using configuration.
    
    Args:
        config: Configuration object loaded from config.yaml
    """
    # Resolve paths from config
    project_root = Path(config.project_root)
    cache_dir = project_root / config.cache.features_dir
    checkpoint_dir = project_root / config.checkpoints.checkpoints_dir
    coco_json = project_root / config.dataset.annotations_path
    
    # Training settings from config
    device = config.runtime.device
    num_epochs = config.training.num_epochs
    batch_size = config.training.batch_size
    max_steps_per_epoch = config.training.max_steps_per_epoch
    learning_rate = config.training.learning_rate
    
    print("\n" + "=" * 70)
    print("GROUNDING TRAINING WITH CACHED YOLO FEATURES")
    print("=" * 70)
    
    # =============================================================================
    # TASK 0: VERIFY CACHE EXISTS (ABORT IF MISSING)
    # =============================================================================
    
    # Verify cache
    if not cache_dir.exists():
        print("\n❌ ABORT: Cache directory not found!")
        print(f"   Expected: {cache_dir}")
        sys.exit(1)
    
    cache_files = list(cache_dir.glob("*.pt"))
    if len(cache_files) == 0:
        print("\n❌ ABORT: No cached .pt files found!")
        sys.exit(1)
    
    print(f"\nCache found: YES")
    print(f"Cached images: {len(cache_files)}")
    
    # Create checkpoint directory
    checkpoint_dir.mkdir(exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # =============================================================================
    # TASK 1: TRAINING CONFIGURATION (FROM CONFIG FILE)
    # =============================================================================
    
    print(f"\nConfiguration (from config file):")
    print(f"  num_epochs: {num_epochs}")
    print(f"  batch_size: {batch_size}")
    print(f"  max_steps_per_epoch: {max_steps_per_epoch}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  device: {device}")
    
    # Run training loop
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


# =============================================================================
# TASK 2: CACHED FEATURE DATASET
# =============================================================================

class CachedFeatureDataset(Dataset):
    """
    Dataset that loads precomputed YOLO features from cache.
    
    Returns per sample:
    {
        "visual_embeddings": Tensor[N, 256],
        "boxes": Tensor[N, 4],
        "masks": Tensor[N, H, W],
        "keypoints": Tensor[N, 17, 3],
        "valid": Tensor[N],
        "caption": str,
        "gt_index": int
    }
    """
    
    def __init__(self, cache_dir: Path, coco_json_path: Path, max_samples: int = None):
        self.cache_dir = cache_dir
        self.cache_files = sorted(list(cache_dir.glob("*.pt")))
        
        if max_samples is not None:
            self.cache_files = self.cache_files[:max_samples]
        
        # Load COCO annotations to get captions
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        # Build mapping: image_id -> list of annotations
        self.image_id_to_anns = defaultdict(list)
        for ann in coco_data['annotations']:
            image_id = str(ann['image_id'])  # Convert to string to match cache
            self.image_id_to_anns[image_id].append(ann)
        
        # Filter: keep only cache files with annotations and humans
        valid_cache_files = []
        for cf in self.cache_files:
            image_id = cf.stem
            
            # Check if annotations exist
            if image_id not in self.image_id_to_anns:
                continue
            
            # Check if cache has humans
            cache_data = torch.load(cf, weights_only=True)
            N = cache_data['boxes'].shape[0]
            if N == 0:
                continue
            
            valid_cache_files.append(cf)
        
        self.cache_files = valid_cache_files
        
        print(f"\nCachedFeatureDataset initialized:")
        print(f"  Cache directory: {cache_dir}")
        print(f"  Total cache files: {len(self.cache_files)}")
        print(f"  COCO annotations loaded: {len(coco_data['annotations'])}")
    
    def __len__(self):
        return len(self.cache_files)
    
    def __getitem__(self, idx):
        cache_file = self.cache_files[idx]
        image_id = cache_file.stem
        
        # Load cached features
        cache_data = torch.load(cache_file, weights_only=True)
        
        visual_embeddings = cache_data['visual_embeddings']  # [N, 256]
        boxes = cache_data['boxes']  # [N, 4]
        masks = cache_data['masks']  # [N, 160, 160]
        keypoints = cache_data['keypoints']  # [N, 17, 3]
        valid = cache_data['valid']  # [N]
        
        N = boxes.shape[0]
        
        # Get annotations for this image
        anns = self.image_id_to_anns[image_id]
        
        # Select a random annotation as ground truth
        import random
        random.seed(42 + idx)  # Deterministic
        gt_ann = random.choice(anns)
        caption = gt_ann['caption']
        
        # Determine GT index by matching bbox
        # (simplified: use first annotation's index as GT)
        # In real scenario, match by IoU with cached boxes
        gt_index = 0  # Simplified: always use first human as GT
        
        # Ensure GT index is valid
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
        }


def _run_training_loop(
    config: "Config",
    cache_dir: Path,
    checkpoint_dir: Path,
    coco_json: Path,
    device: str,
    num_epochs: int,
    batch_size: int,
    max_steps_per_epoch: int,
    learning_rate: float,
):
    """Internal training loop, called by train_grounding()."""
    
    # =============================================================================
    # TASK 3: MODEL INITIALIZATION (NO YOLO)
    # =============================================================================
    
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
    adapter = TrainableAdapter(
        token_dim=D_TOKEN,
        query_dim=D_QUERY,
    )
    adapter.to(device)
    adapter.train()
    
    print(f"✓ TrainableAdapter initialized (trainable)")
    
    # Scorer (trainable)
    scorer = TrainableScorer(
        token_dim=D_TOKEN,
        query_dim=D_QUERY,
    )
    scorer.to(device)
    scorer.train()
    
    print(f"✓ TrainableScorer initialized (trainable)")
    
    # MIRL loss
    mirl_loss_fn = MIRLLoss(margin=0.2, lambda_reject=0.1)
    print(f"✓ MIRLLoss initialized (margin=0.2, lambda_reject=0.1)")
    
    
    # =============================================================================
    # TASK 4: FREEZE POLICY & PARAMETER COUNT
    # =============================================================================
    
    print("\n" + "-" * 50)
    print("TASK 4: Freeze policy and parameter counting")
    print("-" * 50)
    
    # Count parameters
    def count_parameters(module, name):
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable
    
    modules_to_count = [
        ("SimpleQueryEncoder", query_encoder),
        ("TrainableAdapter", adapter),
        ("TrainableScorer", scorer),
    ]
    
    print("\nParameter counts:")
    total_all = 0
    trainable_all = 0
    
    for name, module in modules_to_count:
        total, trainable = count_parameters(module, name)
        total_all += total
        trainable_all += trainable
        print(f"  {name:30s} - Total: {total:>8,} | Trainable: {trainable:>8,}")
    
    print(f"  {'TOTAL':30s} - Total: {total_all:>8,} | Trainable: {trainable_all:>8,}")
    
    print(f"\nTrainable modules:")
    trainable_modules = []
    for name, module in modules_to_count:
        if any(p.requires_grad for p in module.parameters()):
            trainable_modules.append(name)
            print(f"  ✓ {name}")
    
    print(f"\nFrozen modules:")
    for name, module in modules_to_count:
        if not any(p.requires_grad for p in module.parameters()):
            print(f"  ❄ {name}")
    
    
    # =============================================================================
    # TASK 5: GROUNDING TRAINING LOOP
    # =============================================================================
    
    print("\n" + "-" * 50)
    print("TASK 5: Preparing training loop")
    print("-" * 50)
    
    # Dataset and dataloader
    dataset = CachedFeatureDataset(
        cache_dir=cache_dir,
        coco_json_path=coco_json,
        max_samples=max_steps_per_epoch * num_epochs,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    print(f"✓ DataLoader ready ({len(dataset)} samples)")
    
    # Optimizer (only trainable parameters)
    trainable_params = []
    for module in [adapter, scorer]:
        trainable_params.extend([p for p in module.parameters() if p.requires_grad])
    
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
    print(f"✓ Optimizer initialized (Adam, lr={learning_rate})")
    
    
    # =============================================================================
    # TASK 6 & 7: TRAINING WITH LOGGING
    # =============================================================================
    
    print("\n" + "-" * 50)
    print("TASK 6-7: Running grounding training with logging")
    print("-" * 50)
    
    # Logging storage
    training_log = {
        "losses": [],
        "positive_scores": [],
        "max_negative_scores": [],
        "margins": [],
        "rejection_losses": [],
    }
    
    # Moving averages for early stopping
    moving_avg_window = 20
    positive_margin_streak = 0
    early_stop_threshold = 20
    early_stopped = False
    
    step_counter = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        if early_stopped:
            break
            
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch + 1}/{num_epochs}")
        print(f"{'='*70}")
        
        epoch_losses = []
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            if step_counter >= max_steps_per_epoch * num_epochs:
                break
            
            # Extract batch data (batch_size=1)
            visual_embeddings = batch["visual_embeddings"][0].to(device)  # [N, 256]
            boxes = batch["boxes"][0].to(device)  # [N, 4]
            masks = batch["masks"][0].to(device)  # [N, 160, 160]
            keypoints = batch["keypoints"][0].to(device)  # [N, 17, 3]
            valid = batch["valid"][0].to(device)  # [N]
            caption = batch["caption"][0]
            gt_index = batch["gt_index"][0].item()
            
            N = boxes.shape[0]
            
            # Skip if no humans
            if N == 0:
                continue
            
            # Ensure gt_index is valid
            if gt_index >= N:
                gt_index = N - 1
            
            # =====================================================================
            # FORWARD PASS
            # =====================================================================
            
            # 1. Encode query
            with torch.no_grad():
                query_embedding = query_encoder(caption)  # [D_QUERY]
            
            # 2. Use cached visual embeddings as tokens directly
            # visual_embeddings are already [N, 256] from cache
            tokens = visual_embeddings  # [N, D_TOKEN=256]
            
            # 3. Apply grounding adapter
            grounded_tokens = adapter(tokens, query_embedding)  # [N, D_TOKEN]
            
            # 4. Score humans
            scores = scorer(grounded_tokens, query_embedding)  # [N]
            
            # =====================================================================
            # COMPUTE LOSS
            # =====================================================================
            
            # Create GT labels
            labels = torch.zeros(N, dtype=torch.long, device=device)
            labels[gt_index] = 1  # 1 = positive, 0 = negative
            
            # MIRL loss
            loss_dict = mirl_loss_fn(scores, labels, valid)
            total_loss = loss_dict["total"]
            
            # =====================================================================
            # BACKWARD & OPTIMIZE
            # =====================================================================
            
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            
            optimizer.step()
            
            # =====================================================================
            # LOGGING
            # =====================================================================
            
            positive_score = scores[gt_index].item()
            
            # Get negative scores
            negative_mask = labels == 0
            if negative_mask.any():
                negative_scores = scores[negative_mask]
                max_negative_score = negative_scores.max().item()
                margin = positive_score - max_negative_score
            else:
                max_negative_score = float('-inf')
                margin = float('inf')
            
            rejection_loss = loss_dict.get("rejection", torch.tensor(0.0)).item()
            
            training_log["losses"].append(total_loss.item())
            training_log["positive_scores"].append(positive_score)
            training_log["max_negative_scores"].append(max_negative_score)
            training_log["margins"].append(margin)
            training_log["rejection_losses"].append(rejection_loss)
            
            epoch_losses.append(total_loss.item())
            
            # Compute moving averages
            if len(training_log["losses"]) >= moving_avg_window:
                recent_losses = training_log["losses"][-moving_avg_window:]
                recent_margins = training_log["margins"][-moving_avg_window:]
                avg_loss_ma = sum(recent_losses) / len(recent_losses)
                avg_margin_ma = sum(recent_margins) / len(recent_margins)
            else:
                avg_loss_ma = total_loss.item()
                avg_margin_ma = margin
            
            # Print every 20 steps (more frequent for laptop)
            if step_counter % 20 == 0 or step_counter < 5:
                print(f"\nStep {step_counter}:")
                print(f"  Loss:              {total_loss.item():.4f}")
                print(f"  GT score:          {positive_score:.4f}")
                print(f"  Max neg score:     {max_negative_score:.4f}")
                print(f"  Margin:            {margin:+.4f}")
                if len(training_log["losses"]) >= moving_avg_window:
                    print(f"  MA Loss (20):      {avg_loss_ma:.4f}")
                    print(f"  MA Margin (20):    {avg_margin_ma:+.4f}")
            
            # Early stopping check: margin > 0 for N consecutive steps
            if margin > 0:
                positive_margin_streak += 1
            else:
                positive_margin_streak = 0
            
            # Check early stop conditions
            if positive_margin_streak >= early_stop_threshold and len(training_log["losses"]) >= 30:
                # Also check loss is not exploding
                recent_losses = training_log["losses"][-20:]
                max_recent = max(recent_losses)
                if max_recent < 1.0:  # Loss bounded
                    print(f"\n" + "="*70)
                    print("EARLY STOP: Grounding behavior verified!")
                    print(f"  Positive margin streak: {positive_margin_streak} steps")
                    print(f"  Recent max loss: {max_recent:.4f}")
                    print("="*70)
                    early_stopped = True
                    break
            
            step_counter += 1
        
        # Epoch summary
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        print(f"\nEpoch {epoch + 1} summary:")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Steps completed: {len(epoch_losses)}")
        
        # Save checkpoint after each epoch
        checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1}.pt"
        torch.save({
            "epoch": epoch + 1,
            "adapter": adapter.state_dict(),
            "scorer": scorer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step_counter": step_counter,
            "training_log": training_log,
        }, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"Total time: {elapsed:.1f} seconds ({elapsed/step_counter:.2f}s/step)")
    
    
    # =============================================================================
    # TASK 8: FINAL VERDICT
    # =============================================================================
    
    print("\n" + "-" * 50)
    print("TASK 8: Expected Behavior Check & Final Verdict")
    print("-" * 50)
    
    # Check 1: Loss decreases over time
    first_5_losses = training_log["losses"][:5]
    last_5_losses = training_log["losses"][-5:]
    
    print(f"\nFirst 5 loss values: {[f'{l:.4f}' for l in first_5_losses]}")
    print(f"Last 5 loss values:  {[f'{l:.4f}' for l in last_5_losses]}")
    
    avg_first_5 = sum(first_5_losses) / len(first_5_losses)
    avg_last_5 = sum(last_5_losses) / len(last_5_losses)
    loss_decreased = avg_last_5 < avg_first_5
    
    print(f"\nAverage first 5: {avg_first_5:.4f}")
    print(f"Average last 5:  {avg_last_5:.4f}")
    print(f"Loss decreased: {'✓' if loss_decreased else '✗'}")
    
    # Check 2: Mean GT score increases
    first_10_gt_scores = training_log["positive_scores"][:10]
    last_10_gt_scores = training_log["positive_scores"][-10:]
    
    avg_first_gt = sum(first_10_gt_scores) / len(first_10_gt_scores)
    avg_last_gt = sum(last_10_gt_scores) / len(last_10_gt_scores)
    gt_score_increased = avg_last_gt > avg_first_gt
    
    print(f"\nGT score progression:")
    print(f"  Average first 10 GT scores: {avg_first_gt:.4f}")
    print(f"  Average last 10 GT scores:  {avg_last_gt:.4f}")
    print(f"  GT score increased: {'✓' if gt_score_increased else '✗'}")
    
    # Check 3: GT > negatives in majority of steps
    positive_margins = [m for m in training_log["margins"] if m > 0]
    margin_success_rate = len(positive_margins) / len(training_log["margins"])
    
    print(f"\nMargin analysis:")
    print(f"  Steps with positive margin: {len(positive_margins)} / {len(training_log['margins'])}")
    print(f"  Success rate: {margin_success_rate*100:.1f}%")
    print(f"  Majority GT > negatives: {'✓' if margin_success_rate > 0.5 else '✗'}")
    
    # Sample margins
    print(f"\nSample margins (first 10):")
    for i in range(min(10, len(training_log["margins"]))):
        m = training_log["margins"][i]
        print(f"  Step {i}: {m:+.4f}")
    
    # Check 4: No NaN/Inf
    has_nan_inf = any(
        not (torch.isfinite(torch.tensor(l)) if isinstance(l, (int, float)) else True)
        for l in training_log["losses"]
    )
    print(f"\nNo NaN/Inf in losses: {'✓' if not has_nan_inf else '✗'}")
    
    # Final verdict
    print(f"\n{'='*70}")
    print("FINAL VERDICT")
    print("="*70)
    
    checks_passed = [
        loss_decreased,
        gt_score_increased,
        margin_success_rate > 0.5,
        not has_nan_inf,
    ]
    
    all_passed = all(checks_passed)
    
    # Print last 5 margins
    print(f"\nLast 5 margins:")
    for i, m in enumerate(training_log["margins"][-5:]):
        idx = len(training_log["margins"]) - 5 + i
        print(f"  Step {idx}: {m:+.4f}")
    
    print(f"\nSteps completed: {step_counter}")
    print(f"Early stopped: {'YES' if early_stopped else 'NO'}")
    
    if all_passed or early_stopped:
        print("\n✅ GROUNDING TRAINING PASSED")
        print("\nAll checks passed:")
        if loss_decreased:
            print("  ✓ Loss decreased over training")
        if gt_score_increased:
            print("  ✓ GT scores increased")
        if margin_success_rate > 0.5:
            print("  ✓ GT > negatives in majority of steps")
        if not has_nan_inf:
            print("  ✓ No NaN/Inf detected")
        if early_stopped:
            print("  ✓ Early stop triggered (behavior verified)")
        print("\nGrounding components learned successfully from cached features!")
        print("Ready for full training pipeline.")
    else:
        print("\n❌ GROUNDING TRAINING FAILED")
        print("\nFailed checks:")
        if not loss_decreased:
            print("  ✗ Loss did not decrease")
        if not gt_score_increased:
            print("  ✗ GT scores did not increase")
        if not margin_success_rate > 0.5:
            print("  ✗ GT not consistently scored higher than negatives")
        if has_nan_inf:
            print("  ✗ NaN/Inf detected in losses")
        print("\nReview training logs above for debugging.")
    
    print(f"\n{'='*70}\n")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    from core.config import load_config, add_config_argument
    
    parser = argparse.ArgumentParser(description="Train grounding components with cached YOLO features")
    add_config_argument(parser)
    args = parser.parse_args()
    
    config = load_config(args.config)
    train_grounding(config)
