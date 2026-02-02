# -*- coding: utf-8 -*-
"""
Hard Negative Mining Module for RefYOLO-Human (Phase-2)

Implements difficulty-aware negative sampling to improve grounding performance
by focusing on hard negatives (visually similar humans that are NOT the target).

MOTIVATION:
- Baseline shows Mean GT Rank ≈ 2.6
- GT and max-negative scores are often very close
- Most failures are due to hard negatives (similar appearance), not weak features
- Solution: Focus training on distinguishing hard negatives

DIFFICULTY SCORING:
For each negative human relative to the GT human:
    difficulty = w_iou * IoU(neg, gt) + 
                 w_pose * pose_similarity(neg, gt) + 
                 w_size * size_similarity(neg, gt)

Higher difficulty = more similar to GT = harder to distinguish

USAGE:
    from training.hard_negative_mining import HardNegativeMiner
    
    miner = HardNegativeMiner(config.grounding.hard_negative_mining)
    
    # During training:
    weights, hardness_stats = miner.compute_negative_weights(
        boxes=boxes,           # [B, N, 4]
        keypoints=keypoints,   # [B, N, 17, 3]
        gt_indices=gt_indices, # [B]
        valid=valid,           # [B, N]
        epoch=current_epoch,
        max_epochs=total_epochs,
    )
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import math


@dataclass
class HardNegativeMiningConfig:
    """Configuration for hard negative mining."""
    
    # Master switch
    enabled: bool = True
    
    # Difficulty score weights (must sum to 1.0)
    weight_iou: float = 0.5      # IoU overlap with GT
    weight_pose: float = 0.3     # Keypoint similarity
    weight_size: float = 0.2     # Bounding box size similarity
    
    # Curriculum scheduling
    curriculum_enabled: bool = True
    curriculum_start_ratio: float = 0.3   # Hard negative ratio at epoch 0
    curriculum_end_ratio: float = 0.9     # Hard negative ratio at final epoch
    curriculum_warmup_epochs: int = 5     # Linear warmup period
    
    # Mining strategy
    top_k_hard: int = 4          # Number of hardest negatives to focus on
    hard_negative_weight: float = 2.0  # Extra weight for hard negatives in loss
    
    def __post_init__(self):
        """Validate configuration."""
        total_weight = self.weight_iou + self.weight_pose + self.weight_size
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(
                f"Difficulty weights must sum to 1.0, got {total_weight:.4f} "
                f"(iou={self.weight_iou}, pose={self.weight_pose}, size={self.weight_size})"
            )
        
        if self.curriculum_start_ratio > self.curriculum_end_ratio:
            raise ValueError(
                f"curriculum_start_ratio ({self.curriculum_start_ratio}) must be <= "
                f"curriculum_end_ratio ({self.curriculum_end_ratio})"
            )
    
    def __repr__(self):
        return (
            f"HardNegativeMiningConfig(\n"
            f"  enabled={self.enabled},\n"
            f"  weights=(iou={self.weight_iou}, pose={self.weight_pose}, size={self.weight_size}),\n"
            f"  curriculum_enabled={self.curriculum_enabled},\n"
            f"  curriculum_ratio={self.curriculum_start_ratio:.1f}→{self.curriculum_end_ratio:.1f},\n"
            f"  top_k_hard={self.top_k_hard},\n"
            f"  hard_negative_weight={self.hard_negative_weight}\n"
            f")"
        )


class HardNegativeMiner:
    """
    Computes difficulty scores for negative samples and provides
    curriculum-aware sampling weights.
    
    Thread-safe and stateless (no internal state modified during training).
    """
    
    def __init__(self, config: HardNegativeMiningConfig):
        self.config = config
        self._debug = False
    
    def enable_debug(self, enable: bool = True):
        """Enable verbose debug output."""
        self._debug = enable
    
    def get_hard_ratio(self, epoch: int, max_epochs: int) -> float:
        """
        Compute current hard negative ratio based on curriculum schedule.
        
        Args:
            epoch: Current epoch (0-indexed)
            max_epochs: Total number of epochs
        
        Returns:
            Ratio of hard negatives to use (0.0 to 1.0)
        """
        if not self.config.curriculum_enabled:
            return self.config.curriculum_end_ratio
        
        if max_epochs <= 1:
            return self.config.curriculum_end_ratio
        
        warmup = self.config.curriculum_warmup_epochs
        start = self.config.curriculum_start_ratio
        end = self.config.curriculum_end_ratio
        
        if epoch < warmup:
            # Linear warmup
            progress = epoch / warmup
        else:
            # After warmup, linear increase to end
            remaining_epochs = max_epochs - warmup
            if remaining_epochs <= 0:
                progress = 1.0
            else:
                progress = (epoch - warmup) / remaining_epochs
        
        progress = min(1.0, max(0.0, progress))
        ratio = start + progress * (end - start)
        
        return ratio
    
    def compute_iou_similarity(
        self,
        boxes: torch.Tensor,
        gt_indices: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute IoU between each human and the GT human.
        
        Args:
            boxes: [B, N, 4] bounding boxes (x1, y1, x2, y2 normalized)
            gt_indices: [B] ground truth indices
            valid: [B, N] validity mask
        
        Returns:
            iou_scores: [B, N] IoU with GT for each human (0 for GT itself)
        """
        B, N, _ = boxes.shape
        device = boxes.device
        
        iou_scores = torch.zeros(B, N, device=device)
        
        for b in range(B):
            gt_idx = gt_indices[b].item()
            if gt_idx < 0 or gt_idx >= N:
                continue
            
            gt_box = boxes[b, gt_idx]  # [4]
            
            for n in range(N):
                if n == gt_idx or not valid[b, n]:
                    continue
                
                neg_box = boxes[b, n]
                
                # Compute IoU
                x1 = torch.max(gt_box[0], neg_box[0])
                y1 = torch.max(gt_box[1], neg_box[1])
                x2 = torch.min(gt_box[2], neg_box[2])
                y2 = torch.min(gt_box[3], neg_box[3])
                
                inter_w = torch.clamp(x2 - x1, min=0)
                inter_h = torch.clamp(y2 - y1, min=0)
                inter_area = inter_w * inter_h
                
                gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                neg_area = (neg_box[2] - neg_box[0]) * (neg_box[3] - neg_box[1])
                union_area = gt_area + neg_area - inter_area
                
                if union_area > 1e-6:
                    iou_scores[b, n] = inter_area / union_area
        
        return iou_scores
    
    def compute_pose_similarity(
        self,
        keypoints: torch.Tensor,
        gt_indices: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pose similarity between each human and the GT human.
        
        Uses normalized keypoint distance (OKS-like but simplified).
        
        Args:
            keypoints: [B, N, 17, 3] keypoints (x, y, conf)
            gt_indices: [B] ground truth indices
            valid: [B, N] validity mask
        
        Returns:
            pose_scores: [B, N] pose similarity (0-1, higher = more similar)
        """
        B, N, K, _ = keypoints.shape
        device = keypoints.device
        
        pose_scores = torch.zeros(B, N, device=device)
        
        for b in range(B):
            gt_idx = gt_indices[b].item()
            if gt_idx < 0 or gt_idx >= N:
                continue
            
            gt_kp = keypoints[b, gt_idx]  # [17, 3]
            gt_conf = gt_kp[:, 2]  # [17]
            gt_visible = gt_conf > 0.5
            n_visible = gt_visible.sum().item()
            
            if n_visible == 0:
                continue
            
            for n in range(N):
                if n == gt_idx or not valid[b, n]:
                    continue
                
                neg_kp = keypoints[b, n]  # [17, 3]
                
                # Compute distance for visible GT keypoints
                distances = torch.sqrt(
                    (gt_kp[:, 0] - neg_kp[:, 0]) ** 2 +
                    (gt_kp[:, 1] - neg_kp[:, 1]) ** 2
                )  # [17]
                
                # Only consider GT-visible keypoints
                visible_distances = distances[gt_visible]
                
                # Convert distance to similarity (sigmoid-like)
                # Small distance = high similarity
                # Using scale factor 0.1 (in normalized coords)
                mean_dist = visible_distances.mean()
                similarity = torch.exp(-mean_dist / 0.1)
                
                pose_scores[b, n] = similarity
        
        return pose_scores
    
    def compute_size_similarity(
        self,
        boxes: torch.Tensor,
        gt_indices: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute bounding box size similarity between each human and GT.
        
        Args:
            boxes: [B, N, 4] bounding boxes (x1, y1, x2, y2 normalized)
            gt_indices: [B] ground truth indices
            valid: [B, N] validity mask
        
        Returns:
            size_scores: [B, N] size similarity (0-1, higher = more similar)
        """
        B, N, _ = boxes.shape
        device = boxes.device
        
        size_scores = torch.zeros(B, N, device=device)
        
        for b in range(B):
            gt_idx = gt_indices[b].item()
            if gt_idx < 0 or gt_idx >= N:
                continue
            
            gt_box = boxes[b, gt_idx]
            gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
            
            if gt_area < 1e-6:
                continue
            
            for n in range(N):
                if n == gt_idx or not valid[b, n]:
                    continue
                
                neg_box = boxes[b, n]
                neg_area = (neg_box[2] - neg_box[0]) * (neg_box[3] - neg_box[1])
                
                # Size ratio (smaller / larger), gives 0-1 similarity
                if neg_area > 1e-6:
                    ratio = min(gt_area, neg_area) / max(gt_area, neg_area)
                    size_scores[b, n] = ratio
        
        return size_scores
    
    def compute_difficulty_scores(
        self,
        boxes: torch.Tensor,
        keypoints: torch.Tensor,
        gt_indices: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute overall difficulty scores for all negatives.
        
        difficulty = w_iou * IoU + w_pose * pose_sim + w_size * size_sim
        
        Args:
            boxes: [B, N, 4] bounding boxes
            keypoints: [B, N, 17, 3] keypoints
            gt_indices: [B] ground truth indices
            valid: [B, N] validity mask
        
        Returns:
            difficulty: [B, N] difficulty scores (0-1, higher = harder)
        """
        iou_sim = self.compute_iou_similarity(boxes, gt_indices, valid)
        pose_sim = self.compute_pose_similarity(keypoints, gt_indices, valid)
        size_sim = self.compute_size_similarity(boxes, gt_indices, valid)
        
        difficulty = (
            self.config.weight_iou * iou_sim +
            self.config.weight_pose * pose_sim +
            self.config.weight_size * size_sim
        )
        
        # Zero out GT position
        B, N = difficulty.shape
        for b in range(B):
            gt_idx = gt_indices[b].item()
            if 0 <= gt_idx < N:
                difficulty[b, gt_idx] = 0.0
        
        return difficulty
    
    def compute_negative_weights(
        self,
        boxes: torch.Tensor,
        keypoints: torch.Tensor,
        gt_indices: torch.Tensor,
        valid: torch.Tensor,
        epoch: int = 0,
        max_epochs: int = 50,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute per-negative weights for loss weighting.
        
        Hard negatives (top-K by difficulty) get extra weight.
        Curriculum controls how aggressively we focus on hard negatives.
        
        Args:
            boxes: [B, N, 4] bounding boxes
            keypoints: [B, N, 17, 3] keypoints
            gt_indices: [B] ground truth indices
            valid: [B, N] validity mask
            epoch: Current training epoch
            max_epochs: Total training epochs
        
        Returns:
            weights: [B, N] per-human loss weights (1.0 for GT and easy, higher for hard)
            stats: Dictionary with debugging statistics
        """
        B, N = valid.shape
        device = valid.device
        
        # Default: all weights = 1.0
        weights = torch.ones(B, N, device=device)
        
        stats = {
            'hard_ratio': 0.0,
            'mean_difficulty': 0.0,
            'max_difficulty': 0.0,
            'n_hard_negatives': 0,
            'n_total_negatives': 0,
        }
        
        if not self.config.enabled:
            return weights, stats
        
        # Compute difficulty scores
        difficulty = self.compute_difficulty_scores(boxes, keypoints, gt_indices, valid)
        
        # Get curriculum ratio
        hard_ratio = self.get_hard_ratio(epoch, max_epochs)
        stats['hard_ratio'] = hard_ratio
        
        # Collect statistics
        neg_difficulties = []
        
        for b in range(B):
            gt_idx = gt_indices[b].item()
            neg_mask = valid[b].clone()
            if 0 <= gt_idx < N:
                neg_mask[gt_idx] = False
            
            neg_indices = neg_mask.nonzero(as_tuple=True)[0]
            n_neg = len(neg_indices)
            stats['n_total_negatives'] += n_neg
            
            if n_neg == 0:
                continue
            
            # Get difficulties for this sample's negatives
            sample_difficulties = difficulty[b, neg_indices]
            neg_difficulties.extend(sample_difficulties.tolist())
            
            # Determine number of hard negatives based on curriculum
            n_hard = max(1, int(n_neg * hard_ratio))
            n_hard = min(n_hard, self.config.top_k_hard, n_neg)
            
            # Get indices of top-K hardest negatives
            _, hard_indices = sample_difficulties.topk(n_hard)
            hard_neg_indices = neg_indices[hard_indices]
            
            # Apply extra weight to hard negatives
            weights[b, hard_neg_indices] = self.config.hard_negative_weight
            stats['n_hard_negatives'] += n_hard
        
        # Compute aggregate statistics
        if neg_difficulties:
            stats['mean_difficulty'] = sum(neg_difficulties) / len(neg_difficulties)
            stats['max_difficulty'] = max(neg_difficulties)
        
        if self._debug:
            print(f"[HardNegativeMiner] epoch={epoch}, hard_ratio={hard_ratio:.2f}")
            print(f"  mean_difficulty={stats['mean_difficulty']:.4f}")
            print(f"  n_hard={stats['n_hard_negatives']}/{stats['n_total_negatives']}")
        
        return weights, stats


# =============================================================================
# WEIGHTED MIRL LOSS (Phase-2 Extension)
# =============================================================================

class WeightedMIRLLoss(nn.Module):
    """
    MIRL Loss with support for per-negative weights.
    
    Extends the base MIRL loss to support hard negative mining:
    - Hard negatives contribute more to the loss
    - Soft curriculum controls the strength of hard negative focus
    
    UNCHANGED from base MIRL:
    - Margin value (0.2)
    - Rejection loss weight
    - Basic ranking loss structure
    """
    
    def __init__(
        self,
        margin: float = 0.2,
        lambda_reject: float = 0.1,
        use_weights: bool = True,
    ):
        super().__init__()
        self.margin = margin
        self.lambda_reject = lambda_reject
        self.use_weights = use_weights
    
    def forward(
        self,
        scores: torch.Tensor,
        gt_indices: torch.Tensor,
        valid: torch.Tensor,
        negative_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted MIRL loss.
        
        Args:
            scores: [B, N] predicted scores
            gt_indices: [B] ground truth indices
            valid: [B, N] validity mask
            negative_weights: [B, N] optional per-negative weights
        
        Returns:
            Dictionary with 'total', 'ranking', 'rejection' losses
        """
        B, N = scores.shape
        device = scores.device
        
        if negative_weights is None or not self.use_weights:
            negative_weights = torch.ones(B, N, device=device)
        
        total_ranking_loss = torch.tensor(0.0, device=device)
        total_rejection_loss = torch.tensor(0.0, device=device)
        valid_samples = 0
        
        for b in range(B):
            gt_idx = gt_indices[b].item()
            valid_mask = valid[b]
            sample_scores = scores[b]
            sample_weights = negative_weights[b]
            
            # Skip invalid GT
            if gt_idx < 0 or gt_idx >= N or not valid_mask[gt_idx]:
                continue
            
            valid_samples += 1
            pos_score = sample_scores[gt_idx]
            
            # Get negative mask
            neg_mask = valid_mask.clone()
            neg_mask[gt_idx] = False
            
            if neg_mask.any():
                neg_scores = sample_scores[neg_mask]
                neg_weights_selected = sample_weights[neg_mask]
                
                # Weighted ranking loss: margin - (pos - neg), weighted by difficulty
                margins = pos_score - neg_scores
                losses = torch.relu(self.margin - margins)
                
                # Apply weights (hard negatives contribute more)
                weighted_losses = losses * neg_weights_selected
                ranking_loss = weighted_losses.mean()
                
                total_ranking_loss = total_ranking_loss + ranking_loss
        
        # Normalize
        if valid_samples > 0:
            total_ranking_loss = total_ranking_loss / valid_samples
        
        total_loss = total_ranking_loss + self.lambda_reject * total_rejection_loss
        
        return {
            "total": total_loss,
            "ranking": total_ranking_loss,
            "rejection": total_rejection_loss,
        }


# =============================================================================
# SANITY CHECK
# =============================================================================

def sanity_check():
    """Run sanity checks for hard negative mining module."""
    print("\n" + "=" * 70)
    print("Hard Negative Mining Sanity Check")
    print("=" * 70)
    
    # Create config
    config = HardNegativeMiningConfig(
        enabled=True,
        weight_iou=0.5,
        weight_pose=0.3,
        weight_size=0.2,
        curriculum_enabled=True,
        curriculum_start_ratio=0.3,
        curriculum_end_ratio=0.9,
        top_k_hard=4,
        hard_negative_weight=2.0,
    )
    print(f"\nConfig: {config}")
    
    miner = HardNegativeMiner(config)
    miner.enable_debug(True)
    
    # Test 1: Curriculum schedule
    print("\n--- Test 1: Curriculum Schedule ---")
    for epoch in [0, 2, 5, 10, 25, 49]:
        ratio = miner.get_hard_ratio(epoch, 50)
        print(f"  Epoch {epoch:2d}/50: hard_ratio = {ratio:.2f}")
    
    # Test 2: Difficulty scoring
    print("\n--- Test 2: Difficulty Scoring ---")
    B, N = 2, 6
    boxes = torch.rand(B, N, 4)
    boxes[:, :, 2:] += boxes[:, :, :2]  # Ensure x2 > x1, y2 > y1
    boxes = boxes.clamp(0, 1)  # Normalize
    
    keypoints = torch.rand(B, N, 17, 3)
    keypoints[:, :, :, 2] = torch.rand(B, N, 17)  # Confidence
    
    gt_indices = torch.tensor([0, 0])
    valid = torch.ones(B, N, dtype=torch.bool)
    
    difficulty = miner.compute_difficulty_scores(boxes, keypoints, gt_indices, valid)
    print(f"  Difficulty shape: {difficulty.shape}")
    print(f"  Difficulty[0]: {difficulty[0].tolist()}")
    print(f"  GT index difficulty (should be 0): {difficulty[0, 0].item():.4f}")
    
    # Test 3: Negative weights
    print("\n--- Test 3: Negative Weights ---")
    weights, stats = miner.compute_negative_weights(
        boxes, keypoints, gt_indices, valid, epoch=25, max_epochs=50
    )
    print(f"  Weights shape: {weights.shape}")
    print(f"  Stats: {stats}")
    print(f"  Weight range: [{weights.min():.2f}, {weights.max():.2f}]")
    
    # Test 4: Weighted MIRL Loss
    print("\n--- Test 4: Weighted MIRL Loss ---")
    loss_fn = WeightedMIRLLoss(margin=0.2, lambda_reject=0.1)
    scores = torch.randn(B, N)
    
    # Without weights
    loss_dict_no_weight = loss_fn(scores, gt_indices, valid, negative_weights=None)
    print(f"  Loss (no weights): {loss_dict_no_weight['total'].item():.4f}")
    
    # With weights
    loss_dict_weighted = loss_fn(scores, gt_indices, valid, negative_weights=weights)
    print(f"  Loss (weighted): {loss_dict_weighted['total'].item():.4f}")
    
    # Test 5: Disabled mining
    print("\n--- Test 5: Disabled Mining ---")
    config_disabled = HardNegativeMiningConfig(enabled=False)
    miner_disabled = HardNegativeMiner(config_disabled)
    weights_disabled, stats_disabled = miner_disabled.compute_negative_weights(
        boxes, keypoints, gt_indices, valid, epoch=25, max_epochs=50
    )
    assert (weights_disabled == 1.0).all(), "Disabled mining should return all 1.0 weights"
    print(f"  ✓ Disabled mining returns uniform weights")
    
    print("\n" + "=" * 70)
    print("✅ ALL SANITY CHECKS PASSED")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    sanity_check()
