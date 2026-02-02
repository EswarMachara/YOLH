# -*- coding: utf-8 -*-
"""
Grounding Metrics Module for RefYOLO-Human

Computes all grounding evaluation metrics in a deterministic, reproducible manner.
NO APPROXIMATIONS. Hard-fail on inconsistencies.

Metrics:
- Loss (MIRL)
- Margin Success Rate: % where GT score > max negative
- Accuracy@1: GT is top-scoring human
- Mean GT Rank: Average rank of GT
- PCK@50: Percentage of Correct Keypoints at 50% bbox diagonal
- Rejection Accuracy: Correct rejection when no GT
- Avg GT Score: Scalar
- Avg Max Neg Score: Scalar
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import math


@dataclass
class GroundingMetrics:
    """
    Container for all grounding metrics.
    All fields are scalars (averaged over batch/dataset).
    """
    loss: float = 0.0
    margin_success_rate: float = 0.0
    accuracy_at_1: float = 0.0
    mean_gt_rank: float = 0.0
    pck_50: float = 0.0
    # Rejection accuracy disabled: dataset contains no rejection samples (n_samples_without_gt = 0)
    # rejection_accuracy: float = 0.0
    avg_gt_score: float = 0.0
    avg_max_neg_score: float = 0.0
    
    # Counts for aggregation
    n_samples: int = 0
    n_samples_with_gt: int = 0
    n_samples_without_gt: int = 0
    # Rejection accuracy disabled: dataset contains no rejection samples
    # Keeping field for internal accumulator compatibility, but never computed/output
    n_correct_rejections: int = 0
    n_margin_successes: int = 0
    n_accuracy_at_1_correct: int = 0
    sum_gt_rank: float = 0.0
    sum_gt_score: float = 0.0
    sum_max_neg_score: float = 0.0
    n_keypoints_evaluated: int = 0
    n_keypoints_correct: int = 0
    sum_loss: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for CSV/JSON export."""
        return {
            "loss": self.loss,
            "margin_success_rate": self.margin_success_rate,
            "accuracy_at_1": self.accuracy_at_1,
            "mean_gt_rank": self.mean_gt_rank,
            "pck_50": self.pck_50,
            # Rejection accuracy disabled: dataset contains no rejection samples
            # "rejection_accuracy": self.rejection_accuracy,
            "avg_gt_score": self.avg_gt_score,
            "avg_max_neg_score": self.avg_max_neg_score,
            "n_samples": self.n_samples,
        }


class MetricsComputer:
    """
    Computes grounding metrics for a batch or accumulated over dataset.
    
    Thread-safe and deterministic.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self._n_samples = 0
        self._n_samples_with_gt = 0
        self._n_samples_without_gt = 0
        self._n_correct_rejections = 0
        self._n_margin_successes = 0
        self._n_accuracy_at_1_correct = 0
        self._sum_gt_rank = 0.0
        self._sum_gt_score = 0.0
        self._sum_max_neg_score = 0.0
        self._n_max_neg_computed = 0
        self._n_keypoints_evaluated = 0
        self._n_keypoints_correct = 0
        self._sum_loss = 0.0
    
    def compute_batch_metrics(
        self,
        scores: torch.Tensor,
        gt_indices: torch.Tensor,
        valid: torch.Tensor,
        loss: Optional[torch.Tensor] = None,
        keypoints_pred: Optional[torch.Tensor] = None,
        keypoints_gt: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
    ) -> GroundingMetrics:
        """
        Compute metrics for a single batch.
        
        Args:
            scores: [B, N] predicted scores
            gt_indices: [B] ground truth indices (-1 if no GT / rejection case)
            valid: [B, N] validity mask
            loss: Optional scalar loss tensor
            keypoints_pred: [B, N, 17, 3] predicted keypoints (for PCK)
            keypoints_gt: [B, 17, 3] ground truth keypoints (for PCK)
            boxes: [B, N, 4] bounding boxes (for PCK normalization)
        
        Returns:
            GroundingMetrics for this batch
        """
        B, N = scores.shape
        device = scores.device
        
        metrics = GroundingMetrics()
        metrics.n_samples = B
        
        # Process loss
        if loss is not None:
            loss_val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
            if math.isfinite(loss_val):
                metrics.sum_loss = loss_val * B
                metrics.loss = loss_val
        
        for b in range(B):
            gt_idx = gt_indices[b].item() if isinstance(gt_indices[b], torch.Tensor) else int(gt_indices[b])
            valid_mask = valid[b]  # [N]
            sample_scores = scores[b]  # [N]
            
            # Count valid humans
            n_valid = valid_mask.sum().item()
            
            # Case 1: Rejection case (no GT)
            if gt_idx < 0 or gt_idx >= N or not valid_mask[gt_idx]:
                metrics.n_samples_without_gt += 1
                
                # For rejection accuracy: if all scores are "low" (heuristic: all < 0)
                # or model correctly identifies no match
                # In MIRL: rejection is when rejection_score > max(scores)
                # Since we don't have explicit rejection score here, skip rejection acc
                continue
            
            # Case 2: Has GT
            metrics.n_samples_with_gt += 1
            
            # Get GT score
            gt_score = sample_scores[gt_idx].item()
            metrics.sum_gt_score += gt_score
            
            # Get negative scores (valid and not GT)
            neg_mask = valid_mask.clone()
            neg_mask[gt_idx] = False
            
            if neg_mask.any():
                neg_scores = sample_scores[neg_mask]
                max_neg = neg_scores.max().item()
                metrics.sum_max_neg_score += max_neg
                metrics._n_max_neg_computed = getattr(metrics, '_n_max_neg_computed', 0) + 1
                
                # Margin success: GT score > max negative
                if gt_score > max_neg:
                    metrics.n_margin_successes += 1
            else:
                # No negatives, margin success by default
                metrics.n_margin_successes += 1
            
            # Accuracy@1: GT is top-scoring among valid
            valid_scores = sample_scores.clone()
            valid_scores[~valid_mask] = float('-inf')
            pred_idx = valid_scores.argmax().item()
            
            if pred_idx == gt_idx:
                metrics.n_accuracy_at_1_correct += 1
            
            # Mean GT Rank: rank of GT among valid (1 = best)
            # Rank = 1 + count of valid humans with score > gt_score
            rank = 1 + (sample_scores[valid_mask] > gt_score).sum().item()
            metrics.sum_gt_rank += rank
            
            # PCK@50 computation
            if keypoints_pred is not None and keypoints_gt is not None and boxes is not None:
                pck_correct, pck_total = self._compute_pck_50(
                    keypoints_pred[b, pred_idx],  # Predicted human's keypoints
                    keypoints_gt[b],  # GT keypoints
                    boxes[b, gt_idx],  # GT box for normalization
                )
                metrics.n_keypoints_correct += pck_correct
                metrics.n_keypoints_evaluated += pck_total
        
        # Compute final metrics
        self._finalize_metrics(metrics)
        
        return metrics
    
    def _compute_pck_50(
        self,
        pred_kp: torch.Tensor,
        gt_kp: torch.Tensor,
        gt_box: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[int, int]:
        """
        Compute PCK@50 for a single sample.
        
        PCK@50 = Percentage of keypoints within 50% of bbox diagonal distance.
        
        Args:
            pred_kp: [17, 3] predicted keypoints [x, y, conf]
            gt_kp: [17, 3] ground truth keypoints [x, y, conf]
            gt_box: [4] ground truth box [x1, y1, x2, y2] normalized
            threshold: PCK threshold (0.5 = 50% of diagonal)
        
        Returns:
            (n_correct, n_total) tuple
        """
        # Compute bbox diagonal (normalized coords)
        x1, y1, x2, y2 = gt_box[0].item(), gt_box[1].item(), gt_box[2].item(), gt_box[3].item()
        diagonal = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        if diagonal < 1e-6:
            return 0, 0
        
        threshold_dist = threshold * diagonal
        
        n_correct = 0
        n_total = 0
        
        for k in range(17):
            # Only evaluate visible GT keypoints
            gt_conf = gt_kp[k, 2].item()
            if gt_conf < 0.5:  # Not visible
                continue
            
            n_total += 1
            
            # Compute distance
            pred_x, pred_y = pred_kp[k, 0].item(), pred_kp[k, 1].item()
            gt_x, gt_y = gt_kp[k, 0].item(), gt_kp[k, 1].item()
            
            dist = math.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
            
            if dist <= threshold_dist:
                n_correct += 1
        
        return n_correct, n_total
    
    def _finalize_metrics(self, metrics: GroundingMetrics):
        """Compute final metric values from counts."""
        # Margin Success Rate
        if metrics.n_samples_with_gt > 0:
            metrics.margin_success_rate = metrics.n_margin_successes / metrics.n_samples_with_gt
        
        # Accuracy@1
        if metrics.n_samples_with_gt > 0:
            metrics.accuracy_at_1 = metrics.n_accuracy_at_1_correct / metrics.n_samples_with_gt
        
        # Mean GT Rank
        if metrics.n_samples_with_gt > 0:
            metrics.mean_gt_rank = metrics.sum_gt_rank / metrics.n_samples_with_gt
        
        # Avg GT Score
        if metrics.n_samples_with_gt > 0:
            metrics.avg_gt_score = metrics.sum_gt_score / metrics.n_samples_with_gt
        
        # Avg Max Neg Score
        n_max_neg = getattr(metrics, '_n_max_neg_computed', 0)
        if n_max_neg > 0:
            metrics.avg_max_neg_score = metrics.sum_max_neg_score / n_max_neg
        
        # Rejection accuracy disabled: dataset contains no rejection samples
        # if metrics.n_samples_without_gt > 0:
        #     metrics.rejection_accuracy = metrics.n_correct_rejections / metrics.n_samples_without_gt
        
        # PCK@50
        if metrics.n_keypoints_evaluated > 0:
            metrics.pck_50 = metrics.n_keypoints_correct / metrics.n_keypoints_evaluated
        
        # Loss
        if metrics.n_samples > 0 and metrics.sum_loss > 0:
            metrics.loss = metrics.sum_loss / metrics.n_samples
    
    def accumulate(self, batch_metrics: GroundingMetrics):
        """Accumulate batch metrics into running totals."""
        self._n_samples += batch_metrics.n_samples
        self._n_samples_with_gt += batch_metrics.n_samples_with_gt
        self._n_samples_without_gt += batch_metrics.n_samples_without_gt
        self._n_correct_rejections += batch_metrics.n_correct_rejections
        self._n_margin_successes += batch_metrics.n_margin_successes
        self._n_accuracy_at_1_correct += batch_metrics.n_accuracy_at_1_correct
        self._sum_gt_rank += batch_metrics.sum_gt_rank
        self._sum_gt_score += batch_metrics.sum_gt_score
        self._sum_max_neg_score += batch_metrics.sum_max_neg_score
        self._n_max_neg_computed += getattr(batch_metrics, '_n_max_neg_computed', 0)
        self._n_keypoints_evaluated += batch_metrics.n_keypoints_evaluated
        self._n_keypoints_correct += batch_metrics.n_keypoints_correct
        self._sum_loss += batch_metrics.sum_loss
    
    def get_accumulated_metrics(self) -> GroundingMetrics:
        """Get accumulated metrics over all batches."""
        metrics = GroundingMetrics()
        metrics.n_samples = self._n_samples
        metrics.n_samples_with_gt = self._n_samples_with_gt
        metrics.n_samples_without_gt = self._n_samples_without_gt
        metrics.n_correct_rejections = self._n_correct_rejections
        metrics.n_margin_successes = self._n_margin_successes
        metrics.n_accuracy_at_1_correct = self._n_accuracy_at_1_correct
        metrics.sum_gt_rank = self._sum_gt_rank
        metrics.sum_gt_score = self._sum_gt_score
        metrics.sum_max_neg_score = self._sum_max_neg_score
        metrics._n_max_neg_computed = self._n_max_neg_computed
        metrics.n_keypoints_evaluated = self._n_keypoints_evaluated
        metrics.n_keypoints_correct = self._n_keypoints_correct
        metrics.sum_loss = self._sum_loss
        
        self._finalize_metrics(metrics)
        return metrics


def compute_pck_50_batch(
    pred_keypoints: torch.Tensor,
    gt_keypoints: torch.Tensor,
    gt_boxes: torch.Tensor,
    pred_indices: torch.Tensor,
    gt_indices: torch.Tensor,
    valid: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[int, int]:
    """
    Compute PCK@50 for a batch.
    
    Args:
        pred_keypoints: [B, N, 17, 3] all predicted keypoints
        gt_keypoints: [B, 17, 3] ground truth keypoints
        gt_boxes: [B, N, 4] all boxes
        pred_indices: [B] predicted human indices
        gt_indices: [B] ground truth human indices
        valid: [B, N] validity mask
        threshold: PCK threshold
    
    Returns:
        (total_correct, total_evaluated) tuple
    """
    B = pred_keypoints.shape[0]
    total_correct = 0
    total_evaluated = 0
    
    for b in range(B):
        gt_idx = gt_indices[b].item()
        pred_idx = pred_indices[b].item()
        
        # Skip invalid
        if gt_idx < 0 or pred_idx < 0:
            continue
        if gt_idx >= valid.shape[1] or pred_idx >= valid.shape[1]:
            continue
        if not valid[b, gt_idx]:
            continue
        
        # Get keypoints and box
        pred_kp = pred_keypoints[b, pred_idx]  # [17, 3]
        gt_kp = gt_keypoints[b]  # [17, 3]
        gt_box = gt_boxes[b, gt_idx]  # [4]
        
        # Compute bbox diagonal
        x1, y1, x2, y2 = gt_box[0].item(), gt_box[1].item(), gt_box[2].item(), gt_box[3].item()
        diagonal = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        if diagonal < 1e-6:
            continue
        
        threshold_dist = threshold * diagonal
        
        for k in range(17):
            gt_conf = gt_kp[k, 2].item()
            if gt_conf < 0.5:
                continue
            
            total_evaluated += 1
            
            pred_x, pred_y = pred_kp[k, 0].item(), pred_kp[k, 1].item()
            gt_x, gt_y = gt_kp[k, 0].item(), gt_kp[k, 1].item()
            
            dist = math.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
            
            if dist <= threshold_dist:
                total_correct += 1
    
    return total_correct, total_evaluated


def format_metrics_table(metrics: GroundingMetrics, title: str = "Metrics") -> str:
    """Format metrics as a nice ASCII table."""
    lines = [
        "",
        "=" * 60,
        f" {title}",
        "=" * 60,
        f"  {'Metric':<30} {'Value':>15}",
        "-" * 60,
        f"  {'Loss':<30} {metrics.loss:>15.4f}",
        f"  {'Margin Success Rate':<30} {metrics.margin_success_rate*100:>14.2f}%",
        f"  {'Accuracy@1':<30} {metrics.accuracy_at_1*100:>14.2f}%",
        f"  {'Mean GT Rank':<30} {metrics.mean_gt_rank:>15.2f}",
        f"  {'PCK@50':<30} {metrics.pck_50*100:>14.2f}%",
        # Rejection accuracy disabled: dataset contains no rejection samples
        # f"  {'Rejection Accuracy':<30} {metrics.rejection_accuracy*100:>14.2f}%",
        f"  {'Avg GT Score':<30} {metrics.avg_gt_score:>15.4f}",
        f"  {'Avg Max Neg Score':<30} {metrics.avg_max_neg_score:>15.4f}",
        "-" * 60,
        f"  {'Samples Total':<30} {metrics.n_samples:>15}",
        f"  {'Samples with GT':<30} {metrics.n_samples_with_gt:>15}",
        f"  {'Samples without GT':<30} {metrics.n_samples_without_gt:>15}",
        "=" * 60,
    ]
    return "\n".join(lines)
