# -*- coding: utf-8 -*-
"""
Grounding Evaluation Script for RefYOLO-Human

Evaluates the best grounding model on the TEST split.
This script is completely separate from training and does NOT affect training code.

FEATURES:
- ✅ Loads best grounding model
- ✅ Loads cached features
- ✅ Evaluates on TEST split ONLY
- ✅ Computes all metrics (Loss, Margin Success Rate, Accuracy@1, Mean GT Rank, PCK@50, etc.)
- ✅ Saves metrics to CSV and JSON
- ✅ Prints clear summary table

USAGE:
    python evaluation/grounding_eval.py --config config/config.yaml
    python evaluation/grounding_eval.py --config config/config.yaml --checkpoint checkpoints/best_model.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Optional

from core.config import load_config, add_config_argument, Config
from core.metrics import MetricsComputer, GroundingMetrics, format_metrics_table
from core.logging import CSVLogger, JSONLogger
from core.datatypes import D_TOKEN, D_QUERY

# Import from training module (only model definitions, no training logic)
from training.grounding_train_v2 import (
    TrainableAdapter,
    TrainableScorer,
    SimpleQueryEncoder,
    MIRLLoss,
    CachedFeatureDataset,
    collate_variable_humans,
)
from adapter.cross_attention_adapter import CrossAttentionAdapter, create_grounding_adapter


def load_checkpoint(checkpoint_path: Path, device: str) -> Dict:
    """
    Load checkpoint and validate contents.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load to
        
    Returns:
        Checkpoint dictionary
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Validate required keys
    required_keys = ["adapter", "scorer"]
    for key in required_keys:
        if key not in checkpoint:
            raise ValueError(f"Checkpoint missing required key: {key}")
    
    # Print checkpoint info
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    if 'val_margin_success_rate' in checkpoint:
        print(f"  Val Margin Success Rate: {checkpoint['val_margin_success_rate']*100:.2f}%")
    if 'val_metrics' in checkpoint:
        print(f"  Val Metrics: {checkpoint['val_metrics']}")
    
    return checkpoint


@torch.no_grad()
def evaluate_test_split(
    adapter: nn.Module,
    scorer: nn.Module,
    query_encoder: nn.Module,
    mirl_loss_fn: nn.Module,
    test_dataloader: DataLoader,
    device: str,
) -> GroundingMetrics:
    """
    Evaluate on TEST split and compute all metrics.
    
    Args:
        adapter: Grounding adapter module
        scorer: Scoring module
        query_encoder: Query encoder module
        mirl_loss_fn: MIRL loss function
        test_dataloader: Test data loader
        device: Device string
    
    Returns:
        GroundingMetrics for test set
    """
    adapter.eval()
    scorer.eval()
    
    metrics_computer = MetricsComputer()
    
    print("\nEvaluating on TEST split...")
    
    for batch in tqdm(test_dataloader, desc="Evaluating"):
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
        
        # Forward pass
        query_embeddings = query_encoder.forward_batch(captions)
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
            keypoints_gt=keypoints[torch.arange(B, device=device), gt_indices.clamp(0, N-1)],
            boxes=boxes,
        )
        
        metrics_computer.accumulate(batch_metrics)
    
    return metrics_computer.get_accumulated_metrics()


def save_results(
    metrics: GroundingMetrics,
    output_dir: Path,
    checkpoint_path: Path,
    config: Config,
):
    """
    Save evaluation results to CSV and JSON.
    
    Args:
        metrics: Computed metrics
        output_dir: Output directory
        checkpoint_path: Path to evaluated checkpoint
        config: Configuration object
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    # Note: rejection_accuracy removed - dataset contains no rejection samples
    csv_path = output_dir / "metrics_test.csv"
    csv_logger = CSVLogger(csv_path, overwrite=True)
    csv_logger.log({
        "epoch": "test",
        "loss": metrics.loss,
        "margin_success_rate": metrics.margin_success_rate,
        "accuracy_at_1": metrics.accuracy_at_1,
        "mean_gt_rank": metrics.mean_gt_rank,
        "pck_50": metrics.pck_50,
        "avg_gt_score": metrics.avg_gt_score,
        "avg_max_neg_score": metrics.avg_max_neg_score,
    })
    print(f"✓ CSV saved: {csv_path}")
    
    # Save JSON with full details
    json_path = output_dir / "metrics_test.json"
    json_logger = JSONLogger(json_path)
    json_logger.save({
        "metrics": metrics.to_dict(),
        "checkpoint": str(checkpoint_path),
        "config": {
            "splits": {
                "train": config.splits.train,
                "val": config.splits.val,
                "test": config.splits.test,
                "seed": config.splits.seed,
            },
            "yolo_fine_tuned": config.yolo.fine_tune,
        },
        "evaluation_details": {
            "n_samples": metrics.n_samples,
            "n_samples_with_gt": metrics.n_samples_with_gt,
            "n_samples_without_gt": metrics.n_samples_without_gt,
            "n_margin_successes": metrics.n_margin_successes,
            "n_accuracy_at_1_correct": metrics.n_accuracy_at_1_correct,
            "n_keypoints_evaluated": metrics.n_keypoints_evaluated,
            "n_keypoints_correct": metrics.n_keypoints_correct,
        },
    })
    print(f"✓ JSON saved: {json_path}")


def print_final_summary(metrics: GroundingMetrics, checkpoint_path: Path):
    """Print final summary table."""
    print("\n" + "=" * 70)
    print(" FINAL TEST EVALUATION RESULTS")
    print("=" * 70)
    print(f"\n  Checkpoint: {checkpoint_path}")
    print(format_metrics_table(metrics, "TEST METRICS"))
    
    # Detailed breakdown
    print("\n  DETAILED BREAKDOWN:")
    print(f"  {'='*50}")
    print(f"  Total test samples:         {metrics.n_samples:>10}")
    print(f"  Samples with GT:            {metrics.n_samples_with_gt:>10}")
    print(f"  Samples without GT:         {metrics.n_samples_without_gt:>10}")
    print(f"  Margin successes:           {metrics.n_margin_successes:>10}")
    print(f"  Accuracy@1 correct:         {metrics.n_accuracy_at_1_correct:>10}")
    print(f"  Keypoints evaluated:        {metrics.n_keypoints_evaluated:>10}")
    print(f"  Keypoints correct (PCK@50): {metrics.n_keypoints_correct:>10}")
    print(f"  {'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate grounding model on TEST split")
    add_config_argument(parser)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (default: checkpoints/best_model.pt)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for evaluation (default: from config)"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("GROUNDING MODEL EVALUATION - TEST SPLIT")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load config
    config = load_config(args.config)
    
    # Determine device
    device = config.training.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"
    
    print(f"\nDevice: {device}")
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = config.checkpoint_dir / "best_model.pt"
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, device)
    
    # Initialize models
    print("\nInitializing models...")
    
    query_encoder = SimpleQueryEncoder()
    query_encoder.to(device)
    query_encoder.eval()
    print("✓ Query encoder loaded")
    
    # Create adapter based on config (must match training configuration)
    adapter_type = config.grounding.adapter_type
    if adapter_type == "cross_attention":
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
        print(f"✓ CrossAttentionAdapter created (matching config)")
    else:
        adapter = TrainableAdapter(token_dim=D_TOKEN, query_dim=D_QUERY)
        print(f"✓ TrainableAdapter (FiLM) created")
    
    adapter.load_state_dict(checkpoint["adapter"])
    adapter.to(device)
    adapter.eval()
    print("✓ Adapter weights loaded from checkpoint")
    
    scorer = TrainableScorer(token_dim=D_TOKEN, query_dim=D_QUERY)
    scorer.load_state_dict(checkpoint["scorer"])
    scorer.to(device)
    scorer.eval()
    print("✓ Scorer loaded from checkpoint")
    
    mirl_loss_fn = MIRLLoss(margin=0.2, lambda_reject=0.1)
    
    # Load TEST dataset
    print("\nLoading TEST dataset...")
    
    cache_dir = config.features_dir
    coco_json = config.annotations_path
    
    if not cache_dir.exists():
        print(f"\n❌ ABORT: Cache directory not found: {cache_dir}")
        sys.exit(1)
    
    split_config = {
        'train': config.splits.train,
        'val': config.splits.val,
        'test': config.splits.test,
        'seed': config.splits.seed,
    }
    
    test_dataset = CachedFeatureDataset(
        cache_dir=cache_dir,
        coco_json_path=coco_json,
        split="test",  # IMPORTANT: Only test split
        split_config=split_config,
        max_samples=None,
        seed=config.runtime.seed,
    )
    
    batch_size = args.batch_size or config.training.batch_size
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_variable_humans,
        drop_last=False,
    )
    
    print(f"✓ Test dataset loaded: {len(test_dataset)} samples")
    
    # Evaluate
    metrics = evaluate_test_split(
        adapter=adapter,
        scorer=scorer,
        query_encoder=query_encoder,
        mirl_loss_fn=mirl_loss_fn,
        test_dataloader=test_dataloader,
        device=device,
    )
    
    # Save results
    output_dir = config.evaluation_dir
    save_results(metrics, output_dir, checkpoint_path, config)
    
    # Print final summary
    print_final_summary(metrics, checkpoint_path)
    
    print(f"\nEvaluation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return metrics


if __name__ == "__main__":
    main()
