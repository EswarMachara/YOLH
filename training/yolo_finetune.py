# -*- coding: utf-8 -*-
"""
YOLO Fine-Tuning Script for RefYOLO-Human

Fine-tunes large YOLO models (pose and segmentation) on the TRAIN split.
Uses the Ultralytics YOLO API for training.

CONSTRAINTS:
- ✅ CUDA only (no CPU fallback)
- ✅ Uses sample-level splits (converted to image-level for YOLO)
- ✅ Saves best checkpoints to config-specified directory
- ✅ CSV logging of per-epoch metrics
- ❌ No grounding logic
- ❌ No frozen models
- ❌ No TEST split usage (Train on TRAIN, validate on VAL only)

OUTPUT:
    checkpoints/yolo_finetuned/
        ├── pose_best.pt
        ├── seg_best.pt
        └── dataset/
            ├── images/
            ├── labels_pose/
            └── labels_seg/
    
    outputs/logs/
        ├── yolo_pose_metrics.csv
        └── yolo_seg_metrics.csv

USAGE:
    python training/yolo_finetune.py --config config/config.yaml
"""

import sys
import argparse
import shutil
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from ultralytics import YOLO

from core.config import load_config, add_config_argument, Config
from data.yolo_dataset_builder import build_yolo_dataset


def assert_cuda_available():
    """Assert CUDA is available - no CPU fallback allowed."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available! YOLO fine-tuning requires GPU.\n"
            "Please run on a CUDA-enabled system."
        )
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def download_model_if_needed(model_name: str, model_path: Path) -> Path:
    """
    Download YOLO model if not present locally.
    
    Args:
        model_name: Model name (e.g., 'yolo11l-pose.pt')
        model_path: Expected local path
        
    Returns:
        Path to model weights
    """
    if model_path.exists():
        print(f"✓ Found local model: {model_path}")
        return model_path
    
    # Try to download via Ultralytics
    print(f"  Downloading {model_name}...")
    try:
        model = YOLO(model_name)
        # Model weights are cached by Ultralytics
        print(f"✓ Downloaded {model_name}")
        return Path(model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to download {model_name}: {e}")


def parse_yolo_results(results_dir: Path) -> Dict[str, Any]:
    """
    Parse YOLO training results from CSV.
    
    Args:
        results_dir: Path to YOLO run directory
        
    Returns:
        Dict with per-epoch metrics
    """
    results_csv = results_dir / "results.csv"
    if not results_csv.exists():
        return {}
    
    epochs_data = []
    with open(results_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Clean up column names (YOLO adds spaces)
            cleaned = {k.strip(): v.strip() if isinstance(v, str) else v for k, v in row.items()}
            epochs_data.append(cleaned)
    
    return epochs_data


def save_yolo_metrics_csv(
    epochs_data: list,
    output_path: Path,
    model_type: str,  # 'pose' or 'seg'
):
    """
    Save YOLO training metrics to standardized CSV.
    
    Args:
        epochs_data: List of epoch dictionaries from YOLO results
        output_path: Path to save CSV
        model_type: 'pose' or 'seg'
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    columns = [
        "epoch",
        "train_loss",
        "val_loss",
        "mAP50_box",
        "mAP50_mask",
        "OKS_pose",
        "timestamp",
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        
        for i, data in enumerate(epochs_data):
            row = {
                "epoch": i + 1,
                "train_loss": data.get("train/box_loss", data.get("train/cls_loss", "")),
                "val_loss": data.get("val/box_loss", data.get("val/cls_loss", "")),
                "mAP50_box": data.get("metrics/mAP50(B)", ""),
                "mAP50_mask": data.get("metrics/mAP50(M)", "") if model_type == "seg" else "",
                "OKS_pose": data.get("metrics/mAP50(P)", "") if model_type == "pose" else "",
                "timestamp": datetime.now().isoformat(),
            }
            writer.writerow(row)
    
    print(f"✓ Saved {model_type} metrics to: {output_path}")


def train_pose_model(
    config: Config,
    dataset_yaml: Path,
    output_dir: Path,
) -> Path:
    """
    Fine-tune YOLO pose model.
    
    Args:
        config: Configuration object
        dataset_yaml: Path to dataset YAML
        output_dir: Output directory for checkpoints
        
    Returns:
        Path to best weights
    """
    print("\n" + "=" * 60)
    print("FINE-TUNING YOLO POSE MODEL")
    print("=" * 60)
    
    # Load base model
    base_model = config.yolo.pose_model
    print(f"  Base model: {base_model}")
    print(f"  Dataset: {dataset_yaml}")
    
    model = YOLO(base_model)
    
    # Training parameters from config
    results = model.train(
        data=str(dataset_yaml),
        epochs=config.yolo.epochs,
        batch=config.yolo.batch_size,
        imgsz=config.yolo.img_size,
        lr0=config.yolo.learning_rate,
        weight_decay=config.yolo.weight_decay,
        device=0,  # Use first GPU
        project=str(output_dir / 'runs'),
        name='pose',
        exist_ok=True,
        verbose=True,
        plots=True,
        save=True,
        save_period=5,  # Save every 5 epochs
        patience=10,  # Early stopping patience
        workers=4,
        seed=config.runtime.seed,
    )
    
    # Parse and save metrics to CSV
    run_dir = output_dir / 'runs' / 'pose'
    epochs_data = parse_yolo_results(run_dir)
    if epochs_data:
        csv_path = config.logs_dir / "yolo_pose_metrics.csv"
        save_yolo_metrics_csv(epochs_data, csv_path, "pose")
        
        # Print per-epoch summary
        print("\n  Per-epoch metrics (pose):")
        print(f"  {'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'mAP50(P)':>12}")
        print("  " + "-" * 48)
        for i, data in enumerate(epochs_data[-5:], start=max(1, len(epochs_data)-4)):
            tl = data.get("train/box_loss", "N/A")
            vl = data.get("val/box_loss", "N/A")
            mp = data.get("metrics/mAP50(P)", "N/A")
            print(f"  {i:>6} {str(tl):>12} {str(vl):>12} {str(mp):>12}")
    
    # Copy best weights to standard location
    best_weights = output_dir / 'runs' / 'pose' / 'weights' / 'best.pt'
    final_path = output_dir / 'pose_best.pt'
    
    if best_weights.exists():
        shutil.copy2(best_weights, final_path)
        print(f"✓ Pose model saved: {final_path}")
    else:
        # Fall back to last weights
        last_weights = output_dir / 'runs' / 'pose' / 'weights' / 'last.pt'
        if last_weights.exists():
            shutil.copy2(last_weights, final_path)
            print(f"✓ Pose model saved (last): {final_path}")
        else:
            raise RuntimeError("No pose model weights found after training!")
    
    return final_path


def train_seg_model(
    config: Config,
    dataset_yaml: Path,
    output_dir: Path,
) -> Path:
    """
    Fine-tune YOLO segmentation model.
    
    Args:
        config: Configuration object
        dataset_yaml: Path to dataset YAML
        output_dir: Output directory for checkpoints
        
    Returns:
        Path to best weights
    """
    print("\n" + "=" * 60)
    print("FINE-TUNING YOLO SEGMENTATION MODEL")
    print("=" * 60)
    
    # Load base model
    base_model = config.yolo.seg_model
    print(f"  Base model: {base_model}")
    print(f"  Dataset: {dataset_yaml}")
    
    model = YOLO(base_model)
    
    # Training parameters from config
    results = model.train(
        data=str(dataset_yaml),
        epochs=config.yolo.epochs,
        batch=config.yolo.batch_size,
        imgsz=config.yolo.img_size,
        lr0=config.yolo.learning_rate,
        weight_decay=config.yolo.weight_decay,
        device=0,  # Use first GPU
        project=str(output_dir / 'runs'),
        name='seg',
        exist_ok=True,
        verbose=True,
        plots=True,
        save=True,
        save_period=5,
        patience=10,
        workers=4,
        seed=config.runtime.seed,
    )
    
    # Parse and save metrics to CSV
    run_dir = output_dir / 'runs' / 'seg'
    epochs_data = parse_yolo_results(run_dir)
    if epochs_data:
        csv_path = config.logs_dir / "yolo_seg_metrics.csv"
        save_yolo_metrics_csv(epochs_data, csv_path, "seg")
        
        # Print per-epoch summary
        print("\n  Per-epoch metrics (seg):")
        print(f"  {'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'mAP50(B)':>12} {'mAP50(M)':>12}")
        print("  " + "-" * 60)
        for i, data in enumerate(epochs_data[-5:], start=max(1, len(epochs_data)-4)):
            tl = data.get("train/box_loss", "N/A")
            vl = data.get("val/box_loss", "N/A")
            mb = data.get("metrics/mAP50(B)", "N/A")
            mm = data.get("metrics/mAP50(M)", "N/A")
            print(f"  {i:>6} {str(tl):>12} {str(vl):>12} {str(mb):>12} {str(mm):>12}")
    
    # Copy best weights to standard location
    best_weights = output_dir / 'runs' / 'seg' / 'weights' / 'best.pt'
    final_path = output_dir / 'seg_best.pt'
    
    if best_weights.exists():
        shutil.copy2(best_weights, final_path)
        print(f"✓ Seg model saved: {final_path}")
    else:
        last_weights = output_dir / 'runs' / 'seg' / 'weights' / 'last.pt'
        if last_weights.exists():
            shutil.copy2(last_weights, final_path)
            print(f"✓ Seg model saved (last): {final_path}")
        else:
            raise RuntimeError("No seg model weights found after training!")
    
    return final_path


def main():
    """Main entry point for YOLO fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO pose and segmentation models"
    )
    add_config_argument(parser)
    parser.add_argument(
        "--pose-only",
        action="store_true",
        help="Train only pose model"
    )
    parser.add_argument(
        "--seg-only",
        action="store_true",
        help="Train only segmentation model"
    )
    parser.add_argument(
        "--skip-dataset-build",
        action="store_true",
        help="Skip dataset building (use existing)"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("YOLO FINE-TUNING FOR RefYOLO-Human")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load config
    config = load_config(args.config)
    
    # Validate settings
    print("\n[VALIDATION]")
    
    # Assert CUDA
    assert_cuda_available()
    
    # Assert device is cuda in config
    assert config.training.device == "cuda", \
        f"Config device must be 'cuda', got '{config.training.device}'"
    print(f"✓ Config device: {config.training.device}")
    
    # Assert fine_tune is enabled
    assert config.yolo.fine_tune, \
        "Config yolo.fine_tune must be true for fine-tuning"
    print(f"✓ Fine-tune enabled: {config.yolo.fine_tune}")
    
    # Print config
    print("\n[CONFIGURATION]")
    print(f"  Pose model: {config.yolo.pose_model}")
    print(f"  Seg model: {config.yolo.seg_model}")
    print(f"  Epochs: {config.yolo.epochs}")
    print(f"  Batch size: {config.yolo.batch_size}")
    print(f"  Image size: {config.yolo.img_size}")
    print(f"  Learning rate: {config.yolo.learning_rate}")
    print(f"  Weight decay: {config.yolo.weight_decay}")
    print(f"  Output dir: {config.yolo_output_dir}")
    
    # Create output directory
    output_dir = config.yolo_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build YOLO dataset
    dataset_dir = output_dir / 'dataset'
    
    if not args.skip_dataset_build:
        print("\n[BUILDING YOLO DATASET]")
        split_config = {
            'train': config.splits.train,
            'val': config.splits.val,
            'test': config.splits.test,
            'seed': config.splits.seed,
        }
        
        builder = build_yolo_dataset(
            coco_json_path=config.annotations_path,
            images_dir=config.images_dir,
            output_dir=dataset_dir,
            split_config=split_config,
        )
    else:
        print("\n[SKIPPING DATASET BUILD]")
        if not dataset_dir.exists():
            raise RuntimeError(f"Dataset directory not found: {dataset_dir}")
    
    # Paths to dataset YAMLs
    pose_yaml = dataset_dir / 'dataset_pose.yaml'
    seg_yaml = dataset_dir / 'dataset_seg.yaml'
    
    if not pose_yaml.exists():
        raise RuntimeError(f"Pose dataset YAML not found: {pose_yaml}")
    if not seg_yaml.exists():
        raise RuntimeError(f"Seg dataset YAML not found: {seg_yaml}")
    
    # Train models
    pose_path = None
    seg_path = None
    
    if not args.seg_only:
        pose_path = train_pose_model(config, pose_yaml, output_dir)
    
    if not args.pose_only:
        seg_path = train_seg_model(config, seg_yaml, output_dir)
    
    # Final summary
    print("\n" + "=" * 60)
    print("YOLO FINE-TUNING COMPLETE")
    print("=" * 60)
    
    if pose_path:
        print(f"  ✓ Pose weights: {pose_path}")
    if seg_path:
        print(f"  ✓ Seg weights: {seg_path}")
    
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n[NEXT STEPS]")
    print("  1. Re-cache features with fine-tuned weights:")
    print(f"     python vision/cache_yolo_features.py --config {args.config}")
    print("  2. Train grounding components:")
    print(f"     python training/grounding_train_cached.py --config {args.config}")


if __name__ == "__main__":
    main()
