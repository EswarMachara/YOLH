# -*- coding: utf-8 -*-
"""
YOLO Fine-Tuning Script for RefYOLO-Human

Fine-tunes large YOLO models (pose and segmentation) on the TRAIN split.
Uses the Ultralytics YOLO API for training.

CONSTRAINTS:
- ✅ CUDA only (no CPU fallback)
- ✅ Uses sample-level splits (converted to image-level for YOLO)
- ✅ Saves best checkpoints to config-specified directory
- ❌ No grounding logic
- ❌ No frozen models

OUTPUT:
    checkpoints/yolo_finetuned/
        ├── pose_best.pt
        ├── seg_best.pt
        └── dataset/
            ├── images/
            ├── labels_pose/
            └── labels_seg/

USAGE:
    python training/yolo_finetune.py --config config/config.yaml
"""

import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime

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
