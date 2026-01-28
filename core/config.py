# -*- coding: utf-8 -*-
"""
Configuration Loader for RefYOLO-Human

Loads and validates YAML configuration files.
Provides typed access to configuration values.

Usage:
    from core.config import load_config, Config
    
    config = load_config("config/config.yaml")
    print(config.dataset.images_dir)
    print(config.training.device)
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, Dict

import yaml


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class DatasetConfig:
    """Dataset paths configuration."""
    images_dir: str
    annotations_path: str


@dataclass
class CacheConfig:
    """Cache directory configuration."""
    features_dir: str


@dataclass
class ModelsConfig:
    """Model paths configuration."""
    pose_model: str
    seg_model: str


@dataclass
class TrainingConfig:
    """Training settings configuration."""
    device: str
    batch_size: int
    num_epochs: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    max_steps_per_epoch: Optional[int] = None


@dataclass
class CheckpointsConfig:
    """Checkpoint settings configuration."""
    save_dir: str
    save_every_epochs: int
    keep_best: bool


@dataclass
class RuntimeConfig:
    """Runtime settings configuration."""
    seed: int
    deterministic: bool
    mixed_precision: bool


@dataclass
class SplitsConfig:
    """Sample-level train/val/test split configuration."""
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
    seed: int = 42
    
    def __post_init__(self):
        """Validate split ratios sum to 1.0."""
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.4f} "
                f"(train={self.train}, val={self.val}, test={self.test})"
            )


@dataclass
class YOLOConfig:
    """YOLO fine-tuning configuration."""
    pose_model: str = "yolo11l-pose.pt"
    seg_model: str = "yolo11l-seg.pt"
    fine_tune: bool = True
    epochs: int = 50
    batch_size: int = 16
    img_size: int = 640
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    output_dir: str = "checkpoints/yolo_finetuned"
    pose_finetuned: str = "checkpoints/yolo_finetuned/pose_best.pt"
    seg_finetuned: str = "checkpoints/yolo_finetuned/seg_best.pt"


@dataclass
class OutputsConfig:
    """Output directories configuration."""
    base_dir: str = "outputs"
    logs_dir: str = "outputs/logs"
    visualizations_dir: str = "outputs/visualizations"
    plots_dir: str = "outputs/plots"
    evaluation_dir: str = "outputs/evaluation"


@dataclass
class Config:
    """
    Main configuration container.
    
    Provides typed access to all configuration sections.
    """
    dataset: DatasetConfig
    cache: CacheConfig
    models: ModelsConfig
    training: TrainingConfig
    checkpoints: CheckpointsConfig
    runtime: RuntimeConfig
    splits: SplitsConfig = field(default_factory=SplitsConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)
    
    # Project root for resolving relative paths
    project_root: Path = field(default_factory=lambda: Path.cwd())
    
    def resolve_path(self, relative_path: str) -> Path:
        """Resolve a relative path against project root."""
        return self.project_root / relative_path
    
    @property
    def images_dir(self) -> Path:
        """Absolute path to images directory."""
        return self.resolve_path(self.dataset.images_dir)
    
    @property
    def annotations_path(self) -> Path:
        """Absolute path to annotations file."""
        return self.resolve_path(self.dataset.annotations_path)
    
    @property
    def features_dir(self) -> Path:
        """Absolute path to features cache directory."""
        return self.resolve_path(self.cache.features_dir)
    
    @property
    def pose_model_path(self) -> Path:
        """Absolute path to pose model."""
        return self.resolve_path(self.models.pose_model)
    
    @property
    def seg_model_path(self) -> Path:
        """Absolute path to segmentation model."""
        return self.resolve_path(self.models.seg_model)
    
    @property
    def checkpoint_dir(self) -> Path:
        """Absolute path to checkpoint directory."""
        return self.resolve_path(self.checkpoints.save_dir)
    
    @property
    def yolo_output_dir(self) -> Path:
        """Absolute path to YOLO fine-tuned output directory."""
        return self.resolve_path(self.yolo.output_dir)
    
    @property
    def yolo_pose_model_path(self) -> Path:
        """Absolute path to YOLO pose model (base or fine-tuned)."""
        if self.yolo.fine_tune:
            return self.resolve_path(self.yolo.pose_finetuned)
        return self.resolve_path(self.yolo.pose_model)
    
    @property
    def yolo_seg_model_path(self) -> Path:
        """Absolute path to YOLO seg model (base or fine-tuned)."""
        if self.yolo.fine_tune:
            return self.resolve_path(self.yolo.seg_finetuned)
        return self.resolve_path(self.yolo.seg_model)
    
    @property
    def yolo_pose_base_path(self) -> Path:
        """Absolute path to YOLO pose BASE model (for fine-tuning)."""
        return self.resolve_path(self.yolo.pose_model)
    
    @property
    def yolo_seg_base_path(self) -> Path:
        """Absolute path to YOLO seg BASE model (for fine-tuning)."""
        return self.resolve_path(self.yolo.seg_model)
    
    @property
    def logs_dir(self) -> Path:
        """Absolute path to logs directory."""
        return self.resolve_path(self.outputs.logs_dir)
    
    @property
    def visualizations_dir(self) -> Path:
        """Absolute path to visualizations directory."""
        return self.resolve_path(self.outputs.visualizations_dir)
    
    @property
    def plots_dir(self) -> Path:
        """Absolute path to plots directory."""
        return self.resolve_path(self.outputs.plots_dir)
    
    @property
    def evaluation_dir(self) -> Path:
        """Absolute path to evaluation directory."""
        return self.resolve_path(self.outputs.evaluation_dir)


# =============================================================================
# VALIDATION
# =============================================================================

REQUIRED_FIELDS = {
    "dataset": ["images_dir", "annotations_path"],
    "cache": ["features_dir"],
    "models": ["pose_model", "seg_model"],
    "training": ["device", "batch_size", "num_epochs", "num_workers", 
                 "learning_rate", "weight_decay", "grad_clip_norm"],
    "checkpoints": ["save_dir", "save_every_epochs", "keep_best"],
    "runtime": ["seed", "deterministic", "mixed_precision"],
}


def validate_config(raw: Dict[str, Any]) -> None:
    """
    Validate that all required fields are present in config.
    
    Raises:
        ValueError: If required fields are missing.
    """
    errors = []
    
    for section, fields in REQUIRED_FIELDS.items():
        if section not in raw:
            errors.append(f"Missing section: '{section}'")
            continue
        
        section_data = raw[section]
        if section_data is None:
            errors.append(f"Section '{section}' is empty")
            continue
            
        for field_name in fields:
            if field_name not in section_data:
                errors.append(f"Missing field: '{section}.{field_name}'")
            elif section_data[field_name] is None:
                # Allow None for optional fields like max_steps_per_epoch
                if field_name not in ["max_steps_per_epoch"]:
                    errors.append(f"Field '{section}.{field_name}' is null")
    
    if errors:
        raise ValueError(
            "Configuration validation failed:\n  - " + "\n  - ".join(errors)
        )


# =============================================================================
# LOADER
# =============================================================================

def load_config(config_path: str, project_root: Optional[Path] = None) -> Config:
    """
    Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file (absolute or relative to cwd)
        project_root: Optional project root for resolving paths.
                      If None, uses the directory containing the config file.
    
    Returns:
        Config: Validated configuration object.
    
    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config is invalid or missing required fields.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Determine project root
    if project_root is None:
        # Assume config is in config/ subdirectory of project root
        project_root = config_path.parent.parent
    
    # Load YAML
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)
    
    if raw is None:
        raise ValueError(f"Configuration file is empty: {config_path}")
    
    # Validate
    validate_config(raw)
    
    # Handle optional splits section (with defaults)
    if "splits" in raw and raw["splits"] is not None:
        splits_config = SplitsConfig(**raw["splits"])
    else:
        splits_config = SplitsConfig()  # Use defaults
    
    # Handle optional yolo section (with defaults)
    if "yolo" in raw and raw["yolo"] is not None:
        yolo_config = YOLOConfig(**raw["yolo"])
    else:
        yolo_config = YOLOConfig()  # Use defaults
    
    # Handle optional outputs section (with defaults)
    if "outputs" in raw and raw["outputs"] is not None:
        outputs_config = OutputsConfig(**raw["outputs"])
    else:
        outputs_config = OutputsConfig()  # Use defaults
    
    # Build typed config
    config = Config(
        dataset=DatasetConfig(**raw["dataset"]),
        cache=CacheConfig(**raw["cache"]),
        models=ModelsConfig(**raw["models"]),
        training=TrainingConfig(**raw["training"]),
        checkpoints=CheckpointsConfig(**raw["checkpoints"]),
        runtime=RuntimeConfig(**raw["runtime"]),
        splits=splits_config,
        yolo=yolo_config,
        outputs=outputs_config,
        project_root=project_root,
    )
    
    return config


def get_default_config_path() -> Path:
    """Get the default config path relative to this file's location."""
    # Assumes core/config.py is in project_root/core/
    core_dir = Path(__file__).parent
    project_root = core_dir.parent
    return project_root / "config" / "config.yaml"


# =============================================================================
# CLI HELPER
# =============================================================================

def add_config_argument(parser) -> None:
    """
    Add --config argument to an argparse parser.
    
    Args:
        parser: argparse.ArgumentParser instance
    """
    parser.add_argument(
        "--config",
        type=str,
        default=str(get_default_config_path()),
        help="Path to configuration YAML file"
    )


def load_config_from_args(args) -> Config:
    """
    Load config from parsed argparse arguments.
    
    Args:
        args: Parsed argparse namespace with 'config' attribute
    
    Returns:
        Config: Loaded configuration
    """
    return load_config(args.config)


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    # Test config loading
    import argparse
    
    parser = argparse.ArgumentParser(description="Test config loading")
    add_config_argument(parser)
    args = parser.parse_args()
    
    print(f"Loading config from: {args.config}")
    
    try:
        config = load_config(args.config)
        
        print("\n" + "=" * 50)
        print("CONFIGURATION LOADED SUCCESSFULLY")
        print("=" * 50)
        
        print(f"\nProject root: {config.project_root}")
        
        print(f"\n[Dataset]")
        print(f"  images_dir: {config.images_dir}")
        print(f"  annotations_path: {config.annotations_path}")
        
        print(f"\n[Cache]")
        print(f"  features_dir: {config.features_dir}")
        
        print(f"\n[Models]")
        print(f"  pose_model: {config.pose_model_path}")
        print(f"  seg_model: {config.seg_model_path}")
        
        print(f"\n[Training]")
        print(f"  device: {config.training.device}")
        print(f"  batch_size: {config.training.batch_size}")
        print(f"  num_epochs: {config.training.num_epochs}")
        print(f"  learning_rate: {config.training.learning_rate}")
        
        print(f"\n[Runtime]")
        print(f"  seed: {config.runtime.seed}")
        print(f"  deterministic: {config.runtime.deterministic}")
        
        # Verify paths exist
        print(f"\n[Path Verification]")
        print(f"  images_dir exists: {config.images_dir.exists()}")
        print(f"  annotations exists: {config.annotations_path.exists()}")
        print(f"  features_dir exists: {config.features_dir.exists()}")
        print(f"  pose_model exists: {config.pose_model_path.exists()}")
        print(f"  seg_model exists: {config.seg_model_path.exists()}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
