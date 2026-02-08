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
    seed: int = 42


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
class CrossAttentionConfig:
    """Cross-attention adapter specific configuration."""
    num_heads: int = 4
    num_layers: int = 1
    dim_feedforward: int = 512
    dropout: float = 0.1


@dataclass
class HardNegativeMiningConfig:
    """
    Hard Negative Mining configuration (Phase-2).
    
    Controls difficulty-aware negative sampling to improve discrimination
    between visually similar humans.
    """
    # Master switch - when False, behaves like Phase-1
    enabled: bool = False
    
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


@dataclass
class TextEncoderConfig:
    """
    Text encoder configuration for token-level grounding.
    
    Controls tokenization behavior for Phase-3.
    """
    model_type: str = "minilm"  # "minilm" or "clip"
    max_length: int = 64  # Maximum token sequence length


@dataclass
class AugmentationConfig:
    """
    Training-time augmentation configuration.
    
    Feature-level and text-level augmentations to improve generalization.
    """
    # Feature-level augmentation
    feature_dropout: float = 0.0  # Dropout rate for visual features (0.0-0.3)
    feature_noise_std: float = 0.0  # Gaussian noise std for features (0.0-0.05)
    
    # Caption paraphrasing
    use_paraphrases: bool = False  # Enable caption paraphrasing during training
    paraphrase_prob: float = 0.3  # Probability of using paraphrase instead of original


@dataclass
class ContrastiveConfig:
    """
    Contrastive pretraining configuration (Phase-5A).
    
    InfoNCE-based alignment of text and visual embedding spaces
    before margin-based fine-tuning.
    """
    enabled: bool = False  # Enable contrastive pretraining stage
    num_epochs: int = 10  # Number of contrastive pretraining epochs
    temperature: float = 0.07  # Softmax temperature (lower = sharper)
    learning_rate: float = 1e-4  # Learning rate for contrastive stage
    use_hard_negatives: bool = True  # Use in-sample non-GT humans as hard negatives
    warmup_ratio: float = 0.1  # Warmup ratio for learning rate scheduler


@dataclass
class TextVisualAlignmentConfig:
    """
    Text-Visual Alignment adapter configuration (Phase-3).
    
    Token-level cross-modal attention for fine-grained grounding.
    """
    num_heads: int = 4
    num_layers: int = 1
    dim_feedforward: int = 512
    dropout: float = 0.1
    bidirectional: bool = True  # Enable visual→text refinement


@dataclass
class GroundingConfig:
    """
    Grounding adapter configuration.
    
    Controls the fusion mechanism between text query and visual tokens.
    Phase-1 improvement: cross_attention replaces film (baseline).
    Phase-2 improvement: hard_negative_mining for better discrimination.
    Phase-3 improvement: text_visual_alignment for token-level grounding.
    
    EXPERIMENT MODES:
        - "phase1": Cross-attention adapter (sentence-level)
        - "phase2": Cross-attention + hard negative mining
        - "phase3": Text-visual alignment (token-level)
        - "phase3_hnm": Text-visual alignment + hard negative mining
        - None: Manual configuration (use adapter_type and hard_negative_mining.enabled directly)
    """
    # Experiment mode - master switch that overrides adapter_type and HNM
    # Options: None, "phase1", "phase2", "phase3", "phase3_hnm"
    experiment_mode: Optional[str] = None
    
    # Adapter type: "cross_attention" (Phase-1), "film" (baseline), or "text_visual_alignment" (Phase-3)
    adapter_type: str = "cross_attention"
    # Text encoder settings (for token-level grounding)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    # Cross-attention specific settings (Phase-1)
    cross_attention: CrossAttentionConfig = field(default_factory=CrossAttentionConfig)
    # Text-visual alignment settings (Phase-3)
    text_visual_alignment: TextVisualAlignmentConfig = field(default_factory=TextVisualAlignmentConfig)
    # Hard negative mining settings (Phase-2)
    hard_negative_mining: HardNegativeMiningConfig = field(default_factory=HardNegativeMiningConfig)
    # Training augmentation settings
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    # Contrastive pretraining settings (Phase-5A)
    contrastive: ContrastiveConfig = field(default_factory=ContrastiveConfig)
    
    def __post_init__(self):
        """Validate and resolve experiment mode."""
        valid_types = ["cross_attention", "film", "text_visual_alignment"]
        valid_modes = [None, "phase1", "phase2", "phase3", "phase3_hnm", "phase5a"]
        
        # Validate experiment_mode if set
        if self.experiment_mode is not None and self.experiment_mode not in valid_modes:
            raise ValueError(
                f"Invalid experiment_mode: '{self.experiment_mode}'. "
                f"Must be one of: {valid_modes}"
            )
        
        # Validate adapter_type
        if self.adapter_type not in valid_types:
            raise ValueError(
                f"Invalid adapter_type: '{self.adapter_type}'. "
                f"Must be one of: {valid_types}"
            )
    
    def resolve_experiment_mode(self) -> tuple:
        """
        Resolve experiment mode to concrete settings.
        
        Returns:
            (adapter_type, hnm_enabled, mode_description)
        """
        if self.experiment_mode is None:
            # Manual mode - use explicit settings
            return (
                self.adapter_type,
                self.hard_negative_mining.enabled,
                f"Manual ({self.adapter_type}, HNM={'ON' if self.hard_negative_mining.enabled else 'OFF'})"
            )
        
        mode_mapping = {
            "phase1": ("cross_attention", False, "Phase-1 (Cross-Attention)"),
            "phase2": ("cross_attention", True, "Phase-2 (Cross-Attention + Hard Negative Mining)"),
            "phase3": ("text_visual_alignment", False, "Phase-3 (Text-Visual Token Alignment)"),
            "phase3_hnm": ("text_visual_alignment", True, "Phase-3 + Phase-2 (Token Alignment + HNM)"),
            "phase5a": ("text_visual_alignment", True, "Phase-5A (Contrastive Pretrain + Token Alignment + HNM)"),
        }
        
        return mode_mapping[self.experiment_mode]


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
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    
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
    
    # Handle optional grounding section (with defaults)
    if "grounding" in raw and raw["grounding"] is not None:
        grounding_raw = raw["grounding"]
        # Parse cross_attention nested config if present (Phase-1)
        if "cross_attention" in grounding_raw and grounding_raw["cross_attention"] is not None:
            ca_config = CrossAttentionConfig(**grounding_raw["cross_attention"])
        else:
            ca_config = CrossAttentionConfig()
        # Parse text_visual_alignment nested config if present (Phase-3)
        if "text_visual_alignment" in grounding_raw and grounding_raw["text_visual_alignment"] is not None:
            tva_config = TextVisualAlignmentConfig(**grounding_raw["text_visual_alignment"])
        else:
            tva_config = TextVisualAlignmentConfig()
        # Parse hard_negative_mining nested config if present (Phase-2)
        if "hard_negative_mining" in grounding_raw and grounding_raw["hard_negative_mining"] is not None:
            hnm_config = HardNegativeMiningConfig(**grounding_raw["hard_negative_mining"])
        else:
            hnm_config = HardNegativeMiningConfig()
        # Parse text_encoder nested config if present
        if "text_encoder" in grounding_raw and grounding_raw["text_encoder"] is not None:
            te_config = TextEncoderConfig(**grounding_raw["text_encoder"])
        else:
            te_config = TextEncoderConfig()
        grounding_config = GroundingConfig(
            experiment_mode=grounding_raw.get("experiment_mode", None),
            adapter_type=grounding_raw.get("adapter_type", "cross_attention"),
            text_encoder=te_config,
            cross_attention=ca_config,
            text_visual_alignment=tva_config,
            hard_negative_mining=hnm_config,
        )
    else:
        grounding_config = GroundingConfig()  # Use defaults
    
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
        grounding=grounding_config,
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
