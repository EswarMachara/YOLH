# -*- coding: utf-8 -*-
"""
YOLO Feature Caching Script for RefYOLO-Human Training

Precomputes and caches YOLO-based human features for training the grounding
components (adapter + scorer + MIRL loss) without running YOLO in the loop.

Cache Format (per image):
{
    "image_id": int,
    "file_name": str,
    "boxes": Tensor[N, 4],           # normalized xyxy [0,1]
    "masks": Tensor[N, 160, 160],    # resized segmentation masks
    "keypoints": Tensor[N, 17, 3],   # normalized keypoints
    "visual_embeddings": Tensor[N, 256],  # ROI Align features
    "valid": Tensor[N]               # validity flags (all True)
}

Constraints:
- ❌ No training, no unfreezing YOLO
- ✅ YOLO in eval mode, torch.no_grad()
- ✅ Output matches HumanToken/VisionOutput contract inputs

Usage:
    python vision/cache_yolo_features.py --config config/config.yaml
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
from ultralytics import YOLO
from tqdm import tqdm
import time

from core.datatypes import D_VISION, H_MASK, W_MASK, K_KEYPOINTS
from core.config import load_config, add_config_argument, Config

# =============================================================================
# FEATURE EXTRACTION SETTINGS (NOT IN CONFIG - ARCHITECTURAL)
# =============================================================================

FEATURE_EXTRACTION_SETTINGS = {
    "feature_layer_idx": 9,  # C5 backbone output (SPPF)
    "roi_output_size": 7,
    "sampling_ratio": 2,
    "output_dim": D_VISION,  # 256
}


# =============================================================================
# FEATURE EXTRACTOR (from FeatureExtractor class, adapted for caching)
# =============================================================================

class CachingFeatureExtractor:
    """
    Extracts per-human visual embeddings from YOLO backbone features.
    
    Uses Backbone + ROI Align strategy (locked choice from Phase 2).
    
    DEVICE CONTRACT:
    - All operations occur on self.device
    - Model must already be on self.device before initialization
    - Outputs are returned on self.device (moved to CPU before caching)
    """
    
    def __init__(self, model: YOLO, device: str, feature_layer_idx: int = 9, output_dim: int = 256):
        self.model = model
        self.device = device
        self.feature_layer_idx = feature_layer_idx
        self.output_dim = output_dim
        self.features = None
        self._hook_handle = None
        self._feature_channels = None
        self._projection = None
        
        # Verify model is on expected device
        model_device = next(model.model.parameters()).device
        assert str(model_device).startswith(device.split(':')[0]), \
            f"Model device mismatch: model on {model_device}, expected {device}"
        
        # Register hook on backbone layer
        self._register_hook()
        
        # Initialize projection lazily
        self._ensure_projection_initialized()
    
    def _register_hook(self):
        """Register forward hook on target backbone layer."""
        target_layer = self.model.model.model[self.feature_layer_idx]
        
        def hook_fn(module, input, output):
            self.features = output.detach()
        
        self._hook_handle = target_layer.register_forward_hook(hook_fn)
    
    def _ensure_projection_initialized(self):
        """Initialize projection layer with fixed seed for reproducibility."""
        # Run dummy forward to get feature channel count
        # CRITICAL: dummy must be on same device as model
        dummy_input = torch.randn(1, 3, 640, 640, device=self.device)
        
        # Verify device consistency before dummy forward
        assert dummy_input.device == next(self.model.model.parameters()).device, \
            f"Device mismatch: dummy on {dummy_input.device}, model on {next(self.model.model.parameters()).device}"
        
        with torch.no_grad():
            self.model.model(dummy_input)
        
        if self.features is not None:
            _, C, _, _ = self.features.shape
            self._feature_channels = C
            
            # Fixed seed for deterministic projection
            torch.manual_seed(42)
            self._projection = torch.nn.Linear(C, self.output_dim, bias=False)
            torch.nn.init.orthogonal_(self._projection.weight)
            self._projection.eval()
            self._projection.requires_grad_(False)
            
            # CRITICAL: Move projection to same device as model
            self._projection.to(self.device)
    
    def extract(self, boxes_xyxy: torch.Tensor, orig_shape: tuple) -> torch.Tensor:
        """
        Extract per-human visual embeddings using ROI Align.
        
        Args:
            boxes_xyxy: Detected boxes in [N, 4] xyxy pixel coordinates (will be moved to device)
            orig_shape: Original image shape (H, W)
            
        Returns:
            visual_embeddings: Tensor[N, output_dim] - L2 normalized, on self.device
        """
        N = boxes_xyxy.shape[0]
        
        if N == 0:
            return torch.zeros((0, self.output_dim), dtype=torch.float32, device=self.device)
        
        if self.features is None:
            raise RuntimeError("Features not captured. Run model inference first.")
        
        feat = self.features  # [1, C, H_feat, W_feat]
        _, C, H_feat, W_feat = feat.shape
        orig_H, orig_W = orig_shape
        
        # DEVICE ASSERTION: feature map must be on expected device
        assert feat.device == torch.device(self.device) or str(feat.device).startswith(self.device.split(':')[0]), \
            f"Feature map device mismatch: {feat.device} vs {self.device}"
        
        # Scale boxes to feature map coordinates
        # Move boxes to device first
        boxes_scaled = boxes_xyxy.clone().float().to(self.device)
        scale_x = W_feat / orig_W
        scale_y = H_feat / orig_H
        boxes_scaled[:, [0, 2]] *= scale_x
        boxes_scaled[:, [1, 3]] *= scale_y
        
        # Prepare ROIs: [N, 5] with batch index - MUST be on same device as features
        batch_indices = torch.zeros((N, 1), dtype=torch.float32, device=self.device)
        rois = torch.cat([batch_indices, boxes_scaled], dim=1)
        
        # DEVICE ASSERTION before ROI Align
        assert feat.device == rois.device, \
            f"ROI Align device mismatch: features on {feat.device}, rois on {rois.device}"
        
        # ROI Align
        roi_features = roi_align(
            feat,
            rois,
            output_size=(FEATURE_EXTRACTION_SETTINGS["roi_output_size"], FEATURE_EXTRACTION_SETTINGS["roi_output_size"]),
            spatial_scale=1.0,
            sampling_ratio=FEATURE_EXTRACTION_SETTINGS["sampling_ratio"],
            aligned=True,
        )  # [N, C, 7, 7]
        
        # Global average pooling
        pooled = F.adaptive_avg_pool2d(roi_features, (1, 1))  # [N, C, 1, 1]
        pooled = pooled.view(N, C)  # [N, C]
        
        # DEVICE ASSERTION before projection
        assert pooled.device == next(self._projection.parameters()).device, \
            f"Projection device mismatch: pooled on {pooled.device}, projection on {next(self._projection.parameters()).device}"
        
        # Project to output dimension
        with torch.no_grad():
            embeddings = self._projection(pooled)  # [N, output_dim]
        
        # L2 normalize
        norms = embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        embeddings = embeddings / norms
        
        return embeddings
    
    def cleanup(self):
        """Remove forward hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None


# =============================================================================
# IOU MATCHING (Pose to Seg)
# =============================================================================

def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: [N, 4] xyxy format
        boxes2: [M, 4] xyxy format
        
    Returns:
        iou_matrix: [N, M]
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    
    if N == 0 or M == 0:
        return torch.zeros((N, M), dtype=torch.float32)
    
    # Expand for broadcasting
    boxes1_exp = boxes1.unsqueeze(1)  # [N, 1, 4]
    boxes2_exp = boxes2.unsqueeze(0)  # [1, M, 4]
    
    # Intersection
    x1_inter = torch.max(boxes1_exp[..., 0], boxes2_exp[..., 0])
    y1_inter = torch.max(boxes1_exp[..., 1], boxes2_exp[..., 1])
    x2_inter = torch.min(boxes1_exp[..., 2], boxes2_exp[..., 2])
    y2_inter = torch.min(boxes1_exp[..., 3], boxes2_exp[..., 3])
    
    w_inter = (x2_inter - x1_inter).clamp(min=0)
    h_inter = (y2_inter - y1_inter).clamp(min=0)
    area_inter = w_inter * h_inter
    
    # Union
    area1 = (boxes1_exp[..., 2] - boxes1_exp[..., 0]) * (boxes1_exp[..., 3] - boxes1_exp[..., 1])
    area2 = (boxes2_exp[..., 2] - boxes2_exp[..., 0]) * (boxes2_exp[..., 3] - boxes2_exp[..., 1])
    area_union = area1 + area2 - area_inter
    
    iou = area_inter / area_union.clamp(min=1e-8)
    return iou


def match_masks_to_poses(
    boxes_pose: torch.Tensor,
    boxes_seg: torch.Tensor,
    masks_seg: torch.Tensor,
    iou_threshold: float = 0.3,
) -> torch.Tensor:
    """
    Match segmentation masks to pose detections by IoU.
    
    Args:
        boxes_pose: [N_pose, 4] pose detection boxes
        boxes_seg: [N_seg, 4] segmentation detection boxes
        masks_seg: [N_seg, H, W] segmentation masks
        iou_threshold: Minimum IoU for match
        
    Returns:
        masks_matched: [N_pose, H, W] matched masks (zeros if no match)
    """
    N_pose = boxes_pose.shape[0]
    N_seg = boxes_seg.shape[0]
    
    if N_pose == 0:
        return torch.zeros((0, masks_seg.shape[1], masks_seg.shape[2]), dtype=torch.float32)
    
    if N_seg == 0:
        H, W = masks_seg.shape[1:] if masks_seg.dim() == 3 else (160, 160)
        return torch.zeros((N_pose, H, W), dtype=torch.float32)
    
    # Compute IoU
    iou_matrix = compute_iou(boxes_pose, boxes_seg)  # [N_pose, N_seg]
    
    # For each pose detection, find best matching seg mask
    matched_masks = []
    for i in range(N_pose):
        ious = iou_matrix[i]  # [N_seg]
        best_idx = ious.argmax().item()
        best_iou = ious[best_idx].item()
        
        if best_iou >= iou_threshold:
            matched_masks.append(masks_seg[best_idx])
        else:
            # No match - use zero mask
            matched_masks.append(torch.zeros_like(masks_seg[0]))
    
    return torch.stack(matched_masks, dim=0) if matched_masks else torch.zeros((0, masks_seg.shape[1], masks_seg.shape[2]))


# =============================================================================
# MAIN CACHING FUNCTION
# =============================================================================

def cache_yolo_features(config: Config):
    """
    Main function to cache YOLO features for all images.
    
    Args:
        config: Configuration object with paths and settings.
    """
    print("\n" + "=" * 70)
    print("YOLO FEATURE CACHING FOR RefYOLO-Human TRAINING")
    print("=" * 70)
    
    # Get paths from config
    images_dir = config.images_dir
    cache_dir = config.features_dir
    device = config.training.device
    
    # ==========================================================================
    # DEVICE VALIDATION (Global Contract - STEP 1 of audit)
    # ==========================================================================
    print(f"\nDevice configuration: {device}")
    
    # Assert CUDA availability if configured
    if device.startswith("cuda"):
        assert torch.cuda.is_available(), \
            f"Config specifies device='{device}' but CUDA is not available!"
        print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  Running on CPU")
    
    # Create cache directory
    cache_dir.mkdir(exist_ok=True)
    print(f"\nCache directory: {cache_dir}")
    
    # Get all image files
    image_files = sorted(images_dir.glob("*.jpg"))
    print(f"Found {len(image_files)} images in {images_dir}")
    
    if len(image_files) == 0:
        print("ERROR: No images found!")
        return
    
    # ==========================================================================
    # STEP 1: Initialize frozen YOLO models
    # ==========================================================================
    print("\n" + "-" * 50)
    print("STEP 1: Initializing frozen YOLO models")
    print("-" * 50)
    
    model_pose_path = config.pose_model_path
    model_seg_path = config.seg_model_path
    
    print(f"  Loading pose model: {model_pose_path}")
    model_pose = YOLO(str(model_pose_path))
    model_pose.to(device)
    model_pose.model.eval()
    
    # Freeze all parameters
    for param in model_pose.model.parameters():
        param.requires_grad = False
    
    print(f"  Loading seg model: {model_seg_path}")
    model_seg = YOLO(str(model_seg_path))
    model_seg.to(device)
    model_seg.model.eval()
    
    for param in model_seg.model.parameters():
        param.requires_grad = False
    
    print("  ✓ Both models loaded and frozen")
    
    # Verify models are on expected device
    pose_model_device = next(model_pose.model.parameters()).device
    seg_model_device = next(model_seg.model.parameters()).device
    print(f"  Pose model device: {pose_model_device}")
    print(f"  Seg model device: {seg_model_device}")
    
    # DEVICE ASSERTION: Ensure models are on configured device
    assert str(pose_model_device).startswith(device.split(':')[0]), \
        f"Pose model on wrong device: {pose_model_device}, expected {device}"
    assert str(seg_model_device).startswith(device.split(':')[0]), \
        f"Seg model on wrong device: {seg_model_device}, expected {device}"
    
    # Initialize feature extractor (using pose model backbone)
    print("  Initializing feature extractor (Backbone + ROI Align)")
    extractor = CachingFeatureExtractor(
        model_pose,
        device=device,  # CRITICAL: Pass device for device-consistent operations
        feature_layer_idx=FEATURE_EXTRACTION_SETTINGS["feature_layer_idx"],
        output_dim=FEATURE_EXTRACTION_SETTINGS["output_dim"],
    )
    print(f"  ✓ Feature extractor ready (layer {FEATURE_EXTRACTION_SETTINGS['feature_layer_idx']}, dim {FEATURE_EXTRACTION_SETTINGS['output_dim']})")
    
    # ==========================================================================
    # STEP 2-5: Process images and cache features
    # ==========================================================================
    print("\n" + "-" * 50)
    print("STEP 2-5: Processing images and caching features")
    print("-" * 50)
    
    stats = {
        "total_images": len(image_files),
        "processed": 0,
        "total_humans": 0,
        "images_with_humans": 0,
        "failed": 0,
        "nan_inf_count": 0,
    }
    
    start_time = time.time()
    
    for img_path in tqdm(image_files, desc="Caching features"):
        try:
            # Extract image_id from filename (e.g., "100000.jpg" -> 100000)
            image_id = int(img_path.stem)
            file_name = img_path.name
            
            # Run YOLO inference (pose)
            with torch.no_grad():
                results_pose = model_pose(str(img_path), device=device, verbose=False)
            result_pose = results_pose[0]
            orig_shape = result_pose.orig_shape  # (H, W)
            H_orig, W_orig = orig_shape
            
            # Run YOLO inference (seg)
            with torch.no_grad():
                results_seg = model_seg(str(img_path), device=device, verbose=False, classes=[0])
            result_seg = results_seg[0]
            
            # Get pose detections
            if result_pose.boxes is not None and len(result_pose.boxes) > 0:
                boxes_pose = result_pose.boxes.xyxy.cpu()  # [N, 4] pixel coords
                keypoints_raw = result_pose.keypoints.data.cpu()  # [N, 17, 3]
                N = boxes_pose.shape[0]
            else:
                boxes_pose = torch.zeros((0, 4), dtype=torch.float32)
                keypoints_raw = torch.zeros((0, K_KEYPOINTS, 3), dtype=torch.float32)
                N = 0
            
            # Get seg detections
            if result_seg.boxes is not None and result_seg.masks is not None and len(result_seg.boxes) > 0:
                boxes_seg = result_seg.boxes.xyxy.cpu()  # [N_seg, 4]
                masks_raw = result_seg.masks.data.cpu()  # [N_seg, H, W]
            else:
                boxes_seg = torch.zeros((0, 4), dtype=torch.float32)
                masks_raw = torch.zeros((0, H_MASK, W_MASK), dtype=torch.float32)
            
            # Handle zero humans case
            if N == 0:
                cache_data = {
                    "image_id": image_id,
                    "file_name": file_name,
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "masks": torch.zeros((0, H_MASK, W_MASK), dtype=torch.float32),
                    "keypoints": torch.zeros((0, K_KEYPOINTS, 3), dtype=torch.float32),
                    "visual_embeddings": torch.zeros((0, D_VISION), dtype=torch.float32),
                    "valid": torch.zeros((0,), dtype=torch.bool),
                }
            else:
                # Normalize boxes to [0, 1]
                boxes_normalized = boxes_pose.clone()
                boxes_normalized[:, [0, 2]] /= W_orig
                boxes_normalized[:, [1, 3]] /= H_orig
                
                # Normalize keypoints to [0, 1]
                keypoints_normalized = keypoints_raw.clone()
                keypoints_normalized[:, :, 0] /= W_orig  # x
                keypoints_normalized[:, :, 1] /= H_orig  # y
                # Keep confidence (index 2) unchanged
                
                # Match masks to pose detections
                masks_matched = match_masks_to_poses(boxes_pose, boxes_seg, masks_raw)
                
                # Resize masks to fixed size
                if masks_matched.shape[0] > 0:
                    masks_resized = F.interpolate(
                        masks_matched.unsqueeze(1),  # [N, 1, H, W]
                        size=(H_MASK, W_MASK),
                        mode='bilinear',
                        align_corners=False,
                    ).squeeze(1)  # [N, H_MASK, W_MASK]
                else:
                    masks_resized = torch.zeros((0, H_MASK, W_MASK), dtype=torch.float32)
                
                # Extract visual embeddings using ROI Align
                visual_embeddings = extractor.extract(boxes_pose, orig_shape)
                
                # Create validity mask
                valid = torch.ones(N, dtype=torch.bool)
                
                cache_data = {
                    "image_id": image_id,
                    "file_name": file_name,
                    "boxes": boxes_normalized,
                    "masks": masks_resized,
                    "keypoints": keypoints_normalized,
                    "visual_embeddings": visual_embeddings,
                    "valid": valid,
                }
                
                stats["total_humans"] += N
                stats["images_with_humans"] += 1
            
            # Validate: check for NaN/Inf
            has_nan_inf = False
            for key, tensor in cache_data.items():
                if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
                    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                        has_nan_inf = True
                        print(f"\n  WARNING: NaN/Inf in {key} for image {image_id}")
                        break
            
            if has_nan_inf:
                stats["nan_inf_count"] += 1
            
            # CRITICAL: Move all tensors to CPU before saving to disk
            # This ensures cache files are device-agnostic and can be loaded anywhere
            for key, value in cache_data.items():
                if isinstance(value, torch.Tensor):
                    cache_data[key] = value.cpu()
            
            # ASSERTION: Verify no CUDA tensors are being saved
            for key, value in cache_data.items():
                if isinstance(value, torch.Tensor):
                    assert value.device == torch.device('cpu'), \
                        f"CACHE ERROR: Tensor '{key}' is on {value.device}, must be on CPU for disk storage"
            
            # Save cache
            cache_path = cache_dir / f"{image_id}.pt"
            torch.save(cache_data, cache_path)
            
            stats["processed"] += 1
            
        except Exception as e:
            print(f"\n  ERROR processing {img_path}: {e}")
            stats["failed"] += 1
    
    elapsed = time.time() - start_time
    
    # Cleanup
    extractor.cleanup()
    
    # ==========================================================================
    # STEP 6: Validate cache integrity
    # ==========================================================================
    print("\n" + "-" * 50)
    print("STEP 6: Validating cache integrity")
    print("-" * 50)
    
    cache_files = list(cache_dir.glob("*.pt"))
    print(f"  Cache files created: {len(cache_files)}")
    
    validation_errors = []
    sample_shapes = {}
    
    for cf in cache_files[:10]:  # Validate first 10
        data = torch.load(cf, weights_only=True)
        
        # Check required keys
        required_keys = ["image_id", "file_name", "boxes", "masks", "keypoints", "visual_embeddings", "valid"]
        for key in required_keys:
            if key not in data:
                validation_errors.append(f"{cf.name}: Missing key '{key}'")
        
        # Check shapes
        N = data["boxes"].shape[0]
        if data["masks"].shape[0] != N:
            validation_errors.append(f"{cf.name}: masks shape mismatch")
        if data["keypoints"].shape[0] != N:
            validation_errors.append(f"{cf.name}: keypoints shape mismatch")
        if data["visual_embeddings"].shape[0] != N:
            validation_errors.append(f"{cf.name}: visual_embeddings shape mismatch")
        if data["valid"].shape[0] != N:
            validation_errors.append(f"{cf.name}: valid shape mismatch")
        
        # Check dimensions
        if N > 0:
            if data["boxes"].shape != (N, 4):
                validation_errors.append(f"{cf.name}: boxes dim error")
            if data["masks"].shape[1:] != (H_MASK, W_MASK):
                validation_errors.append(f"{cf.name}: masks dim error {data['masks'].shape}")
            if data["keypoints"].shape[1:] != (K_KEYPOINTS, 3):
                validation_errors.append(f"{cf.name}: keypoints dim error")
            if data["visual_embeddings"].shape[1] != D_VISION:
                validation_errors.append(f"{cf.name}: visual_embeddings dim error")
        
        sample_shapes[cf.name] = {
            "N": N,
            "boxes": data["boxes"].shape,
            "masks": data["masks"].shape,
            "keypoints": data["keypoints"].shape,
            "visual_embeddings": data["visual_embeddings"].shape,
            "valid": data["valid"].shape,
        }
    
    if validation_errors:
        print("  VALIDATION ERRORS:")
        for err in validation_errors:
            print(f"    - {err}")
    else:
        print("  ✓ All validated samples passed integrity check")
    
    print("\n  Sample shapes (first 3 with humans):")
    count = 0
    for name, shapes in sample_shapes.items():
        if shapes["N"] > 0:
            print(f"    {name}: N={shapes['N']}")
            print(f"      boxes: {shapes['boxes']}")
            print(f"      masks: {shapes['masks']}")
            print(f"      keypoints: {shapes['keypoints']}")
            print(f"      visual_embeddings: {shapes['visual_embeddings']}")
            count += 1
            if count >= 3:
                break
    
    # ==========================================================================
    # STEP 7: Print cache summary
    # ==========================================================================
    print("\n" + "-" * 50)
    print("STEP 7: Cache Summary")
    print("-" * 50)
    
    # Calculate cache size
    total_size = sum(f.stat().st_size for f in cache_files) / (1024 * 1024)
    
    print(f"""
    ┌─────────────────────────────────────────────────────────────────┐
    │                    FEATURE CACHE SUMMARY                        │
    ├─────────────────────────────────────────────────────────────────┤
    │ Images processed:        {stats['processed']:>6} / {stats['total_images']:<6}                    │
    │ Images with humans:      {stats['images_with_humans']:>6}                                  │
    │ Total humans detected:   {stats['total_humans']:>6}                                  │
    │ Average humans/image:    {stats['total_humans']/max(stats['images_with_humans'],1):>6.2f}                                  │
    │ Failed images:           {stats['failed']:>6}                                  │
    │ NaN/Inf warnings:        {stats['nan_inf_count']:>6}                                  │
    ├─────────────────────────────────────────────────────────────────┤
    │ Processing time:         {elapsed:>6.1f} seconds                           │
    │ Time per image:          {elapsed/max(stats['processed'],1):>6.2f} seconds                           │
    │ Cache size:              {total_size:>6.1f} MB                                 │
    │ Cache directory:         {str(cache_dir):<40}│
    ├─────────────────────────────────────────────────────────────────┤
    │ Cache format per image:                                         │
    │   - image_id: int                                               │
    │   - file_name: str                                              │
    │   - boxes: Tensor[N, 4]        (normalized xyxy)                │
    │   - masks: Tensor[N, 160, 160] (resized)                        │
    │   - keypoints: Tensor[N, 17, 3] (normalized)                    │
    │   - visual_embeddings: Tensor[N, 256] (L2 normalized)           │
    │   - valid: Tensor[N]                                            │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    if stats['failed'] == 0 and stats['nan_inf_count'] == 0:
        print("    ✓ CACHING COMPLETE - READY FOR TRAINING")
    else:
        print("    ⚠ CACHING COMPLETE WITH WARNINGS - REVIEW ERRORS ABOVE")


if __name__ == "__main__":
    import argparse
    from core.config import load_config, add_config_argument
    
    parser = argparse.ArgumentParser(description="Cache YOLO features for grounding training")
    add_config_argument(parser)
    args = parser.parse_args()
    
    config = load_config(args.config)
    cache_yolo_features(config)
