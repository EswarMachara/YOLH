"""
YOLO Model Output Explorer - Phase 1

This script validates YOLO outputs and maps them into VisionOutput contract.
Ultralytics does NOT have a single model for seg+pose+boxes simultaneously.
We use yolo11n-pose for keypoints+boxes (person class only).
Segmentation masks will be handled separately or as placeholders.

⚠️ EXPLORATION SCRIPT - NOT FOR PRODUCTION USE
   - Hardcoded CPU device for local development
   - Use cache_yolo_features.py for device-aware production caching
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import urllib.request
from ultralytics import YOLO

from core.datatypes import (
    VisionOutput,
    D_VISION,
    H_MASK,
    W_MASK,
    K_KEYPOINTS,
)
from core.assertions import assert_vision_output


def download_test_image(url: str, save_path: str) -> str:
    """Download a test image if not already present."""
    if not Path(save_path).exists():
        print(f"Downloading test image to {save_path}...")
        urllib.request.urlretrieve(url, save_path)
    return save_path


def main():
    print("=" * 70)
    print("PHASE 1: YOLO Output Validation & VisionOutput Mapping")
    print("=" * 70)
    
    # =========================================================================
    # TASK 1: Environment Verification
    # =========================================================================
    print("\n[TASK 1] Environment Verification")
    print("-" * 50)
    
    import ultralytics
    print(f"Ultralytics version: {ultralytics.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: CPU (forced)")
    
    # =========================================================================
    # TASK 2: Model Loading (Read-Only)
    # =========================================================================
    print("\n[TASK 2] Model Loading")
    print("-" * 50)
    
    # Load pose model (includes boxes + keypoints, person class)
    model_pose = YOLO("yolo11n-pose.pt")
    model_pose.to("cpu")
    print(f"Loaded model: yolo11n-pose.pt")
    print(f"Model task: {model_pose.task}")
    print(f"Model device: {next(model_pose.model.parameters()).device}")
    
    # Load segmentation model (includes boxes + masks)
    model_seg = YOLO("yolo11n-seg.pt")
    model_seg.to("cpu")
    print(f"Loaded model: yolo11n-seg.pt")
    print(f"Model task: {model_seg.task}")
    
    # =========================================================================
    # TASK 3: Single Image Inference
    # =========================================================================
    print("\n[TASK 3] Single Image Inference")
    print("-" * 50)
    
    # Download a test image with people
    test_image_url = "https://ultralytics.com/images/bus.jpg"
    test_image_path = download_test_image(test_image_url, "test_bus.jpg")
    print(f"Test image: {test_image_path}")
    
    # Run inference - POSE model
    print("\nRunning POSE model inference...")
    results_pose = model_pose(test_image_path, device="cpu", verbose=False)
    print(f"Pose results type: {type(results_pose)}")
    print(f"Pose results length: {len(results_pose)}")
    
    # Run inference - SEG model (filter to person class=0)
    print("\nRunning SEG model inference...")
    results_seg = model_seg(test_image_path, device="cpu", verbose=False, classes=[0])
    print(f"Seg results type: {type(results_seg)}")
    print(f"Seg results length: {len(results_seg)}")
    
    # =========================================================================
    # TASK 4: Results Structure Inspection
    # =========================================================================
    print("\n[TASK 4] Results Structure Inspection")
    print("-" * 50)
    
    result_pose = results_pose[0]
    result_seg = results_seg[0]
    
    print("\n--- POSE RESULT ATTRIBUTES ---")
    pose_attrs = [attr for attr in dir(result_pose) if not attr.startswith('_')]
    print(f"Available attributes: {pose_attrs}")
    
    print("\n--- POSE BOXES ---")
    if result_pose.boxes is not None:
        print(f"  boxes.xyxy shape: {result_pose.boxes.xyxy.shape}")
        print(f"  boxes.conf shape: {result_pose.boxes.conf.shape}")
        print(f"  boxes.cls shape: {result_pose.boxes.cls.shape}")
        print(f"  boxes.xyxy dtype: {result_pose.boxes.xyxy.dtype}")
    else:
        print("  boxes: None")
    
    print("\n--- POSE KEYPOINTS ---")
    if result_pose.keypoints is not None:
        print(f"  keypoints.data shape: {result_pose.keypoints.data.shape}")
        print(f"  keypoints.xy shape: {result_pose.keypoints.xy.shape}")
        print(f"  keypoints.conf shape: {result_pose.keypoints.conf.shape if result_pose.keypoints.conf is not None else 'None'}")
        print(f"  keypoints.data dtype: {result_pose.keypoints.data.dtype}")
    else:
        print("  keypoints: None")
    
    print("\n--- SEG RESULT ATTRIBUTES ---")
    seg_attrs = [attr for attr in dir(result_seg) if not attr.startswith('_')]
    print(f"Available attributes: {seg_attrs}")
    
    print("\n--- SEG MASKS ---")
    if result_seg.masks is not None:
        print(f"  masks.data shape: {result_seg.masks.data.shape}")
        print(f"  masks.xy: {type(result_seg.masks.xy)}")
        print(f"  masks.data dtype: {result_seg.masks.data.dtype}")
    else:
        print("  masks: None")
    
    print("\n--- SEG BOXES ---")
    if result_seg.boxes is not None:
        print(f"  boxes.xyxy shape: {result_seg.boxes.xyxy.shape}")
        print(f"  boxes.conf shape: {result_seg.boxes.conf.shape}")
    else:
        print("  boxes: None")
    
    # =========================================================================
    # TASK 5: Extraction Into Tensors
    # =========================================================================
    print("\n[TASK 5] Extraction Into Tensors")
    print("-" * 50)
    
    # Get image dimensions for normalization
    img_height, img_width = result_pose.orig_shape
    print(f"Original image shape: {img_height} x {img_width}")
    
    # --- Extract from POSE model (boxes + keypoints) ---
    # Filter to person class only (cls == 0)
    pose_boxes = result_pose.boxes
    if pose_boxes is not None and len(pose_boxes) > 0:
        # Pose model is person-only, no filtering needed
        N_pose = len(pose_boxes)
        
        # Boxes: [N, 4] in xyxy format, normalize to [0,1]
        boxes_xyxy = pose_boxes.xyxy.float().cpu()  # [N, 4]
        boxes_normalized = boxes_xyxy.clone()
        boxes_normalized[:, [0, 2]] /= img_width
        boxes_normalized[:, [1, 3]] /= img_height
        
        # Confidence scores
        scores = pose_boxes.conf.float().cpu()  # [N]
        
        print(f"Extracted boxes shape: {boxes_normalized.shape}")
        print(f"Extracted scores shape: {scores.shape}")
    else:
        N_pose = 0
        boxes_normalized = torch.zeros((0, 4), dtype=torch.float32)
        scores = torch.zeros((0,), dtype=torch.float32)
        print("No pose detections")
    
    # --- Extract keypoints ---
    if result_pose.keypoints is not None and N_pose > 0:
        kpts_data = result_pose.keypoints.data.float().cpu()  # [N, 17, 3] (x, y, conf)
        
        # Normalize x, y coordinates relative to bounding box
        # For now, normalize relative to image (we'll adjust later)
        kpts_normalized = kpts_data.clone()
        kpts_normalized[:, :, 0] /= img_width   # x
        kpts_normalized[:, :, 1] /= img_height  # y
        # conf stays as is
        
        print(f"Extracted keypoints shape: {kpts_normalized.shape}")
    else:
        kpts_normalized = torch.zeros((N_pose, K_KEYPOINTS, 3), dtype=torch.float32)
        print(f"Keypoints placeholder shape: {kpts_normalized.shape}")
    
    # --- Extract from SEG model (masks) ---
    # Note: Seg model may detect different number of persons
    if result_seg.masks is not None and len(result_seg.masks.data) > 0:
        masks_raw = result_seg.masks.data.float().cpu()  # [N_seg, H, W]
        N_seg = masks_raw.shape[0]
        
        # Resize masks to fixed size H_MASK x W_MASK
        masks_resized = torch.nn.functional.interpolate(
            masks_raw.unsqueeze(1),  # [N, 1, H, W]
            size=(H_MASK, W_MASK),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # [N, H_MASK, W_MASK]
        
        print(f"Extracted masks shape (from seg): {masks_resized.shape}")
    else:
        N_seg = 0
        masks_resized = None
        print("No segmentation masks detected")
    
    # =========================================================================
    # TASK 6: Map Into VisionOutput
    # =========================================================================
    print("\n[TASK 6] Map Into VisionOutput")
    print("-" * 50)
    
    # We use pose model detections as primary (N = N_pose)
    # Masks: If seg model detected different count, we use placeholders
    N = N_pose
    B = 1  # Single image batch
    
    print(f"Number of detections (N): {N}")
    print(f"Batch size (B): {B}")
    
    # Boxes: [B, N, 4]
    final_boxes = boxes_normalized.unsqueeze(0)  # [1, N, 4]
    
    # boxes_valid: [B, N]
    final_boxes_valid = torch.ones((B, N), dtype=torch.bool)
    
    # Masks: [B, N, H_MASK, W_MASK]
    # Since seg and pose may have different N, use placeholder zeros
    # TEMPORARY: In Phase 2, we'll match detections between models
    final_masks = torch.zeros((B, N, H_MASK, W_MASK), dtype=torch.float32)
    if masks_resized is not None and N_seg == N:
        final_masks[0] = masks_resized
        print(f"Using actual masks from segmentation model")
    else:
        print(f"Using placeholder masks (seg N={N_seg} != pose N={N})")
    
    # Keypoints: [B, N, K, 3]
    final_keypoints = kpts_normalized.unsqueeze(0)  # [1, N, 17, 3]
    
    # Visual embeddings: [B, N, D_VISION]
    # TEMPORARY PLACEHOLDER: Zero tensor until feature extraction is implemented
    # This is explicitly marked as temporary and will be replaced in Phase 2
    final_visual_embeddings = torch.zeros((B, N, D_VISION), dtype=torch.float32)
    print(f"Visual embeddings: PLACEHOLDER zeros [B={B}, N={N}, D={D_VISION}]")
    
    print(f"\nFinal tensor shapes:")
    print(f"  boxes: {final_boxes.shape}")
    print(f"  boxes_valid: {final_boxes_valid.shape}")
    print(f"  masks: {final_masks.shape}")
    print(f"  keypoints: {final_keypoints.shape}")
    print(f"  visual_embeddings: {final_visual_embeddings.shape}")
    
    # Create VisionOutput instance
    vision_output = VisionOutput(
        boxes=final_boxes,
        boxes_valid=final_boxes_valid,
        masks=final_masks,
        keypoints=final_keypoints,
        visual_embeddings=final_visual_embeddings,
    )
    
    print(f"\nVisionOutput instance created:")
    print(f"  {vision_output}")
    
    # =========================================================================
    # TASK 7: Assertion Checks
    # =========================================================================
    print("\n[TASK 7] Assertion Checks")
    print("-" * 50)
    
    try:
        assert_vision_output(vision_output)
        print("[PASS] assert_vision_output passed for human detection case")
    except (TypeError, ValueError) as e:
        print(f"[FAIL] assert_vision_output failed: {e}")
        return
    
    # =========================================================================
    # TASK 8: Zero-Detection Sanity Test
    # =========================================================================
    print("\n[TASK 8] Zero-Detection Sanity Test")
    print("-" * 50)
    
    # Create zero-detection VisionOutput
    N_zero = 0
    zero_vision_output = VisionOutput(
        boxes=torch.zeros((B, N_zero, 4), dtype=torch.float32),
        boxes_valid=torch.zeros((B, N_zero), dtype=torch.bool),
        masks=torch.zeros((B, N_zero, H_MASK, W_MASK), dtype=torch.float32),
        keypoints=torch.zeros((B, N_zero, K_KEYPOINTS, 3), dtype=torch.float32),
        visual_embeddings=torch.zeros((B, N_zero, D_VISION), dtype=torch.float32),
    )
    
    print(f"Zero-detection tensor shapes:")
    print(f"  boxes: {zero_vision_output.boxes.shape}")
    print(f"  boxes_valid: {zero_vision_output.boxes_valid.shape}")
    print(f"  masks: {zero_vision_output.masks.shape}")
    print(f"  keypoints: {zero_vision_output.keypoints.shape}")
    print(f"  visual_embeddings: {zero_vision_output.visual_embeddings.shape}")
    
    try:
        assert_vision_output(zero_vision_output)
        print("[PASS] assert_vision_output passed for zero-detection case")
    except (TypeError, ValueError) as e:
        print(f"[FAIL] assert_vision_output failed: {e}")
        return
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"Ultralytics version: {ultralytics.__version__}")
    print(f"Models loaded: yolo11n-pose.pt, yolo11n-seg.pt")
    print(f"Device: CPU")
    print(f"Detections found: {N}")
    print(f"Assertion checks: ALL PASSED")
    print(f"Zero-detection handling: VERIFIED")
    print("\nNotes:")
    print("  - visual_embeddings are PLACEHOLDER zeros (Phase 2 will add extraction)")
    print("  - Masks may be placeholders if seg/pose detection counts differ")
    print("  - Keypoints normalized to image coordinates (contract allows this)")
    print("=" * 70)


if __name__ == "__main__":
    main()
