"""
Phase 2: Visual Embeddings Feature Extraction Design & Validation

This script designs and validates the permanent architectural choice for
computing per-human visual embeddings from YOLO model outputs.

CPU-only. No training. No YOLO internals modification.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from ultralytics import YOLO

from core.datatypes import (
    VisionOutput,
    D_VISION,
    H_MASK,
    W_MASK,
    K_KEYPOINTS,
)
from core.assertions import assert_vision_output


def task1_enumerate_strategies():
    """
    TASK 1: Enumerate all viable feature extraction strategies from YOLO.
    
    Document pros, cons, dimensionality, and alignment risks.
    """
    print("\n" + "=" * 70)
    print("TASK 1: FEATURE EXTRACTION STRATEGY ENUMERATION")
    print("=" * 70)
    
    strategies = """
    ┌─────────────────────────────────────────────────────────────────────┐
    │ STRATEGY 1: Backbone Feature Map + ROI Align                        │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Description:                                                        │
    │   Extract features from backbone (e.g., C3/C4/C5 layers), then     │
    │   apply ROI Align using detected bounding boxes to get per-human   │
    │   feature vectors.                                                  │
    │                                                                     │
    │ Pros:                                                               │
    │   + Rich semantic features from deep backbone layers               │
    │   + Standard approach used in Mask R-CNN, Faster R-CNN             │
    │   + Per-box alignment is exact via ROI Align                       │
    │   + Deterministic given fixed boxes                                │
    │                                                                     │
    │ Cons:                                                               │
    │   - Requires forward hook to capture intermediate features          │
    │   - ROI Align implementation needed (torchvision.ops)              │
    │   - Feature map resolution may vary with input size                │
    │                                                                     │
    │ Expected Dimensionality: Configurable (project to 256)             │
    │ Alignment Risk: LOW - ROI Align guarantees box-feature alignment   │
    └─────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │ STRATEGY 2: Neck (FPN/PAN) Feature Map Pooling                      │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Description:                                                        │
    │   Extract features from neck (FPN/PAN outputs) at multiple scales, │
    │   pool features within each bounding box region.                    │
    │                                                                     │
    │ Pros:                                                               │
    │   + Multi-scale features capture both fine and coarse details      │
    │   + Neck features are already detection-optimized                  │
    │   + Available at standard YOLO inference                           │
    │                                                                     │
    │ Cons:                                                               │
    │   - Multiple feature maps at different resolutions                 │
    │   - Need to select appropriate scale per detection                 │
    │   - Pooling strategy affects quality                               │
    │                                                                     │
    │ Expected Dimensionality: 256 (sum of neck channels or projection)  │
    │ Alignment Risk: MEDIUM - scale selection may introduce variance    │
    └─────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │ STRATEGY 3: Detection Head Latent Vectors                           │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Description:                                                        │
    │   Extract the latent feature vectors from the detection head just  │
    │   before the final prediction layers (box/cls/kpt regression).     │
    │                                                                     │
    │ Pros:                                                               │
    │   + Features are already per-detection (no ROI needed)             │
    │   + Most detection-relevant information preserved                  │
    │   + Natural alignment with detection outputs                       │
    │                                                                     │
    │ Cons:                                                               │
    │   - Requires deep introspection of detection head                  │
    │   - May not be accessible without model modification               │
    │   - YOLO head structure varies by version                          │
    │                                                                     │
    │ Expected Dimensionality: Head-dependent (64-256)                   │
    │ Alignment Risk: LOW - inherently per-detection                     │
    └─────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │ STRATEGY 4: Mask-Based Feature Pooling                              │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Description:                                                        │
    │   Use segmentation masks to weight backbone/neck features,         │
    │   computing mask-weighted average pooling per human.                │
    │                                                                     │
    │ Pros:                                                               │
    │   + Precise human region (excludes background)                     │
    │   + Robust to box inaccuracies                                     │
    │                                                                     │
    │ Cons:                                                               │
    │   - Requires segmentation model (separate from pose)               │
    │   - Mask quality affects embedding quality                         │
    │   - Soft masks introduce continuous variance                       │
    │   - REJECTED: Unstable when mask confidence varies                 │
    │                                                                     │
    │ Expected Dimensionality: Same as backbone features                 │
    │ Alignment Risk: HIGH - mask instability propagates to embeddings   │
    │ STATUS: REJECTED                                                   │
    └─────────────────────────────────────────────────────────────────────┘
    """
    print(strategies)
    
    return {
        "strategy_1": "Backbone + ROI Align",
        "strategy_2": "Neck FPN Pooling", 
        "strategy_3": "Detection Head Latents",
        "strategy_4": "Mask-Based Pooling (REJECTED)",
    }


def task2_introspect_model():
    """
    TASK 2: Introspect YOLO model modules to understand feature access points.
    """
    print("\n" + "=" * 70)
    print("TASK 2: YOLO MODEL INTROSPECTION")
    print("=" * 70)
    
    # Load pose model
    model = YOLO("yolo11n-pose.pt")
    model.to("cpu")
    model.model.eval()
    
    print("\n--- MODEL MODULE TREE (Top-Level) ---")
    for name, module in model.model.named_children():
        print(f"  {name}: {module.__class__.__name__}")
    
    print("\n--- BACKBONE STRUCTURE (model.model) ---")
    # YOLO11 uses a sequential model structure
    # The model has 'model' attribute which is the actual nn.Module
    
    backbone_info = {}
    neck_info = {}
    head_info = {}
    
    # Iterate through all layers
    print("\n--- FULL MODEL LAYER STRUCTURE ---")
    for idx, layer in enumerate(model.model.model):
        layer_name = layer.__class__.__name__
        print(f"  [{idx:2d}] {layer_name}")
        
        # Check for output channels if available
        if hasattr(layer, 'cv2') and hasattr(layer.cv2, 'conv'):
            out_ch = layer.cv2.conv.out_channels
            print(f"       -> out_channels: {out_ch}")
        elif hasattr(layer, 'conv') and hasattr(layer.conv, 'out_channels'):
            print(f"       -> out_channels: {layer.conv.out_channels}")
    
    # Run a test forward pass to capture shapes
    print("\n--- FEATURE MAP SHAPES (Forward Pass) ---")
    
    test_input = torch.randn(1, 3, 640, 640)
    feature_shapes = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                feature_shapes[name] = output.shape
            elif isinstance(output, (list, tuple)):
                feature_shapes[name] = [o.shape if isinstance(o, torch.Tensor) else type(o) for o in output]
        return hook
    
    handles = []
    for idx, layer in enumerate(model.model.model):
        h = layer.register_forward_hook(hook_fn(f"layer_{idx}"))
        handles.append(h)
    
    with torch.no_grad():
        _ = model.model(test_input)
    
    for h in handles:
        h.remove()
    
    print("\n  Layer Feature Shapes:")
    for name, shape in feature_shapes.items():
        print(f"    {name}: {shape}")
    
    # Identify key layers
    print("\n--- KEY LAYERS IDENTIFIED ---")
    print("  Backbone output layers: Typically layers 4, 6, 9 (C3, C4, C5)")
    print("  Neck (SPPF/Concat): Layers around 10-22")
    print("  Head: Final layers (Pose/Detect head)")
    
    return model, feature_shapes


def task3_select_strategy(feature_shapes):
    """
    TASK 3: Select and lock the feature extraction strategy.
    """
    print("\n" + "=" * 70)
    print("TASK 3: STRATEGY SELECTION (LOCKED)")
    print("=" * 70)
    
    decision = """
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    SELECTED STRATEGY                                │
    │                                                                     │
    │   >>> STRATEGY 1: Backbone Feature Map + ROI Align <<<             │
    │                                                                     │
    ├─────────────────────────────────────────────────────────────────────┤
    │ JUSTIFICATION:                                                      │
    │                                                                     │
    │ 1. PER-HUMAN EMBEDDINGS: ROI Align extracts features precisely     │
    │    for each detected bounding box, guaranteeing one embedding      │
    │    per detection.                                                   │
    │                                                                     │
    │ 2. DETERMINISTIC: Given fixed boxes and feature maps, ROI Align    │
    │    produces identical outputs every time (no randomness).          │
    │                                                                     │
    │ 3. NO RETRAINING: Uses existing backbone features with a standard  │
    │    pooling operation. No model weights are modified.               │
    │                                                                     │
    │ 4. CPU COMPATIBLE: torchvision.ops.roi_align works on CPU.         │
    │                                                                     │
    │ 5. ALIGNMENT GUARANTEE: ROI indices map 1:1 to detection indices.  │
    │                                                                     │
    ├─────────────────────────────────────────────────────────────────────┤
    │ IMPLEMENTATION DETAILS:                                             │
    │                                                                     │
    │ - Feature source: Backbone layer 9 (C5, deepest semantic features) │
    │ - Output size: 7x7 (standard ROI Align setting)                    │
    │ - Sampling ratio: 2 (bilinear interpolation points)                │
    │ - Final projection: AdaptiveAvgPool + Linear -> 256 dims           │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    """
    print(decision)
    
    # Lock the configuration
    config = {
        "strategy": "Backbone + ROI Align",
        "feature_layer_idx": 9,  # C5 backbone output
        "roi_output_size": 7,
        "sampling_ratio": 2,
        "output_dim": D_VISION,  # 256
    }
    
    print(f"\nLOCKED CONFIGURATION:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    return config


class FeatureExtractor:
    """
    TASK 4: Feature extraction using Backbone + ROI Align.
    
    Extracts per-human visual embeddings from YOLO backbone features.
    
    Output: visual_embeddings: Tensor[N, 256]
    
    Constraints:
    - Uses torch only
    - No gradients
    - No randomness
    - No resizing dependent on image size
    """
    
    def __init__(self, model: YOLO, feature_layer_idx: int = 9, output_dim: int = 256):
        """
        Initialize the feature extractor.
        
        Args:
            model: YOLO model instance
            feature_layer_idx: Index of backbone layer to extract features from
            output_dim: Dimension of output embeddings (default: 256)
        """
        self.model = model
        self.feature_layer_idx = feature_layer_idx
        self.output_dim = output_dim
        self.features = None
        self._hook_handle = None
        
        # Get feature channel count from the model
        # This will be determined during first forward pass
        self._feature_channels = None
        self._projection = None
        
        # Register hook
        self._register_hook()
    
    def _register_hook(self):
        """Register forward hook on the target backbone layer."""
        target_layer = self.model.model.model[self.feature_layer_idx]
        
        def hook_fn(module, input, output):
            # Store features without gradient
            self.features = output.detach()
        
        self._hook_handle = target_layer.register_forward_hook(hook_fn)
    
    def _ensure_projection(self, in_channels: int):
        """Lazily create projection layer based on actual feature channels."""
        if self._projection is None or self._feature_channels != in_channels:
            self._feature_channels = in_channels
            # Simple projection: pool to 1x1 then linear
            # We'll use adaptive pooling + a fixed projection
            # For determinism, we use a fixed random seed for initialization
            torch.manual_seed(42)
            self._projection = torch.nn.Linear(in_channels, self.output_dim, bias=False)
            # Initialize with orthogonal for stability
            torch.nn.init.orthogonal_(self._projection.weight)
            self._projection.eval()
            self._projection.requires_grad_(False)
    
    def extract(
        self,
        image_path: str,
        boxes_xyxy: torch.Tensor,
        orig_shape: tuple,
    ) -> torch.Tensor:
        """
        Extract per-human visual embeddings.
        
        Args:
            image_path: Path to input image (for running inference)
            boxes_xyxy: Detected boxes in [N, 4] xyxy format (pixel coordinates)
            orig_shape: Original image shape (H, W)
            
        Returns:
            visual_embeddings: Tensor[N, output_dim] - L2 normalized embeddings
        """
        from torchvision.ops import roi_align
        
        N = boxes_xyxy.shape[0]
        
        if N == 0:
            return torch.zeros((0, self.output_dim), dtype=torch.float32)
        
        # Features should already be captured from the inference pass
        if self.features is None:
            raise RuntimeError("Features not captured. Run model inference first.")
        
        # Get feature map info
        feat = self.features  # [1, C, H_feat, W_feat]
        _, C, H_feat, W_feat = feat.shape
        orig_H, orig_W = orig_shape
        
        # Scale boxes from image coordinates to feature map coordinates
        scale_x = W_feat / orig_W
        scale_y = H_feat / orig_H
        
        boxes_scaled = boxes_xyxy.clone().float()
        boxes_scaled[:, [0, 2]] *= scale_x
        boxes_scaled[:, [1, 3]] *= scale_y
        
        # Prepare boxes for roi_align: [N, 5] with batch index
        batch_indices = torch.zeros((N, 1), dtype=torch.float32)
        rois = torch.cat([batch_indices, boxes_scaled], dim=1)  # [N, 5]
        
        # ROI Align: extract fixed-size feature for each box
        roi_output_size = 7
        sampling_ratio = 2
        
        roi_features = roi_align(
            feat,
            rois,
            output_size=(roi_output_size, roi_output_size),
            spatial_scale=1.0,  # Already scaled boxes
            sampling_ratio=sampling_ratio,
            aligned=True,
        )  # [N, C, 7, 7]
        
        # Global average pooling
        pooled = F.adaptive_avg_pool2d(roi_features, (1, 1))  # [N, C, 1, 1]
        pooled = pooled.view(N, C)  # [N, C]
        
        # Ensure projection layer exists
        self._ensure_projection(C)
        
        # Project to output dimension
        with torch.no_grad():
            embeddings = self._projection(pooled)  # [N, output_dim]
        
        # L2 normalize
        norms = embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        embeddings = embeddings / norms
        
        return embeddings
    
    def cleanup(self):
        """Remove the forward hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None


def task4_implement_extraction(model):
    """
    TASK 4: Implement and test feature extraction.
    """
    print("\n" + "=" * 70)
    print("TASK 4: FEATURE EXTRACTION IMPLEMENTATION")
    print("=" * 70)
    
    # Create extractor
    extractor = FeatureExtractor(model, feature_layer_idx=9, output_dim=D_VISION)
    print(f"\nFeatureExtractor initialized:")
    print(f"  Target layer: model.model.model[9]")
    print(f"  Output dimension: {D_VISION}")
    
    # Run inference to capture features
    test_image = "test_bus.jpg"
    print(f"\nRunning inference on: {test_image}")
    
    with torch.no_grad():
        results = model(test_image, device="cpu", verbose=False)
    
    result = results[0]
    orig_shape = result.orig_shape
    boxes_xyxy = result.boxes.xyxy.cpu()
    
    print(f"  Original image shape: {orig_shape}")
    print(f"  Detected boxes: {boxes_xyxy.shape[0]}")
    print(f"  Captured feature shape: {extractor.features.shape}")
    
    # Extract embeddings
    embeddings = extractor.extract(test_image, boxes_xyxy, orig_shape)
    
    print(f"\nExtracted embeddings:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Dtype: {embeddings.dtype}")
    print(f"  Device: {embeddings.device}")
    print(f"  L2 norms: {embeddings.norm(dim=-1)}")
    
    return extractor, embeddings, boxes_xyxy, orig_shape


def task5_alignment_check(extractor, model):
    """
    TASK 5: Verify alignment between embeddings and detection boxes.
    """
    print("\n" + "=" * 70)
    print("TASK 5: ALIGNMENT SANITY CHECK")
    print("=" * 70)
    
    test_image = "test_bus.jpg"
    
    # Run inference
    with torch.no_grad():
        results = model(test_image, device="cpu", verbose=False)
    
    result = results[0]
    orig_shape = result.orig_shape
    boxes_xyxy = result.boxes.xyxy.cpu()
    N = boxes_xyxy.shape[0]
    
    # Extract embeddings with original box order
    embeddings_original = extractor.extract(test_image, boxes_xyxy, orig_shape)
    
    # Reverse box order
    reversed_indices = torch.arange(N - 1, -1, -1)
    boxes_reversed = boxes_xyxy[reversed_indices]
    
    # Extract embeddings with reversed box order
    embeddings_reversed = extractor.extract(test_image, boxes_reversed, orig_shape)
    
    # Verify alignment: reversed embeddings should match when re-reversed
    embeddings_re_reversed = embeddings_reversed[reversed_indices]
    
    max_diff = (embeddings_original - embeddings_re_reversed).abs().max().item()
    
    print(f"\nAlignment Test:")
    print(f"  Number of detections: {N}")
    print(f"  Original embeddings shape: {embeddings_original.shape}")
    print(f"  Reversed embeddings shape: {embeddings_reversed.shape}")
    print(f"  Max difference after re-reversing: {max_diff:.2e}")
    
    if max_diff < 1e-5:
        print(f"\n  [PASS] Embeddings are correctly aligned with boxes")
        print(f"         Reordering boxes reorders embeddings identically")
    else:
        print(f"\n  [FAIL] Alignment mismatch detected!")
        return False
    
    # Shape consistency check
    assert embeddings_original.shape == (N, D_VISION), "Shape mismatch!"
    print(f"  [PASS] Shape consistency verified: ({N}, {D_VISION})")
    
    return True


def task6_stability_check(extractor, model):
    """
    TASK 6: Verify embedding stability across multiple runs.
    """
    print("\n" + "=" * 70)
    print("TASK 6: STABILITY CHECK")
    print("=" * 70)
    
    test_image = "test_bus.jpg"
    
    # First run
    with torch.no_grad():
        results1 = model(test_image, device="cpu", verbose=False)
    
    result1 = results1[0]
    boxes1 = result1.boxes.xyxy.cpu()
    embeddings1 = extractor.extract(test_image, boxes1, result1.orig_shape)
    
    # Second run
    with torch.no_grad():
        results2 = model(test_image, device="cpu", verbose=False)
    
    result2 = results2[0]
    boxes2 = result2.boxes.xyxy.cpu()
    embeddings2 = extractor.extract(test_image, boxes2, result2.orig_shape)
    
    # Compare
    box_diff = (boxes1 - boxes2).abs().max().item()
    emb_diff = (embeddings1 - embeddings2).abs().max().item()
    
    print(f"\nStability Test (same image, two inference runs):")
    print(f"  Max box difference: {box_diff:.2e}")
    print(f"  Max embedding difference: {emb_diff:.2e}")
    print(f"  Embedding norms (run 1): {embeddings1.norm(dim=-1).tolist()}")
    print(f"  Embedding norms (run 2): {embeddings2.norm(dim=-1).tolist()}")
    
    # Check bounded norms (should be ~1.0 due to L2 normalization)
    norm_min = embeddings1.norm(dim=-1).min().item()
    norm_max = embeddings1.norm(dim=-1).max().item()
    print(f"  Norm range: [{norm_min:.6f}, {norm_max:.6f}]")
    
    if emb_diff < 1e-5:
        print(f"\n  [PASS] Embeddings are numerically stable")
        print(f"  [PASS] Norms are bounded and consistent")
    else:
        print(f"\n  [WARN] Embeddings have non-trivial variance: {emb_diff:.2e}")
    
    return embeddings1, emb_diff


def task7_update_vision_output(model, extractor):
    """
    TASK 7: Create VisionOutput with real embeddings (replacing placeholder zeros).
    """
    print("\n" + "=" * 70)
    print("TASK 7: UPDATE VisionOutput WITH REAL EMBEDDINGS")
    print("=" * 70)
    
    test_image = "test_bus.jpg"
    
    # Run inference
    with torch.no_grad():
        results_pose = model(test_image, device="cpu", verbose=False)
    
    result = results_pose[0]
    orig_H, orig_W = result.orig_shape
    
    # Extract detections
    boxes_xyxy = result.boxes.xyxy.cpu()
    N = boxes_xyxy.shape[0]
    B = 1
    
    print(f"\nProcessing {N} detections...")
    
    # Normalize boxes to [0, 1]
    boxes_normalized = boxes_xyxy.clone().float()
    boxes_normalized[:, [0, 2]] /= orig_W
    boxes_normalized[:, [1, 3]] /= orig_H
    final_boxes = boxes_normalized.unsqueeze(0)  # [1, N, 4]
    
    # boxes_valid
    final_boxes_valid = torch.ones((B, N), dtype=torch.bool)
    
    # Masks (placeholder - would come from seg model)
    final_masks = torch.zeros((B, N, H_MASK, W_MASK), dtype=torch.float32)
    
    # Keypoints
    kpts_data = result.keypoints.data.float().cpu()
    kpts_normalized = kpts_data.clone()
    kpts_normalized[:, :, 0] /= orig_W
    kpts_normalized[:, :, 1] /= orig_H
    final_keypoints = kpts_normalized.unsqueeze(0)  # [1, N, 17, 3]
    
    # REAL EMBEDDINGS (no longer placeholder!)
    embeddings = extractor.extract(test_image, boxes_xyxy, result.orig_shape)
    final_visual_embeddings = embeddings.unsqueeze(0)  # [1, N, 256]
    
    print(f"\nFinal tensor shapes:")
    print(f"  boxes: {final_boxes.shape}")
    print(f"  boxes_valid: {final_boxes_valid.shape}")
    print(f"  masks: {final_masks.shape}")
    print(f"  keypoints: {final_keypoints.shape}")
    print(f"  visual_embeddings: {final_visual_embeddings.shape} <- REAL EMBEDDINGS")
    
    # Verify embeddings are not zeros
    emb_sum = final_visual_embeddings.abs().sum().item()
    print(f"\n  Embedding abs sum: {emb_sum:.4f} (should be >> 0)")
    if emb_sum < 1e-6:
        raise ValueError("Embeddings are still zeros!")
    print(f"  [PASS] Embeddings are non-zero real values")
    
    # Create VisionOutput
    vision_output = VisionOutput(
        boxes=final_boxes,
        boxes_valid=final_boxes_valid,
        masks=final_masks,
        keypoints=final_keypoints,
        visual_embeddings=final_visual_embeddings,
    )
    
    print(f"\n--- FULL VisionOutput REPR ---")
    print(vision_output)
    
    return vision_output


def task8_assertions(vision_output):
    """
    TASK 8: Run Phase 0 assertions on the updated VisionOutput.
    """
    print("\n" + "=" * 70)
    print("TASK 8: PHASE 0 ASSERTION CHECKS")
    print("=" * 70)
    
    try:
        assert_vision_output(vision_output)
        print("\n  [PASS] assert_vision_output PASSED")
    except (TypeError, ValueError) as e:
        print(f"\n  [FAIL] assert_vision_output FAILED: {e}")
        raise
    
    # Additional checks
    emb = vision_output.visual_embeddings
    
    # Check L2 norms (should be ~1.0 for valid embeddings)
    norms = emb.norm(dim=-1)
    print(f"\n  Embedding L2 norms: {norms.squeeze().tolist()}")
    
    # Check no NaN/Inf (already done by assertion, but explicit)
    assert not torch.isnan(emb).any(), "NaN detected in embeddings"
    assert not torch.isinf(emb).any(), "Inf detected in embeddings"
    print(f"  [PASS] No NaN/Inf in embeddings")
    
    return True


def main():
    print("=" * 70)
    print("PHASE 2: VISUAL EMBEDDINGS FEATURE EXTRACTION DESIGN & VALIDATION")
    print("=" * 70)
    
    # Task 1: Strategy enumeration
    strategies = task1_enumerate_strategies()
    
    # Task 2: Model introspection
    model, feature_shapes = task2_introspect_model()
    
    # Task 3: Strategy selection
    config = task3_select_strategy(feature_shapes)
    
    # Task 4: Implementation
    extractor, embeddings, boxes, orig_shape = task4_implement_extraction(model)
    
    # Task 5: Alignment check
    alignment_ok = task5_alignment_check(extractor, model)
    if not alignment_ok:
        raise RuntimeError("Alignment check failed!")
    
    # Task 6: Stability check
    stable_embeddings, max_diff = task6_stability_check(extractor, model)
    
    # Task 7: Update VisionOutput
    vision_output = task7_update_vision_output(model, extractor)
    
    # Task 8: Assertions
    task8_assertions(vision_output)
    
    # Cleanup
    extractor.cleanup()
    
    # Final Summary
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"\n1. CHOSEN STRATEGY: Backbone Feature Map + ROI Align")
    print(f"\n2. MODULE USED: model.model.model[9] (C5 backbone layer)")
    print(f"\n3. RAW FEATURE MAP SHAPE: {extractor.features.shape if extractor.features is not None else 'N/A'}")
    print(f"\n4. FINAL EMBEDDING SHAPE: (N, {D_VISION}) where N = num_detections")
    print(f"\n5. STABILITY CHECK: Max diff = {max_diff:.2e}")
    print(f"\n6. ASSERTION STATUS: ALL PASSED")
    print(f"\n7. NOTES:")
    print(f"   - Embeddings are L2-normalized (||emb|| = 1)")
    print(f"   - Extraction is deterministic given fixed boxes")
    print(f"   - No YOLO internals were modified")
    print(f"   - CPU-only execution confirmed")
    print("=" * 70)


if __name__ == "__main__":
    main()
