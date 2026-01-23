# Vision Outputs Contract

This document defines the formal tensor specifications for all outputs produced by the vision module.

---

## 1. Bounding Boxes

### Tensor Name
`boxes`

### Shape
`(B, N, 4)`

- `B`: Batch size (≥ 1)
- `N`: Number of detected humans per image (≥ 0)
- `4`: Bounding box coordinates

### Data Type
`torch.float32`

### Coordinate Convention
- Format: `[x_min, y_min, x_max, y_max]`
- Coordinate space: Normalized to `[0.0, 1.0]` relative to image dimensions
- `x_min < x_max` and `y_min < y_max` must hold for valid boxes
- Origin: Top-left corner of the image

### Batch Semantics
- Each batch element corresponds to one input image
- `N` is fixed across the batch (padded with zeros if fewer detections)
- A validity mask tensor `boxes_valid` of shape `(B, N)` with dtype `torch.bool` indicates which boxes are real detections vs. padding

---

## 2. Segmentation Masks

### Tensor Name
`masks`

### Shape
`(B, N, H_mask, W_mask)`

- `B`: Batch size (≥ 1)
- `N`: Number of detected humans (matches `boxes`)
- `H_mask`: Mask height (fixed, e.g., 160)
- `W_mask`: Mask width (fixed, e.g., 160)

### Data Type
`torch.float32`

### Value Range
- Values in `[0.0, 1.0]` representing soft segmentation probabilities
- `0.0` = background, `1.0` = foreground (human)

### Coordinate Convention
- Mask is aligned to the corresponding bounding box region
- Mask coordinates are relative to the bounding box, not the full image
- Bilinear interpolation assumed for upsampling to original resolution

### Batch Semantics
- Padded masks (for invalid detections) are filled with `0.0`

---

## 3. Keypoints

### Tensor Name
`keypoints`

### Shape
`(B, N, K, 3)`

- `B`: Batch size (≥ 1)
- `N`: Number of detected humans (matches `boxes`)
- `K`: Number of keypoints (fixed at 17 for COCO skeleton)
- `3`: `[x, y, confidence]`

### Data Type
`torch.float32`

### Keypoint Order (COCO Convention)
```
0:  nose
1:  left_eye
2:  right_eye
3:  left_ear
4:  right_ear
5:  left_shoulder
6:  right_shoulder
7:  left_elbow
8:  right_elbow
9:  left_wrist
10: right_wrist
11: left_hip
12: right_hip
13: left_knee
14: right_knee
15: left_ankle
16: right_ankle
```

### Coordinate Convention
- `x`, `y`: Normalized to `[0.0, 1.0]` relative to the **bounding box** of the corresponding detection
- `confidence`: Value in `[0.0, 1.0]` indicating keypoint visibility/confidence
- Invisible keypoints have `confidence = 0.0` and `x = y = 0.0`

### Batch Semantics
- Padded keypoints (for invalid detections) are filled with `0.0` for all values

---

## 4. Visual Embeddings

### Tensor Name
`visual_embeddings`

### Shape
`(B, N, D_vision)`

- `B`: Batch size (≥ 1)
- `N`: Number of detected humans (matches `boxes`)
- `D_vision`: Embedding dimensionality (fixed at 256)

### Data Type
`torch.float32`

### Value Range
- L2-normalized embeddings: `||embedding||_2 = 1.0` (within floating-point tolerance of 1e-6)
- No NaN or Inf values permitted

### Semantics
- Each embedding encodes the visual appearance of the corresponding detected human
- Embeddings are extracted from the vision backbone (e.g., ROI-pooled features)
- Used downstream for fusion with language representations

### Batch Semantics
- Padded embeddings (for invalid detections) are zero vectors `[0.0, 0.0, ..., 0.0]`
- Zero vectors are explicitly **not** L2-normalized (norm = 0)

---

## Summary Table

| Tensor             | Shape               | dtype       | Value Range       |
|--------------------|---------------------|-------------|-------------------|
| `boxes`            | `(B, N, 4)`         | `float32`   | `[0.0, 1.0]`      |
| `boxes_valid`      | `(B, N)`            | `bool`      | `{True, False}`   |
| `masks`            | `(B, N, H, W)`      | `float32`   | `[0.0, 1.0]`      |
| `keypoints`        | `(B, N, 17, 3)`     | `float32`   | `[0.0, 1.0]`      |
| `visual_embeddings`| `(B, N, 256)`       | `float32`   | L2-norm = 1 or 0  |

---

## Failure Behavior

### Zero Detections
- When no humans are detected in an image, `N = 0` for that batch element
- If batched with other images that have detections, padding is applied to match the maximum `N`
- The `boxes_valid` mask must be checked before accessing any detection data

### Invalid Input
- If the vision module receives invalid input (wrong dtype, wrong shape), it must raise `ValueError` immediately
- No partial outputs are returned on failure
