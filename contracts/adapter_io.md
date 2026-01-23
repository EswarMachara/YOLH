# Adapter I/O Contract

This document defines the formal tensor specifications for all inputs and outputs of the adapter module, which bridges vision outputs to LLM-compatible representations.

---

## 1. HumanToken (Input)

### Description
Tokenized representation of a detected human, derived from vision outputs.

### Tensor Name
`human_tokens`

### Shape
`(B, N, D_token)`

- `B`: Batch size (≥ 1)
- `N`: Number of detected humans per image (≥ 0, matches vision output)
- `D_token`: Token dimensionality (fixed at 512)

### Data Type
`torch.float32`

### Composition
Each human token is a concatenation/projection of:
- Visual embedding: `D_vision = 256`
- Spatial encoding (from boxes): `128`
- Pose encoding (from keypoints): `128`

Total: `256 + 128 + 128 = 512` → projected to `D_token = 512`

### Value Range
- Arbitrary floating-point values (no normalization required at this stage)
- No NaN or Inf values permitted

### Batch Semantics
- Padded tokens (for invalid detections) are zero vectors
- A validity mask `human_tokens_valid` of shape `(B, N)` with dtype `torch.bool` must accompany the tensor

---

## 2. QueryEmbedding (Input)

### Description
Encoded representation of the natural language referring expression.

### Tensor Name
`query_embedding`

### Shape
`(B, D_query)`

- `B`: Batch size (≥ 1)
- `D_query`: Query embedding dimensionality (fixed at 512)

### Data Type
`torch.float32`

### Value Range
- L2-normalized: `||embedding||_2 = 1.0` (within tolerance of 1e-6)
- No NaN or Inf values permitted

### Semantics
- One query embedding per batch element (one referring expression per image)
- If multiple queries per image are needed, batch dimension must be expanded accordingly

### Failure Behavior
- Empty or invalid query text results in a zero vector (norm = 0)
- Downstream modules must check for zero-norm query embeddings

---

## 3. Adapter Output

### Description
Fused representation combining human tokens with query context, ready for LLM processing.

### Tensor Name
`adapter_output`

### Shape
`(B, N, D_fused)`

- `B`: Batch size (≥ 1)
- `N`: Number of detected humans (matches input)
- `D_fused`: Fused representation dimensionality (fixed at 768)

### Data Type
`torch.float32`

### Value Range
- Arbitrary floating-point values
- No NaN or Inf values permitted

### Semantics
- Each `(b, n)` entry represents the fused representation of human `n` in image `b`, contextualized by the query
- The adapter performs cross-attention or concatenation-based fusion (implementation detail)

### Batch Semantics
- Padded outputs (for invalid human tokens) are zero vectors
- Validity mask is propagated from input: `adapter_output_valid` has shape `(B, N)` and dtype `torch.bool`

---

## Dimensionality Constants

| Constant     | Value | Description                           |
|--------------|-------|---------------------------------------|
| `D_vision`   | 256   | Visual embedding dimension            |
| `D_spatial`  | 128   | Spatial (box) encoding dimension      |
| `D_pose`     | 128   | Pose (keypoint) encoding dimension    |
| `D_token`    | 512   | Human token dimension                 |
| `D_query`    | 512   | Query embedding dimension             |
| `D_fused`    | 768   | Adapter output dimension              |

---

## Allowed Batch Sizes

| Scenario          | Minimum `B` | Maximum `B` | Notes                        |
|-------------------|-------------|-------------|------------------------------|
| Training          | 1           | 64          | GPU memory dependent         |
| Inference (CPU)   | 1           | 16          | Deterministic execution      |
| Single-image      | 1           | 1           | Typical deployment case      |

---

## Failure Behavior

### Zero Humans Detected (`N = 0`)
- `human_tokens` has shape `(B, 0, D_token)` — valid empty tensor
- `adapter_output` has shape `(B, 0, D_fused)` — valid empty tensor
- No error is raised; downstream modules must handle empty detection gracefully
- Grounding result should return empty matches

### Invalid Input Shapes
- If `human_tokens` shape does not match `(B, N, 512)`, raise `ValueError`
- If `query_embedding` shape does not match `(B, 512)`, raise `ValueError`
- If batch dimensions `B` do not match between inputs, raise `ValueError`

### NaN/Inf Detection
- If any input contains NaN or Inf, raise `ValueError` immediately
- No partial processing occurs on invalid input

### Dtype Mismatch
- All inputs must be `torch.float32`
- If dtype does not match, raise `TypeError`
