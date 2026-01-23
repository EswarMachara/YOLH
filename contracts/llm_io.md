# LLM I/O Contract

This document defines the formal tensor specifications for all inputs and outputs of the LLM module, which produces grounding scores for human-query matching.

---

## 1. LLM Input

### Description
The LLM module receives fused representations from the adapter, containing human tokens contextualized by the query.

### Tensor Name
`llm_input`

### Shape
`(B, N, D_fused)`

- `B`: Batch size (≥ 1)
- `N`: Number of detected humans per image (≥ 0)
- `D_fused`: Fused representation dimensionality (fixed at 768)

### Data Type
`torch.float32`

### Value Range
- Arbitrary floating-point values
- No NaN or Inf values permitted

### Accompanying Tensors

#### Validity Mask
- **Name**: `llm_input_valid`
- **Shape**: `(B, N)`
- **dtype**: `torch.bool`
- **Semantics**: `True` for real detections, `False` for padding

#### Query Context (Optional)
- **Name**: `query_context`
- **Shape**: `(B, D_query)` where `D_query = 512`
- **dtype**: `torch.float32`
- **Semantics**: Original query embedding for optional cross-attention in LLM

---

## 2. Output Score Tensor

### Description
Grounding scores indicating how well each detected human matches the referring expression.

### Tensor Name
`grounding_scores`

### Shape
`(B, N)`

- `B`: Batch size (≥ 1)
- `N`: Number of detected humans per image (≥ 0)

### Data Type
`torch.float32`

### Value Range
- Raw logits: arbitrary floating-point values in `(-∞, +∞)`
- For ranking: higher score = better match
- No NaN or Inf values in output

### Score Semantics

#### Positive Scores
- `score > 0`: Human is a potential match for the query
- Higher positive scores indicate stronger confidence

#### Negative Scores
- `score < 0`: Human is unlikely to match the query
- More negative scores indicate stronger rejection

#### Zero Score
- `score = 0`: Neutral/uncertain match (threshold boundary)

---

## 3. Rejection Score Semantics

### Description
Explicit handling of rejection cases where no detected human matches the query.

### Rejection Tensor
- **Name**: `rejection_score`
- **Shape**: `(B,)`
- **dtype**: `torch.float32`

### Semantics
- `rejection_score[b]`: Confidence that **no human** in image `b` matches the query
- Value range: `(-∞, +∞)` (logit space)

### Decision Logic

```
For each batch element b:
    max_human_score = max(grounding_scores[b, :])  # Best human match
    
    if rejection_score[b] > max_human_score:
        result = NO_MATCH  # Query refers to no one in the image
    else:
        result = argmax(grounding_scores[b, :])  # Index of best-matching human
```

### Special Cases

#### Zero Detections (`N = 0`)
- `grounding_scores` has shape `(B, 0)` — valid empty tensor
- `rejection_score` is the only output used for decision
- Result is always `NO_MATCH`

#### Tie Breaking
- If multiple humans have the same maximum score, return the **lowest index**
- Deterministic behavior required for reproducibility

---

## 4. Confidence Calibration

### Softmax Normalization (Optional Post-Processing)

For probabilistic interpretation, scores can be normalized:

```
combined_scores = concat([grounding_scores, rejection_score.unsqueeze(-1)], dim=-1)
# Shape: (B, N+1)

probabilities = softmax(combined_scores, dim=-1)
# probabilities[:, :-1] = P(human_i matches query)
# probabilities[:, -1]  = P(no match)
```

### Temperature Scaling
- Default temperature: `τ = 1.0`
- Lower temperature → sharper distributions (more confident)
- Higher temperature → softer distributions (less confident)

---

## Summary Table

| Tensor             | Shape         | dtype     | Value Range        | Required |
|--------------------|---------------|-----------|--------------------| ---------|
| `llm_input`        | `(B, N, 768)` | `float32` | `(-∞, +∞)`         | Yes      |
| `llm_input_valid`  | `(B, N)`      | `bool`    | `{True, False}`    | Yes      |
| `query_context`    | `(B, 512)`    | `float32` | L2-norm = 1 or 0   | Optional |
| `grounding_scores` | `(B, N)`      | `float32` | `(-∞, +∞)`         | Yes      |
| `rejection_score`  | `(B,)`        | `float32` | `(-∞, +∞)`         | Yes      |

---

## Dimensionality Constants

| Constant   | Value | Description                    |
|------------|-------|--------------------------------|
| `D_fused`  | 768   | LLM input feature dimension    |
| `D_query`  | 512   | Query context dimension        |

---

## Failure Behavior

### Invalid Input Shape
- If `llm_input` shape is not `(B, N, 768)`, raise `ValueError`
- If `llm_input_valid` shape does not match `(B, N)`, raise `ValueError`

### NaN/Inf in Input
- If any input tensor contains NaN or Inf, raise `ValueError`
- LLM module must not produce NaN/Inf in outputs

### Empty Detection (`N = 0`)
- Valid scenario; `grounding_scores` has shape `(B, 0)`
- Only `rejection_score` is meaningful
- No error raised

### Dtype Mismatch
- All float tensors must be `torch.float32`
- Validity mask must be `torch.bool`
- Raise `TypeError` on mismatch

---

## Output Guarantees

1. **Determinism**: Same input always produces same output (CPU execution)
2. **Finiteness**: No NaN or Inf in any output tensor
3. **Shape Consistency**: Output shapes always match input batch/detection dimensions
4. **Validity Propagation**: Invalid inputs (padding) produce score of `-inf` in `grounding_scores`
