# RefYOLO-Human: System Status Report

**Date:** 2026-01-23  
**Status:** Architecture Frozen. Inference Complete. Training Interfaces Defined.

---

## 1. Project Status (What Is Done)

- **Phase 0:** Repository structure, contracts, datatypes, assertions
- **Phase 1:** YOLO model loading (pose + seg), inference, VisionOutput mapping
- **Phase 2:** Feature extraction via ROI Align, stability verified
- **Phase 3:** Structural embeddings, HumanToken (256D) construction
- **Phase 4:** Query encoding using MiniLM-L6-v2, D_QUERY=256
- **Phase 5:** Dynamic Grounding Adapter (FiLM + gating mechanism)
- **Phase 6:** LLM Scorer (MLP-based scoring head)
- **Phase 7:** Selection and rejection logic
- **Phase 8:** Complete end-to-end inference pipeline
- **Phase 9:** Runtime instrumentation, failure diagnostics, GPU-readiness checks
- **Phase 10:** Training interfaces (MIRL loss, supervision signals)

**Architecture is frozen. No structural changes permitted.**

---

## 2. System Architecture Overview

RefYOLO-Human is a referring expression grounding system specialized for human detection. It uses YOLO11 for human detection (pose and segmentation), extracts visual and structural features into a unified 256-dimensional token space, and grounds natural language queries via a FiLM-conditioned adapter. A lightweight MLP scorer produces per-human matching scores. The system explicitly supports rejection when no human matches the query. Inference and training paths are strictly separated.

---

## 3. Data Flow (Step-by-Step)

1. **Image + Query** — Raw RGB image and natural language query enter the system.
2. **YOLO (pose + seg)** — YOLO11n-pose and YOLO11n-seg detect humans, yielding bboxes, keypoints, and masks.
3. **VisionOutput** — Detections unified into structured VisionOutput dataclass.
4. **Visual Embeddings** — ROI Align extracts 256D visual features per human from YOLO backbone.
5. **Structural Embeddings** — Keypoints and spatial info encoded into 256D structural features.
6. **HumanToken (256D)** — Visual and structural embeddings fused into unified human representation.
7. **QueryEmbedding (256D)** — MiniLM-L6-v2 encodes query text, projected to 256D.
8. **Dynamic Grounding Adapter** — FiLM-style conditioning modulates human tokens with query embedding.
9. **Scoring Head** — MLP produces scalar matching score per human.
10. **Selection / Rejection** — Thresholding selects best-matching human(s) or rejects if none qualify.

---

## 4. Core Design Decisions (Locked)

- ROI Align on YOLO backbone features for visual embeddings
- FiLM-style modulation for query-to-vision grounding
- MLP scorer (no autoregressive generation)
- MIRL (Multi-Instance Rejection Loss) for supervision with rejection support
- 256D unified token space across all representations
- Deterministic execution, CPU-first design
- Strict separation of inference pipeline and training loss modules
- No external LLM dependencies at inference time

---

## 5. What Is Not Done Yet (Explicit)

- GPU migration
- Dataset loaders (RefCOCO, RefCOCO+, RefCOCOg)
- Training loop execution
- Benchmark evaluation
- Ablation studies
- Paper writing

---

## 6. Next Immediate Steps

1. GPU migration — Move pipeline and loss to CUDA.
2. Data loading — Implement dataset loaders for RefCOCO family.
3. Training — Execute MIRL-supervised training runs.
4. Evaluation — Benchmark on standard referring expression datasets.

---

*End of status report.*
