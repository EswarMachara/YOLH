# RefYOLO-Human

## 1. Project Overview

RefYOLO-Human is a human-centric referring expression grounding system. Given an image and a natural language query describing a person, it identifies the matching human and outputs their bounding box, segmentation mask, and body keypoints. The system reasons at the human level rather than the pixel level, and explicitly handles rejection when no human matches the query.

---

## 2. Architecture Overview

The system detects all humans in an image using YOLO, constructs a unified 256-dimensional representation for each human, encodes the query into the same embedding space, and uses a query-conditioned adapter to compute matching scores. A lightweight scoring head outputs per-human scores, and selection logic picks the best match or rejects.

- **YOLO (pose + segmentation):** Detects humans and extracts bboxes, keypoints, and masks.
- **Visual feature extraction (ROI Align):** Extracts 256D visual features from YOLO backbone per human.
- **Structural embeddings:** Encodes spatial layout, pose structure, and mask shape into 256D.
- **HumanToken (256D):** Fuses visual and structural embeddings into a unified human representation.
- **QueryEmbedding (256D):** Encodes natural language query via sentence transformer, projected to 256D.
- **Dynamic Grounding Adapter:** FiLM-style conditioning modulates human tokens with query embedding.
- **Scoring head:** MLP produces a scalar matching score per human.
- **Selection / rejection logic:** Thresholding selects best-matching human or rejects if none qualify.

---

## 3. Data Flow

1. **Image + Query** — A raw RGB image and a natural language query enter the system.
2. **Human detection (YOLO)** — YOLO11n-pose and YOLO11n-seg detect all humans, yielding bboxes, keypoints, and masks.
3. **Per-human feature extraction** — ROI Align extracts visual features from YOLO backbone for each detected human.
4. **HumanToken construction** — Visual and structural embeddings are fused into a 256D token per human.
5. **Query encoding** — The query is encoded by a sentence transformer and projected to 256D.
6. **Query-conditioned grounding** — The Dynamic Grounding Adapter modulates human tokens using the query embedding.
7. **Scoring** — An MLP scoring head computes a scalar matching score for each human.
8. **Selection / rejection** — The highest-scoring human is selected if above threshold; otherwise, the system rejects.

---

## 4. Design Principles

- **Human-level reasoning:** The system operates on detected humans, not raw pixels.
- **Deterministic inference:** No sampling or stochastic behavior at inference time.
- **Separation of inference and training:** Loss modules are isolated from the inference pipeline.
- **Explicit rejection handling:** The system can reject queries when no human matches.
- **Architecture-first design:** Core architecture is independent of dataset-specific curation.

---

## 5. Planned Next Steps

- Dataset curation and filtering for human-centric referring expressions
- GPU migration for training and fast inference
- Training with MIRL (Multi-Instance Rejection Loss)
- Benchmarking on RefCOCO, RefCOCO+, and RefCOCOg
- Video-level extensions for temporal grounding
