"""
Core data types for cross-module tensor exchange.

This module defines immutable dataclasses representing all tensor contracts
between vision, adapter, and LLM modules. Each dataclass contains only
tensor fields with documented shapes and no logic.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class VisionOutput:
    """
    Output from the vision module containing all detected human information.

    Attributes:
        boxes: Bounding boxes for detected humans.
            Shape: (B, N, 4)
            dtype: torch.float32
            Format: [x_min, y_min, x_max, y_max] normalized to [0, 1]

        boxes_valid: Validity mask indicating real detections vs padding.
            Shape: (B, N)
            dtype: torch.bool

        masks: Segmentation masks for detected humans.
            Shape: (B, N, H_mask, W_mask) where H_mask=W_mask=160
            dtype: torch.float32
            Values: [0, 1] probability

        keypoints: Pose keypoints for detected humans.
            Shape: (B, N, K, 3) where K=17 (COCO skeleton)
            dtype: torch.float32
            Format: [x, y, confidence] normalized to [0, 1]

        visual_embeddings: Visual feature embeddings for detected humans.
            Shape: (B, N, D_vision) where D_vision=256
            dtype: torch.float32
            Constraint: L2-normalized (||emb||=1) for valid detections, zero for padding

    Shape Variables:
        B: Batch size (>= 1)
        N: Number of detected humans per image (>= 0)
        H_mask: Mask height (fixed at 160)
        W_mask: Mask width (fixed at 160)
        K: Number of keypoints (fixed at 17)
        D_vision: Visual embedding dimension (fixed at 256)
    """

    boxes: "torch.Tensor"
    boxes_valid: "torch.Tensor"
    masks: "torch.Tensor"
    keypoints: "torch.Tensor"
    visual_embeddings: "torch.Tensor"


@dataclass(frozen=True)
class HumanToken:
    """
    Tokenized representation of detected humans for adapter input.

    Attributes:
        tokens: Human token embeddings combining visual, spatial, and pose information.
            Shape: (B, N, D_token) where D_token=256
            dtype: torch.float32
            Composition: Projection of [visual_emb(256) | struct_emb(160)]

        valid: Validity mask indicating real tokens vs padding.
            Shape: (B, N)
            dtype: torch.bool

    Shape Variables:
        B: Batch size (>= 1)
        N: Number of detected humans per image (>= 0)
        D_token: Token dimension (fixed at 256)
    """

    tokens: "torch.Tensor"
    valid: "torch.Tensor"


@dataclass(frozen=True)
class QueryEmbedding:
    """
    Encoded representation of the natural language referring expression.

    Attributes:
        embedding: Query embedding vector.
            Shape: (B, D_query) where D_query=512
            dtype: torch.float32
            Constraint: L2-normalized (||emb||=1) for valid queries, zero for invalid

    Shape Variables:
        B: Batch size (>= 1)
        D_query: Query embedding dimension (fixed at 512)
    """

    embedding: "torch.Tensor"


@dataclass(frozen=True)
class GroundingScores:
    """
    Output scores from the LLM module for human-query matching.

    Attributes:
        scores: Grounding scores for each detected human.
            Shape: (B, N)
            dtype: torch.float32
            Values: Logits in (-inf, +inf); higher = better match

        rejection_score: Confidence that no human matches the query.
            Shape: (B,)
            dtype: torch.float32
            Values: Logits in (-inf, +inf)

        valid: Validity mask propagated from input.
            Shape: (B, N)
            dtype: torch.bool

    Shape Variables:
        B: Batch size (>= 1)
        N: Number of detected humans per image (>= 0)

    Decision Logic:
        If rejection_score[b] > max(scores[b, :]):
            result = NO_MATCH
        Else:
            result = argmax(scores[b, :])
    """

    scores: "torch.Tensor"
    rejection_score: "torch.Tensor"
    valid: "torch.Tensor"


# Dimensionality constants as defined in contracts
D_VISION: int = 256
D_SPATIAL: int = 128
D_POSE: int = 128
D_TOKEN: int = 256
D_QUERY: int = 512
D_FUSED: int = 768
H_MASK: int = 160
W_MASK: int = 160
K_KEYPOINTS: int = 17
