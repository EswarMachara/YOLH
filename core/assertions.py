"""
Hard assertion functions for contract validation.

This module provides strict runtime validation for all dataclass contracts.
Assertions check tensor rank, shape consistency, dtype, and absence of NaN/Inf.
All functions raise exceptions on failure; no logging is performed.
"""

import torch

from core.datatypes import (
    VisionOutput,
    HumanToken,
    QueryEmbedding,
    GroundingScores,
    D_VISION,
    D_TOKEN,
    D_QUERY,
    H_MASK,
    W_MASK,
    K_KEYPOINTS,
)


def _assert_no_nan_inf(tensor: torch.Tensor, name: str) -> None:
    """
    Assert that a tensor contains no NaN or Inf values.

    Args:
        tensor: The tensor to check.
        name: Name of the tensor for error messages.

    Raises:
        ValueError: If tensor contains NaN or Inf values.
    """
    if torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN values")
    if torch.isinf(tensor).any():
        raise ValueError(f"{name} contains Inf values")


def _assert_dtype(tensor: torch.Tensor, expected_dtype: torch.dtype, name: str) -> None:
    """
    Assert that a tensor has the expected dtype.

    Args:
        tensor: The tensor to check.
        expected_dtype: The expected dtype.
        name: Name of the tensor for error messages.

    Raises:
        TypeError: If tensor dtype does not match expected.
    """
    if tensor.dtype != expected_dtype:
        raise TypeError(
            f"{name} has dtype {tensor.dtype}, expected {expected_dtype}"
        )


def _assert_rank(tensor: torch.Tensor, expected_rank: int, name: str) -> None:
    """
    Assert that a tensor has the expected rank (number of dimensions).

    Args:
        tensor: The tensor to check.
        expected_rank: The expected number of dimensions.
        name: Name of the tensor for error messages.

    Raises:
        ValueError: If tensor rank does not match expected.
    """
    if tensor.ndim != expected_rank:
        raise ValueError(
            f"{name} has rank {tensor.ndim}, expected {expected_rank}"
        )


def _assert_shape_dim(
    tensor: torch.Tensor, dim: int, expected_size: int, name: str
) -> None:
    """
    Assert that a specific dimension of a tensor has the expected size.

    Args:
        tensor: The tensor to check.
        dim: The dimension index to check.
        expected_size: The expected size of the dimension.
        name: Name of the tensor for error messages.

    Raises:
        ValueError: If dimension size does not match expected.
    """
    if tensor.shape[dim] != expected_size:
        raise ValueError(
            f"{name} has size {tensor.shape[dim]} at dim {dim}, expected {expected_size}"
        )


def assert_vision_output(vision_output: VisionOutput) -> None:
    """
    Validate a VisionOutput dataclass against the contract.

    Checks:
        - boxes: rank 3, shape (B, N, 4), float32, no NaN/Inf
        - boxes_valid: rank 2, shape (B, N), bool
        - masks: rank 4, shape (B, N, H_MASK, W_MASK), float32, no NaN/Inf
        - keypoints: rank 4, shape (B, N, K_KEYPOINTS, 3), float32, no NaN/Inf
        - visual_embeddings: rank 3, shape (B, N, D_VISION), float32, no NaN/Inf
        - Batch and detection dimensions are consistent across all tensors

    Args:
        vision_output: The VisionOutput instance to validate.

    Raises:
        TypeError: If any tensor has incorrect dtype.
        ValueError: If any tensor has incorrect shape or contains NaN/Inf.
    """
    boxes = vision_output.boxes
    boxes_valid = vision_output.boxes_valid
    masks = vision_output.masks
    keypoints = vision_output.keypoints
    visual_embeddings = vision_output.visual_embeddings

    # Boxes: (B, N, 4)
    _assert_rank(boxes, 3, "boxes")
    _assert_dtype(boxes, torch.float32, "boxes")
    _assert_shape_dim(boxes, 2, 4, "boxes")
    _assert_no_nan_inf(boxes, "boxes")

    B = boxes.shape[0]
    N = boxes.shape[1]

    if B < 1:
        raise ValueError(f"Batch size must be >= 1, got {B}")

    # boxes_valid: (B, N)
    _assert_rank(boxes_valid, 2, "boxes_valid")
    _assert_dtype(boxes_valid, torch.bool, "boxes_valid")
    _assert_shape_dim(boxes_valid, 0, B, "boxes_valid")
    _assert_shape_dim(boxes_valid, 1, N, "boxes_valid")

    # masks: (B, N, H_MASK, W_MASK)
    _assert_rank(masks, 4, "masks")
    _assert_dtype(masks, torch.float32, "masks")
    _assert_shape_dim(masks, 0, B, "masks")
    _assert_shape_dim(masks, 1, N, "masks")
    _assert_shape_dim(masks, 2, H_MASK, "masks")
    _assert_shape_dim(masks, 3, W_MASK, "masks")
    _assert_no_nan_inf(masks, "masks")

    # keypoints: (B, N, K_KEYPOINTS, 3)
    _assert_rank(keypoints, 4, "keypoints")
    _assert_dtype(keypoints, torch.float32, "keypoints")
    _assert_shape_dim(keypoints, 0, B, "keypoints")
    _assert_shape_dim(keypoints, 1, N, "keypoints")
    _assert_shape_dim(keypoints, 2, K_KEYPOINTS, "keypoints")
    _assert_shape_dim(keypoints, 3, 3, "keypoints")
    _assert_no_nan_inf(keypoints, "keypoints")

    # visual_embeddings: (B, N, D_VISION)
    _assert_rank(visual_embeddings, 3, "visual_embeddings")
    _assert_dtype(visual_embeddings, torch.float32, "visual_embeddings")
    _assert_shape_dim(visual_embeddings, 0, B, "visual_embeddings")
    _assert_shape_dim(visual_embeddings, 1, N, "visual_embeddings")
    _assert_shape_dim(visual_embeddings, 2, D_VISION, "visual_embeddings")
    _assert_no_nan_inf(visual_embeddings, "visual_embeddings")


def assert_human_token(human_token: HumanToken) -> None:
    """
    Validate a HumanToken dataclass against the contract.

    Checks:
        - tokens: rank 3, shape (B, N, D_TOKEN), float32, no NaN/Inf
        - valid: rank 2, shape (B, N), bool
        - Batch and detection dimensions are consistent

    Args:
        human_token: The HumanToken instance to validate.

    Raises:
        TypeError: If any tensor has incorrect dtype.
        ValueError: If any tensor has incorrect shape or contains NaN/Inf.
    """
    tokens = human_token.tokens
    valid = human_token.valid

    # tokens: (B, N, D_TOKEN)
    _assert_rank(tokens, 3, "tokens")
    _assert_dtype(tokens, torch.float32, "tokens")
    _assert_shape_dim(tokens, 2, D_TOKEN, "tokens")
    _assert_no_nan_inf(tokens, "tokens")

    B = tokens.shape[0]
    N = tokens.shape[1]

    if B < 1:
        raise ValueError(f"Batch size must be >= 1, got {B}")

    # valid: (B, N)
    _assert_rank(valid, 2, "valid")
    _assert_dtype(valid, torch.bool, "valid")
    _assert_shape_dim(valid, 0, B, "valid")
    _assert_shape_dim(valid, 1, N, "valid")


def assert_query_embedding(query_embedding: QueryEmbedding) -> None:
    """
    Validate a QueryEmbedding dataclass against the contract.

    Checks:
        - embedding: rank 2, shape (B, D_QUERY), float32, no NaN/Inf
        - Batch size >= 1

    Args:
        query_embedding: The QueryEmbedding instance to validate.

    Raises:
        TypeError: If tensor has incorrect dtype.
        ValueError: If tensor has incorrect shape or contains NaN/Inf.
    """
    embedding = query_embedding.embedding

    # embedding: (B, D_QUERY)
    _assert_rank(embedding, 2, "embedding")
    _assert_dtype(embedding, torch.float32, "embedding")
    _assert_shape_dim(embedding, 1, D_QUERY, "embedding")
    _assert_no_nan_inf(embedding, "embedding")

    B = embedding.shape[0]
    if B < 1:
        raise ValueError(f"Batch size must be >= 1, got {B}")


def assert_grounding_scores(grounding_scores: GroundingScores) -> None:
    """
    Validate a GroundingScores dataclass against the contract.

    Checks:
        - scores: rank 2, shape (B, N), float32, no NaN/Inf
        - rejection_score: rank 1, shape (B,), float32, no NaN/Inf
        - valid: rank 2, shape (B, N), bool
        - Batch and detection dimensions are consistent

    Args:
        grounding_scores: The GroundingScores instance to validate.

    Raises:
        TypeError: If any tensor has incorrect dtype.
        ValueError: If any tensor has incorrect shape or contains NaN/Inf.
    """
    scores = grounding_scores.scores
    rejection_score = grounding_scores.rejection_score
    valid = grounding_scores.valid

    # scores: (B, N)
    _assert_rank(scores, 2, "scores")
    _assert_dtype(scores, torch.float32, "scores")
    _assert_no_nan_inf(scores, "scores")

    B = scores.shape[0]
    N = scores.shape[1]

    if B < 1:
        raise ValueError(f"Batch size must be >= 1, got {B}")

    # rejection_score: (B,)
    _assert_rank(rejection_score, 1, "rejection_score")
    _assert_dtype(rejection_score, torch.float32, "rejection_score")
    _assert_shape_dim(rejection_score, 0, B, "rejection_score")
    _assert_no_nan_inf(rejection_score, "rejection_score")

    # valid: (B, N)
    _assert_rank(valid, 2, "valid")
    _assert_dtype(valid, torch.bool, "valid")
    _assert_shape_dim(valid, 0, B, "valid")
    _assert_shape_dim(valid, 1, N, "valid")
