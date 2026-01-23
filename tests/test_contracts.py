"""
Contract sanity tests.

This module creates dummy tensors, instantiates all dataclasses,
and runs all assertion functions to verify contract integrity.
All tests are designed to run on CPU only.
"""

import sys
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(__file__).rsplit("tests", 1)[0].rstrip("\\/"))

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
from core.assertions import (
    assert_vision_output,
    assert_human_token,
    assert_query_embedding,
    assert_grounding_scores,
)


def create_dummy_vision_output(batch_size: int, num_detections: int) -> VisionOutput:
    """
    Create a valid dummy VisionOutput for testing.

    Args:
        batch_size: Number of images in batch (B).
        num_detections: Number of detected humans per image (N).

    Returns:
        A valid VisionOutput instance with random tensors.
    """
    # Boxes: (B, N, 4) - normalized coordinates
    boxes = torch.rand(batch_size, num_detections, 4, dtype=torch.float32)
    # Ensure x_min < x_max and y_min < y_max
    boxes[..., 0] = boxes[..., 0] * 0.4  # x_min in [0, 0.4]
    boxes[..., 1] = boxes[..., 1] * 0.4  # y_min in [0, 0.4]
    boxes[..., 2] = 0.5 + boxes[..., 2] * 0.5  # x_max in [0.5, 1.0]
    boxes[..., 3] = 0.5 + boxes[..., 3] * 0.5  # y_max in [0.5, 1.0]

    # boxes_valid: (B, N) - all valid for this test
    boxes_valid = torch.ones(batch_size, num_detections, dtype=torch.bool)

    # masks: (B, N, H_MASK, W_MASK)
    masks = torch.rand(batch_size, num_detections, H_MASK, W_MASK, dtype=torch.float32)

    # keypoints: (B, N, K_KEYPOINTS, 3)
    keypoints = torch.rand(
        batch_size, num_detections, K_KEYPOINTS, 3, dtype=torch.float32
    )

    # visual_embeddings: (B, N, D_VISION) - L2 normalized
    visual_embeddings = torch.randn(
        batch_size, num_detections, D_VISION, dtype=torch.float32
    )
    # L2 normalize along last dimension
    norms = visual_embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    visual_embeddings = visual_embeddings / norms

    return VisionOutput(
        boxes=boxes,
        boxes_valid=boxes_valid,
        masks=masks,
        keypoints=keypoints,
        visual_embeddings=visual_embeddings,
    )


def create_dummy_human_token(batch_size: int, num_detections: int) -> HumanToken:
    """
    Create a valid dummy HumanToken for testing.

    Args:
        batch_size: Number of images in batch (B).
        num_detections: Number of detected humans per image (N).

    Returns:
        A valid HumanToken instance with random tensors.
    """
    # tokens: (B, N, D_TOKEN)
    tokens = torch.randn(batch_size, num_detections, D_TOKEN, dtype=torch.float32)

    # valid: (B, N) - all valid for this test
    valid = torch.ones(batch_size, num_detections, dtype=torch.bool)

    return HumanToken(tokens=tokens, valid=valid)


def create_dummy_query_embedding(batch_size: int) -> QueryEmbedding:
    """
    Create a valid dummy QueryEmbedding for testing.

    Args:
        batch_size: Number of queries in batch (B).

    Returns:
        A valid QueryEmbedding instance with L2-normalized embedding.
    """
    # embedding: (B, D_QUERY) - L2 normalized
    embedding = torch.randn(batch_size, D_QUERY, dtype=torch.float32)
    norms = embedding.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    embedding = embedding / norms

    return QueryEmbedding(embedding=embedding)


def create_dummy_grounding_scores(
    batch_size: int, num_detections: int
) -> GroundingScores:
    """
    Create a valid dummy GroundingScores for testing.

    Args:
        batch_size: Number of images in batch (B).
        num_detections: Number of detected humans per image (N).

    Returns:
        A valid GroundingScores instance with random scores.
    """
    # scores: (B, N) - logits
    scores = torch.randn(batch_size, num_detections, dtype=torch.float32)

    # rejection_score: (B,)
    rejection_score = torch.randn(batch_size, dtype=torch.float32)

    # valid: (B, N) - all valid for this test
    valid = torch.ones(batch_size, num_detections, dtype=torch.bool)

    return GroundingScores(scores=scores, rejection_score=rejection_score, valid=valid)


def test_vision_output_valid() -> None:
    """Test that valid VisionOutput passes assertions."""
    vision_output = create_dummy_vision_output(batch_size=2, num_detections=5)
    assert_vision_output(vision_output)
    print("[PASS] test_vision_output_valid")


def test_vision_output_zero_detections() -> None:
    """Test that VisionOutput with zero detections passes assertions."""
    vision_output = create_dummy_vision_output(batch_size=2, num_detections=0)
    assert_vision_output(vision_output)
    print("[PASS] test_vision_output_zero_detections")


def test_human_token_valid() -> None:
    """Test that valid HumanToken passes assertions."""
    human_token = create_dummy_human_token(batch_size=2, num_detections=5)
    assert_human_token(human_token)
    print("[PASS] test_human_token_valid")


def test_human_token_zero_detections() -> None:
    """Test that HumanToken with zero detections passes assertions."""
    human_token = create_dummy_human_token(batch_size=2, num_detections=0)
    assert_human_token(human_token)
    print("[PASS] test_human_token_zero_detections")


def test_query_embedding_valid() -> None:
    """Test that valid QueryEmbedding passes assertions."""
    query_embedding = create_dummy_query_embedding(batch_size=2)
    assert_query_embedding(query_embedding)
    print("[PASS] test_query_embedding_valid")


def test_grounding_scores_valid() -> None:
    """Test that valid GroundingScores passes assertions."""
    grounding_scores = create_dummy_grounding_scores(batch_size=2, num_detections=5)
    assert_grounding_scores(grounding_scores)
    print("[PASS] test_grounding_scores_valid")


def test_grounding_scores_zero_detections() -> None:
    """Test that GroundingScores with zero detections passes assertions."""
    grounding_scores = create_dummy_grounding_scores(batch_size=2, num_detections=0)
    assert_grounding_scores(grounding_scores)
    print("[PASS] test_grounding_scores_zero_detections")


def test_vision_output_invalid_nan() -> None:
    """Test that VisionOutput with NaN raises ValueError."""
    vision_output = create_dummy_vision_output(batch_size=2, num_detections=5)
    # Inject NaN
    vision_output.boxes[0, 0, 0] = float("nan")
    try:
        assert_vision_output(vision_output)
        print("[FAIL] test_vision_output_invalid_nan - expected ValueError")
        sys.exit(1)
    except ValueError as e:
        if "NaN" in str(e):
            print("[PASS] test_vision_output_invalid_nan")
        else:
            print(f"[FAIL] test_vision_output_invalid_nan - wrong error: {e}")
            sys.exit(1)


def test_human_token_invalid_shape() -> None:
    """Test that HumanToken with wrong shape raises ValueError."""
    # Create with wrong D_TOKEN
    tokens = torch.randn(2, 5, 256, dtype=torch.float32)  # Wrong: should be 512
    valid = torch.ones(2, 5, dtype=torch.bool)
    human_token = HumanToken(tokens=tokens, valid=valid)
    try:
        assert_human_token(human_token)
        print("[FAIL] test_human_token_invalid_shape - expected ValueError")
        sys.exit(1)
    except ValueError as e:
        print("[PASS] test_human_token_invalid_shape")


def test_query_embedding_invalid_dtype() -> None:
    """Test that QueryEmbedding with wrong dtype raises TypeError."""
    # Create with wrong dtype
    embedding = torch.randn(2, D_QUERY, dtype=torch.float64)  # Wrong: should be float32
    query_embedding = QueryEmbedding(embedding=embedding)
    try:
        assert_query_embedding(query_embedding)
        print("[FAIL] test_query_embedding_invalid_dtype - expected TypeError")
        sys.exit(1)
    except TypeError as e:
        print("[PASS] test_query_embedding_invalid_dtype")


def test_dataclass_immutability() -> None:
    """Test that dataclasses are frozen (immutable)."""
    vision_output = create_dummy_vision_output(batch_size=2, num_detections=5)
    try:
        vision_output.boxes = torch.zeros(2, 5, 4)
        print("[FAIL] test_dataclass_immutability - expected FrozenInstanceError")
        sys.exit(1)
    except AttributeError:
        # FrozenInstanceError is a subclass of AttributeError
        print("[PASS] test_dataclass_immutability")


def run_all_tests() -> None:
    """Run all contract tests."""
    print("=" * 60)
    print("Running Contract Sanity Tests (CPU Only)")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: CPU (CUDA available: {torch.cuda.is_available()})")
    print("=" * 60)

    # Valid input tests
    test_vision_output_valid()
    test_vision_output_zero_detections()
    test_human_token_valid()
    test_human_token_zero_detections()
    test_query_embedding_valid()
    test_grounding_scores_valid()
    test_grounding_scores_zero_detections()

    # Invalid input tests (should raise exceptions)
    test_vision_output_invalid_nan()
    test_human_token_invalid_shape()
    test_query_embedding_invalid_dtype()

    # Immutability test
    test_dataclass_immutability()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
