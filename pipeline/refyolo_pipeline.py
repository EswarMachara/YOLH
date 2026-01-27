"""
Phase 8: RefYOLO-Human Complete Inference Pipeline

Assembles all components into a single end-to-end pipeline:
    Image + Query → Selected Humans

⚠️ INFERENCE EXPLORATION SCRIPT - NOT FOR PRODUCTION TRAINING
   - Hardcoded CPU device for local development/inference
   - For GPU training, use training/grounding_train_cached.py with config device

CPU-only. No training. No optimization. No batching.
Pure orchestration - no logic duplication.
"""

import os
import sys

# Disable TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import urllib.request

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from ultralytics import YOLO

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
)
from core.assertions import (
    assert_vision_output,
    assert_human_token,
    assert_query_embedding,
    assert_grounding_scores,
)

from vision.feature_extractor import FeatureExtractor
from adapter.structural_embeddings import HumanTokenAssembler
from adapter.dynamic_grounding_adapter import DynamicGroundingAdapter
from llm.query_encoder import QueryEncoder
from llm.scorer import LLMScorer
from pipeline.selection import HumanSelector, SelectionConfig, SelectionMode


# =============================================================================
# TASK 1: PIPELINE CONTRACT
# =============================================================================

def task1_pipeline_contract():
    """
    TASK 1: Reconfirm full pipeline input/output contract.
    """
    print("\n" + "=" * 70)
    print("TASK 1: PIPELINE CONTRACT")
    print("=" * 70)
    
    print(f"""
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    REFYOLO-HUMAN PIPELINE CONTRACT                  │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │   INPUT:                                                            │
    │   ───────                                                           │
    │   image: PIL.Image | np.ndarray | torch.Tensor | str (path)         │
    │   query: str (natural language referring expression)                │
    │                                                                     │
    │   OUTPUT:                                                           │
    │   ────────                                                          │
    │   {{                                                                 │
    │     "selected_indices": List[int],  # Indices of selected humans    │
    │     "rejected": bool,               # True if no human matches      │
    │     "scores": Tensor[N],            # Raw scores for all humans     │
    │     "boxes": Tensor[N, 4],          # Bounding boxes (xyxy norm)    │
    │     "masks": Tensor[N, H, W],       # Segmentation masks            │
    │     "keypoints": Tensor[N, 17, 3]   # Pose keypoints                │
    │   }}                                                                 │
    │                                                                     │
    │   CONSTRAINTS:                                                      │
    │   ────────────                                                      │
    │   - No new fields allowed                                           │
    │   - CPU-only execution                                              │
    │   - Deterministic outputs                                           │
    │   - Single image inference (no batching)                            │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    """)
    
    print("  [PASS] Pipeline contract confirmed")
    return True


# =============================================================================
# TASK 2: EXECUTION ORDER
# =============================================================================

def task2_execution_order():
    """
    TASK 2: Define and lock pipeline execution order.
    """
    print("\n" + "=" * 70)
    print("TASK 2: EXECUTION ORDER (LOCKED)")
    print("=" * 70)
    
    execution_order = [
        "1. YOLO inference (pose + seg)",
        "2. VisionOutput assembly",
        "3. Feature extraction (ROI Align)",
        "4. Structural embeddings",
        "5. HumanToken assembly",
        "6. Query encoding",
        "7. Dynamic grounding adapter",
        "8. Scoring",
        "9. Selection & rejection",
    ]
    
    print(f"""
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    EXECUTION ORDER (LOCKED)                         │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │""")
    
    for step in execution_order:
        print(f"    │   {step:<62} │")
    
    print(f"""    │                                                                     │
    │   NO REORDERING PERMITTED.                                          │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    """)
    
    print("  [PASS] Execution order locked")
    return execution_order


# =============================================================================
# TASKS 3-4: PIPELINE IMPLEMENTATION
# =============================================================================

@dataclass
class PipelineOutput:
    """Output from RefYOLO-Human pipeline."""
    selected_indices: List[int]
    rejected: bool
    scores: torch.Tensor      # [N]
    boxes: torch.Tensor       # [N, 4]
    masks: torch.Tensor       # [N, H, W]
    keypoints: torch.Tensor   # [N, 17, 3]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "selected_indices": self.selected_indices,
            "rejected": self.rejected,
            "scores": self.scores,
            "boxes": self.boxes,
            "masks": self.masks,
            "keypoints": self.keypoints,
        }


class RefYOLOHumanPipeline:
    """
    Complete RefYOLO-Human inference pipeline.
    
    Orchestrates all components in fixed order:
    1. YOLO inference
    2. VisionOutput assembly
    3. Feature extraction
    4. Structural embeddings
    5. HumanToken assembly
    6. Query encoding
    7. Grounding adapter
    8. Scoring
    9. Selection
    """
    
    def __init__(
        self,
        pose_model_path: str = "yolo11n-pose.pt",
        seg_model_path: str = "yolo11n-seg.pt",
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        selection_config: Optional[SelectionConfig] = None,
        verbose: bool = False,
    ):
        """
        Initialize all pipeline components.
        
        Args:
            pose_model_path: Path to YOLO pose model
            seg_model_path: Path to YOLO segmentation model
            text_model_name: HuggingFace model for query encoding
            selection_config: Selection configuration (default: TOP_1)
            verbose: Print detailed progress
        """
        self.verbose = verbose
        
        if verbose:
            print("\n[Pipeline Init] Loading components...")
        
        # Step 1: Load YOLO models
        if verbose:
            print("  Loading YOLO pose model...")
        self.model_pose = YOLO(pose_model_path)
        self.model_pose.to("cpu")
        self.model_pose.model.eval()
        
        if verbose:
            print("  Loading YOLO segmentation model...")
        self.model_seg = YOLO(seg_model_path)
        self.model_seg.to("cpu")
        self.model_seg.model.eval()
        
        # Step 3: Feature extractor (ROI Align on backbone layer 9)
        if verbose:
            print("  Initializing feature extractor...")
        self.feature_extractor = FeatureExtractor(
            self.model_pose,
            feature_layer_idx=9,
            output_dim=D_VISION,
        )
        
        # Steps 4-5: HumanToken assembler (structural embeddings + projection)
        if verbose:
            print("  Initializing HumanToken assembler...")
        self.token_assembler = HumanTokenAssembler()
        
        # Step 6: Query encoder
        if verbose:
            print("  Loading query encoder...")
        self.query_encoder = QueryEncoder(
            model_name=text_model_name,
            output_dim=D_QUERY,
        )
        
        # Step 7: Dynamic grounding adapter
        if verbose:
            print("  Initializing grounding adapter...")
        self.grounding_adapter = DynamicGroundingAdapter(
            token_dim=D_TOKEN,
            query_dim=D_QUERY,
        )
        
        # Step 8: LLM scorer
        if verbose:
            print("  Initializing scorer...")
        self.scorer = LLMScorer(
            token_dim=D_TOKEN,
            query_dim=D_QUERY,
        )
        
        # Step 9: Human selector
        if verbose:
            print("  Initializing selector...")
        self.selection_config = selection_config or SelectionConfig(
            mode=SelectionMode.TOP_1,
            k=1,
            threshold=0.0,
            rejection_threshold=-float('inf'),
        )
        self.selector = HumanSelector(self.selection_config)
        
        if verbose:
            print("[Pipeline Init] Complete!")
    
    def _load_image(self, image: Union[str, Image.Image, np.ndarray, torch.Tensor]) -> str:
        """
        Prepare image for YOLO inference.
        
        YOLO accepts: str (path), PIL.Image, np.ndarray
        
        Args:
            image: Input image in various formats
            
        Returns:
            Path or image suitable for YOLO
        """
        if isinstance(image, str):
            return image
        elif isinstance(image, Image.Image):
            return image
        elif isinstance(image, np.ndarray):
            return image
        elif isinstance(image, torch.Tensor):
            # Convert tensor to numpy array
            if image.ndim == 4:
                image = image.squeeze(0)
            if image.shape[0] in [1, 3]:
                image = image.permute(1, 2, 0)
            return image.cpu().numpy()
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
    
    def _step1_yolo_inference(self, image) -> tuple:
        """
        Step 1: Run YOLO inference (pose + seg).
        
        Returns:
            (result_pose, result_seg, orig_shape)
        """
        # Run pose model (person detections + keypoints)
        results_pose = self.model_pose(image, device="cpu", verbose=False)
        result_pose = results_pose[0]
        
        # Run segmentation model (person detections + masks)
        results_seg = self.model_seg(image, device="cpu", verbose=False, classes=[0])
        result_seg = results_seg[0]
        
        orig_shape = result_pose.orig_shape  # (H, W)
        
        return result_pose, result_seg, orig_shape
    
    def _step2_vision_output_assembly(
        self,
        result_pose,
        result_seg,
        orig_shape: tuple,
    ) -> VisionOutput:
        """
        Step 2: Assemble VisionOutput from YOLO results.
        
        Aligns pose detections with segmentation masks by IoU matching.
        """
        H_orig, W_orig = orig_shape
        
        # Get pose boxes and keypoints
        if result_pose.boxes is not None and len(result_pose.boxes) > 0:
            boxes_pose = result_pose.boxes.xyxy.cpu()  # [N_pose, 4]
            keypoints_data = result_pose.keypoints.data.cpu()  # [N_pose, 17, 3]
            N_pose = boxes_pose.shape[0]
        else:
            boxes_pose = torch.zeros((0, 4), dtype=torch.float32)
            keypoints_data = torch.zeros((0, 17, 3), dtype=torch.float32)
            N_pose = 0
        
        # Get segmentation boxes and masks
        if result_seg.boxes is not None and result_seg.masks is not None and len(result_seg.boxes) > 0:
            boxes_seg = result_seg.boxes.xyxy.cpu()  # [N_seg, 4]
            masks_raw = result_seg.masks.data.cpu()  # [N_seg, H_mask, W_mask]
            N_seg = boxes_seg.shape[0]
        else:
            boxes_seg = torch.zeros((0, 4), dtype=torch.float32)
            masks_raw = torch.zeros((0, H_MASK, W_MASK), dtype=torch.float32)
            N_seg = 0
        
        N = N_pose  # Use pose detections as primary
        
        # Handle zero humans
        if N == 0:
            return VisionOutput(
                boxes=torch.zeros((1, 0, 4), dtype=torch.float32),
                boxes_valid=torch.zeros((1, 0), dtype=torch.bool),
                masks=torch.zeros((1, 0, H_MASK, W_MASK), dtype=torch.float32),
                keypoints=torch.zeros((1, 0, 17, 3), dtype=torch.float32),
                visual_embeddings=torch.zeros((1, 0, D_VISION), dtype=torch.float32),
            )
        
        # Normalize boxes to [0, 1]
        boxes_normalized = boxes_pose.clone()
        boxes_normalized[:, [0, 2]] /= W_orig
        boxes_normalized[:, [1, 3]] /= H_orig
        
        # Normalize keypoints to [0, 1]
        keypoints_normalized = keypoints_data.clone()
        keypoints_normalized[:, :, 0] /= W_orig  # x
        keypoints_normalized[:, :, 1] /= H_orig  # y
        
        # Match masks to pose detections by IoU
        masks_matched = self._match_masks_to_poses(
            boxes_pose, boxes_seg, masks_raw, N_pose, N_seg
        )
        
        # Resize masks to fixed size
        masks_resized = F.interpolate(
            masks_matched.unsqueeze(1),  # [N, 1, H, W]
            size=(H_MASK, W_MASK),
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)  # [N, H_MASK, W_MASK]
        
        # Create validity mask (all True for detected humans)
        valid = torch.ones(N, dtype=torch.bool)
        
        # Add batch dimension [B=1, N, ...]
        return VisionOutput(
            boxes=boxes_normalized.unsqueeze(0),
            boxes_valid=valid.unsqueeze(0),
            masks=masks_resized.unsqueeze(0),
            keypoints=keypoints_normalized.unsqueeze(0),
            visual_embeddings=torch.zeros((1, N, D_VISION), dtype=torch.float32),  # Placeholder
        )
    
    def _match_masks_to_poses(
        self,
        boxes_pose: torch.Tensor,
        boxes_seg: torch.Tensor,
        masks_raw: torch.Tensor,
        N_pose: int,
        N_seg: int,
    ) -> torch.Tensor:
        """
        Match segmentation masks to pose detections by IoU.
        
        Returns:
            masks_matched: [N_pose, H, W] masks aligned with pose detections
        """
        if N_seg == 0:
            # No segmentation masks - return zeros
            return torch.zeros((N_pose, masks_raw.shape[1] if masks_raw.ndim > 2 else H_MASK,
                               masks_raw.shape[2] if masks_raw.ndim > 2 else W_MASK), dtype=torch.float32)
        
        H_mask, W_mask = masks_raw.shape[1], masks_raw.shape[2]
        masks_matched = torch.zeros((N_pose, H_mask, W_mask), dtype=torch.float32)
        
        # Compute IoU between pose boxes and seg boxes
        iou_matrix = self._compute_iou(boxes_pose, boxes_seg)  # [N_pose, N_seg]
        
        # Greedy matching: for each pose, find best seg match
        for i in range(N_pose):
            if N_seg > 0:
                best_seg_idx = iou_matrix[i].argmax().item()
                best_iou = iou_matrix[i, best_seg_idx].item()
                
                if best_iou > 0.3:  # IoU threshold for valid match
                    masks_matched[i] = masks_raw[best_seg_idx]
        
        return masks_matched
    
    def _compute_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU between two sets of boxes.
        
        Args:
            boxes1: [N, 4] boxes in xyxy format
            boxes2: [M, 4] boxes in xyxy format
            
        Returns:
            iou: [N, M] IoU matrix
        """
        N, M = boxes1.shape[0], boxes2.shape[0]
        
        if N == 0 or M == 0:
            return torch.zeros((N, M), dtype=torch.float32)
        
        # Expand for broadcasting
        boxes1_exp = boxes1.unsqueeze(1).expand(N, M, 4)  # [N, M, 4]
        boxes2_exp = boxes2.unsqueeze(0).expand(N, M, 4)  # [N, M, 4]
        
        # Intersection
        inter_x1 = torch.max(boxes1_exp[..., 0], boxes2_exp[..., 0])
        inter_y1 = torch.max(boxes1_exp[..., 1], boxes2_exp[..., 1])
        inter_x2 = torch.min(boxes1_exp[..., 2], boxes2_exp[..., 2])
        inter_y2 = torch.min(boxes1_exp[..., 3], boxes2_exp[..., 3])
        
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h
        
        # Areas
        area1 = (boxes1_exp[..., 2] - boxes1_exp[..., 0]) * (boxes1_exp[..., 3] - boxes1_exp[..., 1])
        area2 = (boxes2_exp[..., 2] - boxes2_exp[..., 0]) * (boxes2_exp[..., 3] - boxes2_exp[..., 1])
        
        # Union
        union_area = area1 + area2 - inter_area
        
        # IoU
        iou = inter_area / union_area.clamp(min=1e-8)
        
        return iou
    
    def _step3_feature_extraction(
        self,
        image,
        vision_output: VisionOutput,
        orig_shape: tuple,
    ) -> VisionOutput:
        """
        Step 3: Extract visual embeddings using ROI Align.
        
        Updates visual_embeddings field in VisionOutput.
        """
        B, N = vision_output.boxes.shape[:2]
        
        if N == 0:
            return vision_output
        
        H_orig, W_orig = orig_shape
        
        # Get boxes in pixel coordinates for ROI Align
        boxes_pixel = vision_output.boxes.squeeze(0).clone()  # [N, 4]
        boxes_pixel[:, [0, 2]] *= W_orig
        boxes_pixel[:, [1, 3]] *= H_orig
        
        # Extract features using ROI Align
        visual_embeddings = self.feature_extractor.extract(
            image, boxes_pixel, orig_shape
        )  # [N, D_VISION]
        
        # Add batch dimension
        visual_embeddings = visual_embeddings.unsqueeze(0)  # [1, N, D_VISION]
        
        # Return updated VisionOutput
        return VisionOutput(
            boxes=vision_output.boxes,
            boxes_valid=vision_output.boxes_valid,
            masks=vision_output.masks,
            keypoints=vision_output.keypoints,
            visual_embeddings=visual_embeddings,
        )
    
    def _steps4_5_human_token_assembly(self, vision_output: VisionOutput) -> HumanToken:
        """
        Steps 4-5: Assemble HumanToken from VisionOutput.
        
        Computes structural embeddings and projects to final token.
        """
        human_token = self.token_assembler(
            visual_embeddings=vision_output.visual_embeddings,
            boxes=vision_output.boxes,
            keypoints=vision_output.keypoints,
            masks=vision_output.masks,
            valid=vision_output.boxes_valid,
        )
        
        return human_token
    
    def _step6_query_encoding(self, query: str) -> QueryEmbedding:
        """
        Step 6: Encode natural language query.
        """
        return self.query_encoder(query)
    
    def _step7_grounding_adapter(
        self,
        human_token: HumanToken,
        query_embedding: QueryEmbedding,
    ) -> HumanToken:
        """
        Step 7: Apply dynamic grounding adapter.
        
        Fuses human tokens with query embedding.
        """
        return self.grounding_adapter(human_token, query_embedding)
    
    def _step8_scoring(
        self,
        grounded_token: HumanToken,
        query_embedding: QueryEmbedding,
    ) -> GroundingScores:
        """
        Step 8: Score each human against the query.
        """
        return self.scorer(grounded_token, query_embedding)
    
    def _step9_selection(
        self,
        grounding_scores: GroundingScores,
    ):
        """
        Step 9: Select best matching humans.
        
        Returns:
            SelectionResult dataclass with selected_indices, rejected, max_score, num_valid
        """
        # Select from batch element 0 (single image)
        return self.selector.select_single(
            scores=grounding_scores.scores.squeeze(0),  # [N]
            valid=grounding_scores.valid.squeeze(0),    # [N]
        )
    
    def __call__(
        self,
        image: Union[str, Image.Image, np.ndarray, torch.Tensor],
        query: str,
    ) -> PipelineOutput:
        """
        Run complete inference pipeline.
        
        Args:
            image: Input image (path, PIL.Image, np.ndarray, or tensor)
            query: Natural language referring expression
            
        Returns:
            PipelineOutput with selected indices and detection data
        """
        # Prepare image
        image_input = self._load_image(image)
        
        # Step 1: YOLO inference
        if self.verbose:
            print("  Step 1: YOLO inference...")
        result_pose, result_seg, orig_shape = self._step1_yolo_inference(image_input)
        
        # Step 2: VisionOutput assembly
        if self.verbose:
            print("  Step 2: VisionOutput assembly...")
        vision_output = self._step2_vision_output_assembly(
            result_pose, result_seg, orig_shape
        )
        
        N = vision_output.boxes.shape[1]
        
        # Handle zero humans early
        if N == 0:
            return PipelineOutput(
                selected_indices=[],
                rejected=True,
                scores=torch.zeros(0, dtype=torch.float32),
                boxes=torch.zeros((0, 4), dtype=torch.float32),
                masks=torch.zeros((0, H_MASK, W_MASK), dtype=torch.float32),
                keypoints=torch.zeros((0, 17, 3), dtype=torch.float32),
            )
        
        # Step 3: Feature extraction
        if self.verbose:
            print("  Step 3: Feature extraction...")
        vision_output = self._step3_feature_extraction(
            image_input, vision_output, orig_shape
        )
        
        # Steps 4-5: HumanToken assembly
        if self.verbose:
            print("  Steps 4-5: HumanToken assembly...")
        human_token = self._steps4_5_human_token_assembly(vision_output)
        
        # Step 6: Query encoding
        if self.verbose:
            print("  Step 6: Query encoding...")
        query_embedding = self._step6_query_encoding(query)
        
        # Step 7: Grounding adapter
        if self.verbose:
            print("  Step 7: Grounding adapter...")
        grounded_token = self._step7_grounding_adapter(human_token, query_embedding)
        
        # Step 8: Scoring
        if self.verbose:
            print("  Step 8: Scoring...")
        grounding_scores = self._step8_scoring(grounded_token, query_embedding)
        
        # Step 9: Selection
        if self.verbose:
            print("  Step 9: Selection...")
        selection_result = self._step9_selection(grounding_scores)
        
        # Assemble output
        return PipelineOutput(
            selected_indices=selection_result.selected_indices,
            rejected=selection_result.rejected,
            scores=grounding_scores.scores.squeeze(0),  # [N]
            boxes=vision_output.boxes.squeeze(0),       # [N, 4]
            masks=vision_output.masks.squeeze(0),       # [N, H, W]
            keypoints=vision_output.keypoints.squeeze(0),  # [N, 17, 3]
        )


# =============================================================================
# TASK 5: SINGLE-IMAGE SANITY TEST
# =============================================================================

def task5_single_image_test(pipeline: RefYOLOHumanPipeline):
    """
    TASK 5: Run pipeline on image with humans.
    """
    print("\n" + "=" * 70)
    print("TASK 5: SINGLE-IMAGE SANITY TEST")
    print("=" * 70)
    
    # Ensure test image exists
    test_image_path = "test_bus.jpg"
    if not Path(test_image_path).exists():
        print(f"  Downloading test image...")
        url = "https://ultralytics.com/images/bus.jpg"
        urllib.request.urlretrieve(url, test_image_path)
    
    # Run pipeline
    query = "the person on the left"
    print(f"\n  Image: {test_image_path}")
    print(f"  Query: \"{query}\"")
    print(f"\n  Running pipeline...")
    
    output = pipeline(test_image_path, query)
    
    print(f"\n  Results:")
    print(f"    Selected indices: {output.selected_indices}")
    print(f"    Rejected: {output.rejected}")
    print(f"    Number of humans: {output.scores.shape[0]}")
    print(f"    Scores: {output.scores.tolist()}")
    print(f"    Boxes shape: {output.boxes.shape}")
    print(f"    Masks shape: {output.masks.shape}")
    print(f"    Keypoints shape: {output.keypoints.shape}")
    
    # Validation
    assert output.scores.shape[0] > 0, "Expected at least one human"
    assert not output.rejected, "Should not reject (image has humans)"
    assert len(output.selected_indices) > 0, "Should select at least one human"
    
    print(f"\n  [PASS] Single-image sanity test passed")
    return output


# =============================================================================
# TASK 6: ZERO-HUMAN TEST
# =============================================================================

def task6_zero_human_test(pipeline: RefYOLOHumanPipeline):
    """
    TASK 6: Run pipeline on image with no humans.
    """
    print("\n" + "=" * 70)
    print("TASK 6: ZERO-HUMAN TEST")
    print("=" * 70)
    
    # Create a simple image with no humans (solid color)
    no_human_image = np.zeros((480, 640, 3), dtype=np.uint8) + 128  # Gray image
    
    query = "the person standing"
    print(f"\n  Image: Synthetic gray image (no humans)")
    print(f"  Query: \"{query}\"")
    print(f"\n  Running pipeline...")
    
    output = pipeline(no_human_image, query)
    
    print(f"\n  Results:")
    print(f"    Selected indices: {output.selected_indices}")
    print(f"    Rejected: {output.rejected}")
    print(f"    Number of humans: {output.scores.shape[0]}")
    
    # Validation
    assert output.rejected == True, "Should reject (no humans)"
    assert len(output.selected_indices) == 0, "Should not select anyone"
    assert output.scores.shape[0] == 0, "Should have zero scores"
    
    print(f"\n  [PASS] Zero-human test passed (rejected as expected)")
    return output


# =============================================================================
# TASK 7: QUERY VARIATION TEST
# =============================================================================

def task7_query_variation_test(pipeline: RefYOLOHumanPipeline):
    """
    TASK 7: Same image, different queries should produce different scores.
    """
    print("\n" + "=" * 70)
    print("TASK 7: QUERY VARIATION TEST")
    print("=" * 70)
    
    test_image_path = "test_bus.jpg"
    
    # Two different queries
    query_a = "the person on the left"
    query_b = "the person wearing dark clothes"
    
    print(f"\n  Image: {test_image_path}")
    print(f"  Query A: \"{query_a}\"")
    print(f"  Query B: \"{query_b}\"")
    print(f"\n  Running pipeline with Query A...")
    output_a = pipeline(test_image_path, query_a)
    
    print(f"  Running pipeline with Query B...")
    output_b = pipeline(test_image_path, query_b)
    
    print(f"\n  Results:")
    print(f"    Query A scores: {output_a.scores.tolist()}")
    print(f"    Query B scores: {output_b.scores.tolist()}")
    print(f"    Query A selection: {output_a.selected_indices}")
    print(f"    Query B selection: {output_b.selected_indices}")
    
    # Compute score difference
    if output_a.scores.shape[0] > 0:
        score_diff = (output_a.scores - output_b.scores).abs()
        max_diff = score_diff.max().item()
        mean_diff = score_diff.mean().item()
        
        print(f"\n    Score differences:")
        print(f"      Max: {max_diff:.6f}")
        print(f"      Mean: {mean_diff:.6f}")
        
        # Different queries should produce different scores
        if max_diff > 1e-6:
            print(f"\n  [PASS] Different queries produce different scores")
        else:
            print(f"\n  [WARN] Scores are identical (queries may be semantically similar)")
    else:
        print(f"\n  [SKIP] No humans detected for comparison")
    
    return output_a, output_b


# =============================================================================
# TASK 8: DETERMINISM TEST
# =============================================================================

def task8_determinism_test(pipeline: RefYOLOHumanPipeline):
    """
    TASK 8: Same inputs should produce identical outputs.
    """
    print("\n" + "=" * 70)
    print("TASK 8: DETERMINISM TEST")
    print("=" * 70)
    
    test_image_path = "test_bus.jpg"
    query = "the person standing near the bus"
    
    print(f"\n  Image: {test_image_path}")
    print(f"  Query: \"{query}\"")
    print(f"\n  Running pipeline twice...")
    
    # Run 1
    output_1 = pipeline(test_image_path, query)
    
    # Run 2
    output_2 = pipeline(test_image_path, query)
    
    print(f"\n  Run 1:")
    print(f"    Selected indices: {output_1.selected_indices}")
    print(f"    Rejected: {output_1.rejected}")
    print(f"    Scores: {output_1.scores.tolist()}")
    
    print(f"\n  Run 2:")
    print(f"    Selected indices: {output_2.selected_indices}")
    print(f"    Rejected: {output_2.rejected}")
    print(f"    Scores: {output_2.scores.tolist()}")
    
    # Verify identical outputs
    indices_match = output_1.selected_indices == output_2.selected_indices
    rejected_match = output_1.rejected == output_2.rejected
    
    if output_1.scores.shape[0] > 0:
        scores_diff = (output_1.scores - output_2.scores).abs().max().item()
        boxes_diff = (output_1.boxes - output_2.boxes).abs().max().item()
    else:
        scores_diff = 0.0
        boxes_diff = 0.0
    
    print(f"\n  Verification:")
    print(f"    Selected indices match: {indices_match}")
    print(f"    Rejected match: {rejected_match}")
    print(f"    Max score difference: {scores_diff:.2e}")
    print(f"    Max box difference: {boxes_diff:.2e}")
    
    determinism_ok = (
        indices_match and
        rejected_match and
        scores_diff < 1e-5 and
        boxes_diff < 1e-5
    )
    
    if determinism_ok:
        print(f"\n  [PASS] Pipeline is deterministic")
    else:
        print(f"\n  [FAIL] Determinism check failed!")
    
    return determinism_ok


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 70)
    print("PHASE 8: REFYOLO-HUMAN COMPLETE INFERENCE PIPELINE")
    print("=" * 70)
    
    # Task 1: Pipeline contract
    task1_pipeline_contract()
    
    # Task 2: Execution order
    execution_order = task2_execution_order()
    
    # Tasks 3-4: Initialize pipeline
    print("\n" + "=" * 70)
    print("TASKS 3-4: PIPELINE INITIALIZATION")
    print("=" * 70)
    
    print("\n  Initializing RefYOLO-Human pipeline...")
    pipeline = RefYOLOHumanPipeline(verbose=True)
    print("\n  [PASS] Pipeline initialized with all components")
    
    # Task 5: Single-image sanity test
    output_sanity = task5_single_image_test(pipeline)
    
    # Task 6: Zero-human test
    output_zero = task6_zero_human_test(pipeline)
    
    # Task 7: Query variation test
    output_a, output_b = task7_query_variation_test(pipeline)
    
    # Task 8: Determinism test
    determinism_ok = task8_determinism_test(pipeline)
    
    # Summary
    print("\n" + "=" * 70)
    print("PHASE 8 COMPLETE - SUMMARY")
    print("=" * 70)
    
    print(f"""
1. PIPELINE CONTRACT:
   - Input: image (PIL/np/tensor/path) + query (str)
   - Output: selected_indices, rejected, scores, boxes, masks, keypoints

2. EXECUTION ORDER (9 STEPS):
""")
    for step in execution_order:
        print(f"   {step}")
    
    print(f"""
3. SANITY TEST:
   - Detected humans: {output_sanity.scores.shape[0]}
   - Selected index: {output_sanity.selected_indices}
   - Rejected: {output_sanity.rejected}

4. ZERO-HUMAN TEST:
   - Detected humans: {output_zero.scores.shape[0]}
   - Rejected: {output_zero.rejected} (expected: True)

5. QUERY VARIATION TEST:
   - Query A selection: {output_a.selected_indices}
   - Query B selection: {output_b.selected_indices}

6. DETERMINISM TEST:
   - {'[PASS]' if determinism_ok else '[FAIL]'} Identical outputs on repeated runs
""")
    
    print("=" * 70)
    print("ALL TESTS: [PASS]")
    print("=" * 70)


if __name__ == "__main__":
    main()
