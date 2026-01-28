# -*- coding: utf-8 -*-
"""
Visualization Utilities for RefYOLO-Human

Standalone visualization module for grounding results.
Does NOT affect training code.

FEATURES:
- ✅ Draws bounding boxes
- ✅ Draws keypoints (COCO skeleton)
- ✅ Highlights GT vs predicted
- ✅ Shows scores
- ✅ Saves to configurable directory

USAGE:
    # From code:
    from visualization.visualize import GroundingVisualizer
    viz = GroundingVisualizer(output_dir="outputs/visualizations")
    viz.visualize_sample(image, caption, boxes, keypoints, scores, gt_idx, pred_idx)
    
    # From command line:
    python visualization/visualize.py --config config/config.yaml --image path/to/image.jpg --caption "person on the left"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not available. Install with: pip install opencv-python")

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL not available. Install with: pip install Pillow")


# COCO skeleton connections
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
]

# Colors (BGR for OpenCV, RGB for PIL)
COLORS = {
    'gt': (0, 255, 0),  # Green for ground truth
    'pred': (255, 0, 0),  # Red for prediction (when wrong)
    'pred_correct': (0, 255, 0),  # Green for correct prediction
    'other': (128, 128, 128),  # Gray for other humans
    'skeleton': (255, 255, 0),  # Yellow for skeleton
}


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    box_thickness: int = 2
    keypoint_radius: int = 4
    skeleton_thickness: int = 2
    font_scale: float = 0.6
    show_scores: bool = True
    show_keypoints: bool = True
    show_boxes: bool = True
    min_keypoint_conf: float = 0.3


class GroundingVisualizer:
    """
    Visualizer for grounding results.
    
    Can use either OpenCV or PIL backend.
    """
    
    def __init__(
        self,
        output_dir: Path = Path("outputs/visualizations"),
        config: Optional[VisualizationConfig] = None,
    ):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            config: Visualization configuration
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or VisualizationConfig()
        
        # Choose backend
        if HAS_CV2:
            self.backend = "cv2"
        elif HAS_PIL:
            self.backend = "pil"
        else:
            raise RuntimeError("Neither OpenCV nor PIL available!")
    
    def visualize_sample(
        self,
        image: np.ndarray,
        caption: str,
        boxes: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        gt_idx: int,
        pred_idx: Optional[int] = None,
        valid: Optional[np.ndarray] = None,
        save_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Visualize a single grounding sample.
        
        Args:
            image: Image array (H, W, 3) in BGR or RGB
            caption: Caption string
            boxes: Bounding boxes (N, 4) normalized [x1, y1, x2, y2]
            keypoints: Keypoints (N, 17, 3) normalized [x, y, conf]
            scores: Grounding scores (N,)
            gt_idx: Ground truth human index
            pred_idx: Predicted human index (if None, uses argmax of scores)
            valid: Valid mask (N,) for humans
            save_name: Filename to save (optional)
            
        Returns:
            Annotated image
        """
        H, W = image.shape[:2]
        N = boxes.shape[0]
        
        if pred_idx is None:
            pred_idx = int(np.argmax(scores))
        
        if valid is None:
            valid = np.ones(N, dtype=bool)
        
        # Create copy
        vis_img = image.copy()
        
        # Draw each human
        for i in range(N):
            if not valid[i]:
                continue
            
            # Determine color
            if i == gt_idx and i == pred_idx:
                color = COLORS['pred_correct']
                label = "GT+PRED"
            elif i == gt_idx:
                color = COLORS['gt']
                label = "GT"
            elif i == pred_idx:
                color = COLORS['pred']
                label = "PRED"
            else:
                color = COLORS['other']
                label = None
            
            # Draw box
            if self.config.show_boxes:
                box = boxes[i]
                x1 = int(box[0] * W)
                y1 = int(box[1] * H)
                x2 = int(box[2] * W)
                y2 = int(box[3] * H)
                
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, self.config.box_thickness)
                
                # Draw label and score
                if label or self.config.show_scores:
                    text_parts = []
                    if label:
                        text_parts.append(label)
                    if self.config.show_scores:
                        text_parts.append(f"{scores[i]:.2f}")
                    text = " ".join(text_parts)
                    
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                  self.config.font_scale, 1)
                    cv2.rectangle(vis_img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(vis_img, text, (x1 + 2, y1 - 2), 
                               cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, 
                               (0, 0, 0), 1)
            
            # Draw keypoints
            if self.config.show_keypoints:
                kps = keypoints[i]  # (17, 3)
                
                # Draw skeleton
                for j1, j2 in COCO_SKELETON:
                    if kps[j1, 2] > self.config.min_keypoint_conf and \
                       kps[j2, 2] > self.config.min_keypoint_conf:
                        p1 = (int(kps[j1, 0] * W), int(kps[j1, 1] * H))
                        p2 = (int(kps[j2, 0] * W), int(kps[j2, 1] * H))
                        cv2.line(vis_img, p1, p2, COLORS['skeleton'], 
                                self.config.skeleton_thickness)
                
                # Draw keypoints
                for k in range(17):
                    if kps[k, 2] > self.config.min_keypoint_conf:
                        px = int(kps[k, 0] * W)
                        py = int(kps[k, 1] * H)
                        cv2.circle(vis_img, (px, py), self.config.keypoint_radius, 
                                  color, -1)
        
        # Draw caption at top
        caption_text = f'"{caption}"'
        cv2.rectangle(vis_img, (0, 0), (W, 30), (0, 0, 0), -1)
        cv2.putText(vis_img, caption_text, (10, 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save if requested
        if save_name:
            save_path = self.output_dir / save_name
            cv2.imwrite(str(save_path), vis_img)
            print(f"Saved visualization: {save_path}")
        
        return vis_img
    
    def visualize_batch(
        self,
        images: List[np.ndarray],
        captions: List[str],
        boxes_list: List[np.ndarray],
        keypoints_list: List[np.ndarray],
        scores_list: List[np.ndarray],
        gt_indices: List[int],
        pred_indices: Optional[List[int]] = None,
        prefix: str = "sample",
    ) -> List[np.ndarray]:
        """
        Visualize a batch of samples.
        
        Args:
            images: List of images
            captions: List of captions
            boxes_list: List of boxes arrays
            keypoints_list: List of keypoints arrays
            scores_list: List of scores arrays
            gt_indices: List of GT indices
            pred_indices: List of predicted indices
            prefix: Filename prefix
            
        Returns:
            List of annotated images
        """
        results = []
        
        for i in range(len(images)):
            pred_idx = pred_indices[i] if pred_indices else None
            
            vis_img = self.visualize_sample(
                image=images[i],
                caption=captions[i],
                boxes=boxes_list[i],
                keypoints=keypoints_list[i],
                scores=scores_list[i],
                gt_idx=gt_indices[i],
                pred_idx=pred_idx,
                save_name=f"{prefix}_{i:04d}.jpg",
            )
            results.append(vis_img)
        
        return results
    
    def create_comparison_grid(
        self,
        images: List[np.ndarray],
        captions: List[str],
        n_cols: int = 3,
        cell_size: Tuple[int, int] = (400, 400),
        save_name: str = "comparison_grid.jpg",
    ) -> np.ndarray:
        """
        Create a grid of comparison images.
        
        Args:
            images: List of annotated images
            captions: List of captions
            n_cols: Number of columns
            cell_size: Size of each cell (width, height)
            save_name: Output filename
            
        Returns:
            Grid image
        """
        n_images = len(images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        cell_w, cell_h = cell_size
        grid = np.zeros((n_rows * cell_h, n_cols * cell_w, 3), dtype=np.uint8)
        
        for idx, img in enumerate(images):
            row = idx // n_cols
            col = idx % n_cols
            
            # Resize image to fit cell
            resized = cv2.resize(img, cell_size)
            
            y1 = row * cell_h
            x1 = col * cell_w
            grid[y1:y1+cell_h, x1:x1+cell_w] = resized
        
        # Save
        save_path = self.output_dir / save_name
        cv2.imwrite(str(save_path), grid)
        print(f"Saved comparison grid: {save_path}")
        
        return grid


def visualize_from_cache(
    cache_file: Path,
    image_file: Path,
    caption: str,
    output_dir: Path,
    gt_idx: int = 0,
) -> np.ndarray:
    """
    Utility function to visualize a cached sample.
    
    Args:
        cache_file: Path to cached .pt file
        image_file: Path to original image
        caption: Caption string
        output_dir: Output directory
        gt_idx: Ground truth index
        
    Returns:
        Annotated image
    """
    import torch
    
    # Load cache
    cache_data = torch.load(cache_file, weights_only=True)
    boxes = cache_data['boxes'].numpy()
    keypoints = cache_data['keypoints'].numpy()
    valid = cache_data['valid'].numpy()
    
    # Create dummy scores (for visualization without model)
    N = boxes.shape[0]
    scores = np.zeros(N)
    scores[gt_idx] = 1.0
    
    # Load image
    image = cv2.imread(str(image_file))
    if image is None:
        raise ValueError(f"Could not load image: {image_file}")
    
    # Visualize
    viz = GroundingVisualizer(output_dir=output_dir)
    vis_img = viz.visualize_sample(
        image=image,
        caption=caption,
        boxes=boxes,
        keypoints=keypoints,
        scores=scores,
        gt_idx=gt_idx,
        valid=valid,
        save_name=f"{cache_file.stem}_viz.jpg",
    )
    
    return vis_img


def main():
    """Command-line interface for visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize grounding results")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--cache", type=str, required=True, help="Path to cache .pt file")
    parser.add_argument("--caption", type=str, required=True, help="Caption")
    parser.add_argument("--gt-idx", type=int, default=0, help="Ground truth index")
    parser.add_argument("--output-dir", type=str, default="outputs/visualizations", 
                       help="Output directory")
    args = parser.parse_args()
    
    if not HAS_CV2:
        print("Error: OpenCV required for visualization")
        sys.exit(1)
    
    visualize_from_cache(
        cache_file=Path(args.cache),
        image_file=Path(args.image),
        caption=args.caption,
        output_dir=Path(args.output_dir),
        gt_idx=args.gt_idx,
    )


if __name__ == "__main__":
    main()
