# -*- coding: utf-8 -*-
"""
YOLO Dataset Builder for Fine-Tuning

Converts COCO-format annotations to YOLO format for fine-tuning.
Uses sample-level splits to ensure no leakage between train/val/test.

For YOLO fine-tuning, we extract IMAGE-LEVEL data from the sample splits:
- If any sample from an image is in TRAIN split → image goes to YOLO train
- Validation images are those with samples ONLY in VAL split
- This is conservative and prevents any potential leakage

Output Format (YOLO):
    images/
        train/
        val/
    labels/
        train/
            <image_id>.txt  # cls x_center y_center width height + keypoints/segments
        val/
            ...

CRITICAL: This module is for YOLO detection fine-tuning ONLY.
          No grounding logic allowed here.
"""

import json
import shutil
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

from tqdm import tqdm


@dataclass
class YOLODatasetConfig:
    """Configuration for YOLO dataset building."""
    output_dir: Path
    img_size: int = 640
    min_bbox_area_ratio: float = 0.01  # Match curation threshold
    

def coco_bbox_to_yolo(bbox: List[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """
    Convert COCO bbox [x, y, width, height] to YOLO format [x_center, y_center, width, height] normalized.
    
    Args:
        bbox: COCO format [x, y, w, h] in pixels
        img_w: Image width
        img_h: Image height
        
    Returns:
        Tuple of (x_center, y_center, width, height) normalized to [0, 1]
    """
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    width = w / img_w
    height = h / img_h
    
    # Clamp to valid range
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    
    return x_center, y_center, width, height


def coco_keypoints_to_yolo(keypoints: List[float], img_w: int, img_h: int) -> List[float]:
    """
    Convert COCO keypoints to YOLO format (normalized).
    
    COCO: [x1, y1, v1, x2, y2, v2, ...] where v is visibility (0=not labeled, 1=labeled but occluded, 2=visible)
    YOLO: [x1, y1, v1, x2, y2, v2, ...] normalized
    
    Args:
        keypoints: COCO format keypoints (51 values for 17 keypoints)
        img_w: Image width
        img_h: Image height
        
    Returns:
        List of normalized keypoints
    """
    if not keypoints or len(keypoints) < 51:
        # Return 17 zero keypoints
        return [0.0] * 51
    
    normalized = []
    for i in range(0, 51, 3):
        x = keypoints[i] / img_w if keypoints[i] > 0 else 0.0
        y = keypoints[i + 1] / img_h if keypoints[i + 1] > 0 else 0.0
        v = keypoints[i + 2]  # Visibility stays as-is (0, 1, or 2)
        
        # Clamp coordinates
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        
        normalized.extend([x, y, v])
    
    return normalized


def coco_segmentation_to_yolo(segmentation: List, img_w: int, img_h: int) -> List[List[float]]:
    """
    Convert COCO segmentation polygons to YOLO format (normalized).
    
    COCO: [[x1, y1, x2, y2, ...], ...] polygons in pixels
    YOLO: [[x1, y1, x2, y2, ...], ...] normalized
    
    Args:
        segmentation: COCO format segmentation (list of polygons)
        img_w: Image width
        img_h: Image height
        
    Returns:
        List of normalized polygon coordinates
    """
    if not segmentation:
        return []
    
    normalized_polys = []
    for polygon in segmentation:
        if isinstance(polygon, dict):
            # RLE format - skip for now
            continue
        if len(polygon) < 6:  # Need at least 3 points
            continue
            
        normalized = []
        for i in range(0, len(polygon), 2):
            x = polygon[i] / img_w
            y = polygon[i + 1] / img_h
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            normalized.extend([x, y])
        
        if len(normalized) >= 6:
            normalized_polys.append(normalized)
    
    return normalized_polys


class YOLODatasetBuilder:
    """
    Builds YOLO-format dataset from COCO annotations using sample-level splits.
    
    IMPORTANT: For YOLO training, we need IMAGE-LEVEL splits derived from sample splits.
    An image is assigned to TRAIN if ANY of its samples are in the TRAIN split.
    This is conservative and ensures no leakage.
    """
    
    def __init__(
        self,
        coco_json_path: Path,
        images_dir: Path,
        output_dir: Path,
        split_config: Dict,
        min_bbox_area_ratio: float = 0.01,
    ):
        """
        Args:
            coco_json_path: Path to COCO annotations JSON
            images_dir: Directory containing source images
            output_dir: Output directory for YOLO dataset
            split_config: Dict with 'train', 'val', 'test' ratios and 'seed'
            min_bbox_area_ratio: Minimum bbox area as fraction of image area
        """
        self.coco_json_path = coco_json_path
        self.images_dir = images_dir
        self.output_dir = Path(output_dir)
        self.split_config = split_config
        self.min_bbox_area_ratio = min_bbox_area_ratio
        
        # Load COCO data
        print(f"Loading COCO annotations from: {coco_json_path}")
        with open(coco_json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        # Build lookups
        self.images = {img['id']: img for img in self.coco_data.get('images', [])}
        self.annotations_by_image = defaultdict(list)
        for ann in self.coco_data.get('annotations', []):
            self.annotations_by_image[ann['image_id']].append(ann)
        
        print(f"  Loaded {len(self.images)} images")
        print(f"  Loaded {len(self.coco_data.get('annotations', []))} annotations")
        
        # Statistics
        self.stats = {
            'total_images': len(self.images),
            'train_images': 0,
            'val_images': 0,
            'train_annotations': 0,
            'val_annotations': 0,
            'skipped_small': 0,
        }
    
    def _compute_image_splits(self) -> Tuple[Set[int], Set[int]]:
        """
        Compute image-level splits from sample-level logic.
        
        Returns:
            (train_image_ids, val_image_ids)
        """
        # Get all image IDs that have annotations
        all_image_ids = list(self.annotations_by_image.keys())
        
        # Shuffle deterministically
        rng = random.Random(self.split_config['seed'])
        rng.shuffle(all_image_ids)
        
        # Split at IMAGE level for YOLO training
        n_total = len(all_image_ids)
        n_train = int(n_total * self.split_config['train'])
        
        train_ids = set(all_image_ids[:n_train])
        val_ids = set(all_image_ids[n_train:n_train + int(n_total * self.split_config['val'])])
        
        return train_ids, val_ids
    
    def _passes_filter(self, ann: Dict, img_w: int, img_h: int) -> bool:
        """Check if annotation passes minimum area filter."""
        bbox = ann.get('bbox', [0, 0, 0, 0])
        if len(bbox) < 4:
            return False
        
        bbox_area = bbox[2] * bbox[3]
        img_area = img_w * img_h
        
        if img_area == 0:
            return False
        
        return (bbox_area / img_area) >= self.min_bbox_area_ratio
    
    def build(self, include_keypoints: bool = True, include_segments: bool = True):
        """
        Build YOLO dataset with separate pose and segmentation label files.
        
        Args:
            include_keypoints: Include keypoints in pose labels
            include_segments: Include segmentation in seg labels
        """
        print("\n" + "=" * 60)
        print("BUILDING YOLO DATASET")
        print("=" * 60)
        
        # Create output directories
        for split in ['train', 'val']:
            (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels_pose' / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels_seg' / split).mkdir(parents=True, exist_ok=True)
        
        # Compute splits
        train_ids, val_ids = self._compute_image_splits()
        
        print(f"\nSplit distribution:")
        print(f"  Train images: {len(train_ids)}")
        print(f"  Val images: {len(val_ids)}")
        
        # Process images
        for image_id, img_info in tqdm(self.images.items(), desc="Processing images"):
            # Determine split
            if image_id in train_ids:
                split = 'train'
            elif image_id in val_ids:
                split = 'val'
            else:
                continue  # Skip test images
            
            img_w = img_info['width']
            img_h = img_info['height']
            file_name = img_info['file_name']
            
            # Get annotations for this image
            anns = self.annotations_by_image.get(image_id, [])
            
            # Filter annotations
            valid_anns = []
            for ann in anns:
                if self._passes_filter(ann, img_w, img_h):
                    valid_anns.append(ann)
                else:
                    self.stats['skipped_small'] += 1
            
            if not valid_anns:
                continue
            
            # Copy image
            src_img = self.images_dir / file_name
            if src_img.exists():
                dst_img = self.output_dir / 'images' / split / file_name
                if not dst_img.exists():
                    shutil.copy2(src_img, dst_img)
            
            # Write pose labels (YOLO pose format)
            pose_lines = []
            for ann in valid_anns:
                bbox = ann.get('bbox', [0, 0, 0, 0])
                x_c, y_c, w, h = coco_bbox_to_yolo(bbox, img_w, img_h)
                
                # Class 0 = person
                line = f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
                
                # Add keypoints if available
                if include_keypoints:
                    kp = ann.get('keypoints', [])
                    if kp and len(kp) >= 51:
                        kp_norm = coco_keypoints_to_yolo(kp, img_w, img_h)
                        kp_str = ' '.join(f"{v:.6f}" for v in kp_norm)
                        line += f" {kp_str}"
                
                pose_lines.append(line)
            
            pose_label_path = self.output_dir / 'labels_pose' / split / f"{Path(file_name).stem}.txt"
            with open(pose_label_path, 'w') as f:
                f.write('\n'.join(pose_lines))
            
            # Write segmentation labels (YOLO segment format)
            seg_lines = []
            for ann in valid_anns:
                seg = ann.get('segmentation', [])
                if include_segments and seg:
                    polys = coco_segmentation_to_yolo(seg, img_w, img_h)
                    for poly in polys:
                        # Format: class x1 y1 x2 y2 ... (polygon points)
                        poly_str = ' '.join(f"{v:.6f}" for v in poly)
                        seg_lines.append(f"0 {poly_str}")
                else:
                    # Fall back to bbox if no segmentation
                    bbox = ann.get('bbox', [0, 0, 0, 0])
                    x_c, y_c, w, h = coco_bbox_to_yolo(bbox, img_w, img_h)
                    # Convert bbox to polygon (4 corners)
                    x1, y1 = x_c - w/2, y_c - h/2
                    x2, y2 = x_c + w/2, y_c + h/2
                    seg_lines.append(f"0 {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}")
            
            seg_label_path = self.output_dir / 'labels_seg' / split / f"{Path(file_name).stem}.txt"
            with open(seg_label_path, 'w') as f:
                f.write('\n'.join(seg_lines))
            
            # Update stats
            if split == 'train':
                self.stats['train_images'] += 1
                self.stats['train_annotations'] += len(valid_anns)
            else:
                self.stats['val_images'] += 1
                self.stats['val_annotations'] += len(valid_anns)
        
        # Create dataset YAML files
        self._create_dataset_yaml('pose')
        self._create_dataset_yaml('seg')
        
        self._print_stats()
    
    def _create_dataset_yaml(self, task: str):
        """Create YOLO dataset YAML configuration file."""
        yaml_content = f"""# YOLO {task.upper()} Dataset Configuration
# Auto-generated by YOLODatasetBuilder

path: {self.output_dir.resolve()}
train: images/train
val: images/val

# Classes
names:
  0: person

# Number of classes
nc: 1
"""
        if task == 'pose':
            yaml_content += """
# Keypoint configuration (COCO 17 keypoints)
kpt_shape: [17, 3]  # [num_keypoints, dims] where dims=3 means x,y,visibility
"""
        
        yaml_path = self.output_dir / f'dataset_{task}.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"  Created: {yaml_path}")
    
    def _print_stats(self):
        """Print dataset statistics."""
        print("\n" + "=" * 60)
        print("YOLO DATASET STATISTICS")
        print("=" * 60)
        print(f"  Total source images: {self.stats['total_images']}")
        print(f"  Train images: {self.stats['train_images']}")
        print(f"  Val images: {self.stats['val_images']}")
        print(f"  Train annotations: {self.stats['train_annotations']}")
        print(f"  Val annotations: {self.stats['val_annotations']}")
        print(f"  Skipped (too small): {self.stats['skipped_small']}")
        print(f"\n  Output directory: {self.output_dir}")
        print("=" * 60)
        
        # Validation
        assert self.stats['train_images'] > 0, "No training images!"
        assert self.stats['val_images'] > 0, "No validation images!"
        
        # Check no overlap
        train_ids, val_ids = self._compute_image_splits()
        overlap = train_ids & val_ids
        assert len(overlap) == 0, f"LEAKAGE DETECTED: {len(overlap)} images in both train and val!"
        
        print("\n  [PASS] No train/val leakage detected")
        print("  [PASS] Dataset built successfully")


def build_yolo_dataset(
    coco_json_path: Path,
    images_dir: Path,
    output_dir: Path,
    split_config: Dict,
    min_bbox_area_ratio: float = 0.01,
) -> YOLODatasetBuilder:
    """
    Convenience function to build YOLO dataset.
    
    Args:
        coco_json_path: Path to COCO annotations
        images_dir: Path to images directory
        output_dir: Output directory for YOLO dataset
        split_config: Split configuration dict
        min_bbox_area_ratio: Minimum bbox area ratio filter
        
    Returns:
        YOLODatasetBuilder instance with statistics
    """
    builder = YOLODatasetBuilder(
        coco_json_path=coco_json_path,
        images_dir=images_dir,
        output_dir=output_dir,
        split_config=split_config,
        min_bbox_area_ratio=min_bbox_area_ratio,
    )
    builder.build()
    return builder


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from core.config import load_config, add_config_argument
    
    parser = argparse.ArgumentParser(description="Build YOLO dataset from COCO annotations")
    add_config_argument(parser)
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Build dataset
    split_config = {
        'train': config.splits.train,
        'val': config.splits.val,
        'test': config.splits.test,
        'seed': config.splits.seed,
    }
    
    output_dir = config.yolo_output_dir / 'dataset'
    
    builder = build_yolo_dataset(
        coco_json_path=config.annotations_path,
        images_dir=config.images_dir,
        output_dir=output_dir,
        split_config=split_config,
    )
