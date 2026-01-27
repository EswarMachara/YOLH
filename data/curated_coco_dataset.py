# -*- coding: utf-8 -*-
"""
Curated COCO Dataset for RefYOLO-Human Training

Implements:
- Language filtering (English-only)
- Instance-level filtering (min bbox area ratio)
- Image-level caption deduplication
- Global caption frequency cap
- Deterministic negative sampling

Output format per sample:
{
    "image": PIL.Image,
    "caption": str,
    "boxes": Tensor[N, 4],
    "masks": List[polygon],
    "keypoints": Tensor[N, 17, 3],
    "valid": Tensor[N],
    "gt_index": int
}
"""

import json
import re
import string
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import Dataset
from PIL import Image

# Language detection with deterministic seed
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # Ensure deterministic behavior


# =============================================================================
# CONFIGURATION (LOCKED THRESHOLDS)
# =============================================================================

@dataclass
class CurationConfig:
    """Configuration for dataset curation thresholds."""
    min_bbox_area_ratio: float = 0.01       # 1% of image area
    max_occurrences_per_caption: int = 50   # Global cap
    max_negatives_per_sample: int = 8       # Per-sample negative limit
    
    def __repr__(self):
        return (
            f"CurationConfig(\n"
            f"  min_bbox_area_ratio={self.min_bbox_area_ratio},\n"
            f"  max_occurrences_per_caption={self.max_occurrences_per_caption},\n"
            f"  max_negatives_per_sample={self.max_negatives_per_sample}\n"
            f")"
        )


# =============================================================================
# LANGUAGE DETECTION (ENGLISH-ONLY FILTER)
# =============================================================================

def is_english_caption(caption: str) -> bool:
    """
    Check if caption is in English.
    
    Uses langdetect with seed=0 for deterministic behavior.
    
    Args:
        caption: Raw caption string
        
    Returns:
        True if caption is detected as English, False otherwise.
        On detection failure, returns False (conservative).
    """
    if not caption or not caption.strip():
        return False
    
    try:
        detected_lang = detect(caption)
        return detected_lang == 'en'
    except Exception:
        # On any detection error, treat as non-English (drop)
        return False


# =============================================================================
# CAPTION NORMALIZATION
# =============================================================================

def normalize_caption(caption: str) -> str:
    """
    Normalize caption for deduplication and counting.
    
    Operations:
    - Lowercase
    - Strip punctuation
    - Collapse multiple spaces
    - Strip leading/trailing whitespace
    
    Args:
        caption: Raw caption string
        
    Returns:
        Normalized caption string
    """
    if not caption:
        return ""
    
    # Lowercase
    text = caption.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Strip
    text = text.strip()
    
    return text


# =============================================================================
# TASK 4: INSTANCE-LEVEL FILTERING
# =============================================================================

def compute_bbox_area_ratio(bbox: List[float], image_width: int, image_height: int) -> float:
    """
    Compute bbox area as fraction of image area.
    
    Args:
        bbox: COCO format [x, y, width, height]
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        Ratio of bbox area to image area
    """
    if len(bbox) < 4:
        return 0.0
    
    bbox_area = bbox[2] * bbox[3]  # width * height
    image_area = image_width * image_height
    
    if image_area == 0:
        return 0.0
    
    return bbox_area / image_area


def passes_instance_filter(
    ann: Dict,
    image_width: int,
    image_height: int,
    config: CurationConfig
) -> bool:
    """
    Check if annotation passes instance-level filters.
    
    Args:
        ann: Annotation dictionary
        image_width: Image width
        image_height: Image height
        config: Curation configuration
        
    Returns:
        True if instance passes all filters
    """
    bbox = ann.get('bbox', [0, 0, 0, 0])
    area_ratio = compute_bbox_area_ratio(bbox, image_width, image_height)
    
    return area_ratio >= config.min_bbox_area_ratio


# =============================================================================
# TASK 5: IMAGE-LEVEL CAPTION DEDUPLICATION
# =============================================================================

def compute_instance_priority(ann: Dict) -> Tuple[float, float, int]:
    """
    Compute priority tuple for selecting best instance among duplicates.
    
    Priority order:
    1. Largest bbox area (descending)
    2. Highest mean keypoint confidence (descending)
    3. First occurrence / annotation ID (ascending)
    
    Args:
        ann: Annotation dictionary
        
    Returns:
        Tuple for sorting (higher = better)
    """
    # Bbox area
    bbox = ann.get('bbox', [0, 0, 0, 0])
    bbox_area = bbox[2] * bbox[3] if len(bbox) >= 4 else 0
    
    # Mean keypoint confidence
    keypoints = ann.get('keypoints', [])
    if keypoints and len(keypoints) >= 3:
        # COCO format: [x1, y1, v1, x2, y2, v2, ...]
        visibilities = [keypoints[i] for i in range(2, len(keypoints), 3)]
        mean_conf = sum(visibilities) / len(visibilities) if visibilities else 0
    else:
        mean_conf = 0
    
    # Annotation ID (lower = earlier, so negate for descending sort)
    ann_id = ann.get('id', float('inf'))
    
    return (bbox_area, mean_conf, -ann_id)


def deduplicate_captions_in_image(
    annotations: List[Dict]
) -> Dict[str, Tuple[Dict, List[Dict]]]:
    """
    Deduplicate captions within a single image.
    
    For each normalized caption:
    - Select best instance as positive
    - Track other instances (cannot be positives for this caption)
    
    Args:
        annotations: List of annotations for one image
        
    Returns:
        Dict mapping normalized_caption -> (best_annotation, other_annotations)
    """
    # Group by normalized caption
    caption_groups: Dict[str, List[Dict]] = defaultdict(list)
    
    for ann in annotations:
        caption = ann.get('caption', '')
        norm_cap = normalize_caption(caption)
        if norm_cap:  # Skip empty captions
            caption_groups[norm_cap].append(ann)
    
    # Select best per group
    result = {}
    for norm_cap, group in caption_groups.items():
        if not group:
            continue
        
        # Sort by priority (descending)
        sorted_group = sorted(group, key=compute_instance_priority, reverse=True)
        
        best = sorted_group[0]
        others = sorted_group[1:]
        
        result[norm_cap] = (best, others)
    
    return result


# =============================================================================
# CURATED COCO DATASET CLASS
# =============================================================================

class CuratedCocoDataset(Dataset):
    """
    Curated COCO Dataset for RefYOLO-Human training.
    
    Applies:
    - Instance-level filtering (min bbox area)
    - Image-level caption deduplication
    - Global caption frequency cap
    - Deterministic negative sampling
    
    Args:
        json_path: Path to COCO format JSON file
        image_dir: Directory containing images
        config: CurationConfig with thresholds
        verbose: Print curation statistics
    """
    
    def __init__(
        self,
        json_path: str,
        image_dir: str,
        config: Optional[CurationConfig] = None,
        verbose: bool = True
    ):
        self.json_path = Path(json_path)
        self.image_dir = Path(image_dir)
        self.config = config or CurationConfig()
        self.verbose = verbose
        
        # Statistics tracking
        self.stats = {
            'total_images': 0,
            'total_annotations': 0,
            'dropped_by_language': 0,
            'dropped_by_area': 0,
            'dropped_by_dedup': 0,
            'dropped_by_cap': 0,
            'final_samples': 0,
            'final_instances': 0,
        }
        
        # Load and curate
        self._load_coco_json()
        self._curate_dataset()
        
        if verbose:
            self._print_statistics()
    
    # =========================================================================
    # TASK 2: LOAD COCO JSON
    # =========================================================================
    
    def _load_coco_json(self):
        """Load COCO JSON and build lookup structures."""
        if self.verbose:
            print(f"Loading COCO JSON: {self.json_path}")
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # Build image lookup: image_id -> {id, width, height, file_name}
        self.images = {}
        for img in coco_data.get('images', []):
            self.images[img['id']] = {
                'id': img['id'],
                'width': img['width'],
                'height': img['height'],
                'file_name': img['file_name']
            }
        
        # Build annotation lookup: image_id -> list of annotations
        self.annotations_by_image: Dict[int, List[Dict]] = defaultdict(list)
        for ann in coco_data.get('annotations', []):
            # Keep only necessary fields
            filtered_ann = {
                'id': ann['id'],
                'image_id': ann['image_id'],
                'bbox': ann.get('bbox', [0, 0, 0, 0]),
                'segmentation': ann.get('segmentation', []),
                'keypoints': ann.get('keypoints', []),
                'caption': ann.get('caption', ''),
            }
            self.annotations_by_image[ann['image_id']].append(filtered_ann)
        
        self.stats['total_images'] = len(self.images)
        self.stats['total_annotations'] = sum(len(v) for v in self.annotations_by_image.values())
        
        if self.verbose:
            print(f"  Loaded {self.stats['total_images']:,} images")
            print(f"  Loaded {self.stats['total_annotations']:,} annotations")
    
    # =========================================================================
    # CURATION PIPELINE
    # =========================================================================
    
    def _curate_dataset(self):
        """
        Apply full curation pipeline:
        1. Language filtering (English-only)
        2. Instance-level filtering
        3. Image-level caption deduplication
        4. Global caption frequency cap
        5. Build final sample list
        """
        if self.verbose:
            print("Curating dataset...")
        
        # Global caption counter for frequency cap
        caption_usage_count: Dict[str, int] = defaultdict(int)
        
        # Final samples list
        # Each sample: (image_id, caption, gt_ann, all_valid_anns)
        self.samples: List[Tuple[int, str, Dict, List[Dict]]] = []
        
        for image_id, annotations in self.annotations_by_image.items():
            image_info = self.images.get(image_id)
            if image_info is None:
                continue
            
            img_w = image_info['width']
            img_h = image_info['height']
            
            # -----------------------------------------------------------------
            # STEP 1: Language filtering (BEFORE area, dedup, cap)
            # -----------------------------------------------------------------
            english_annotations = []
            for ann in annotations:
                caption = ann.get('caption', '')
                if is_english_caption(caption):
                    english_annotations.append(ann)
                else:
                    self.stats['dropped_by_language'] += 1
            
            if not english_annotations:
                continue
            
            # -----------------------------------------------------------------
            # STEP 2: Instance-level filtering
            # -----------------------------------------------------------------
            valid_annotations = []
            for ann in english_annotations:
                if passes_instance_filter(ann, img_w, img_h, self.config):
                    valid_annotations.append(ann)
                else:
                    self.stats['dropped_by_area'] += 1
            
            if not valid_annotations:
                continue
            
            # -----------------------------------------------------------------
            # STEP 3: Image-level caption deduplication
            # -----------------------------------------------------------------
            deduped = deduplicate_captions_in_image(valid_annotations)
            
            # Count duplicates dropped
            for norm_cap, (best, others) in deduped.items():
                self.stats['dropped_by_dedup'] += len(others)
            
            # -----------------------------------------------------------------
            # STEP 4: Global caption frequency cap
            # -----------------------------------------------------------------
            for norm_cap, (best_ann, _) in deduped.items():
                # Check cap
                if caption_usage_count[norm_cap] >= self.config.max_occurrences_per_caption:
                    self.stats['dropped_by_cap'] += 1
                    continue
                
                # Increment counter
                caption_usage_count[norm_cap] += 1
                
                # Add sample with original caption (not normalized)
                original_caption = best_ann.get('caption', '')
                self.samples.append((
                    image_id,
                    original_caption,
                    best_ann,
                    valid_annotations  # All valid instances for negative sampling
                ))
        
        self.stats['final_samples'] = len(self.samples)
        
        # Count unique instances in final samples
        seen_ann_ids = set()
        for _, _, gt_ann, valid_anns in self.samples:
            seen_ann_ids.add(gt_ann['id'])
            for ann in valid_anns:
                seen_ann_ids.add(ann['id'])
        self.stats['final_instances'] = len(seen_ann_ids)
        
        # Store caption frequencies for stats
        self.caption_frequencies = caption_usage_count
        
        if self.verbose:
            print(f"  Final samples: {self.stats['final_samples']:,}")
    
    # =========================================================================
    # TASK 7: NEGATIVE SAMPLING
    # =========================================================================
    
    def _select_negatives(
        self,
        gt_ann: Dict,
        all_valid_anns: List[Dict]
    ) -> List[Dict]:
        """
        Select negative instances for a sample.
        
        Rules:
        - Exclude GT instance
        - Sort by bbox area (descending)
        - Limit to max_negatives_per_sample
        
        Args:
            gt_ann: Ground truth annotation
            all_valid_anns: All valid annotations in image
            
        Returns:
            List of negative annotations
        """
        gt_id = gt_ann['id']
        
        # Filter out GT
        candidates = [ann for ann in all_valid_anns if ann['id'] != gt_id]
        
        # Sort by bbox area descending
        def get_bbox_area(ann):
            bbox = ann.get('bbox', [0, 0, 0, 0])
            return bbox[2] * bbox[3] if len(bbox) >= 4 else 0
        
        candidates.sort(key=get_bbox_area, reverse=True)
        
        # Limit
        return candidates[:self.config.max_negatives_per_sample]
    
    # =========================================================================
    # DATASET INTERFACE
    # =========================================================================
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a training sample.
        
        Returns:
            {
                "image": PIL.Image,
                "caption": str,
                "boxes": Tensor[N, 4],
                "masks": List[polygon],
                "keypoints": Tensor[N, 17, 3],
                "valid": Tensor[N],
                "gt_index": int
            }
        """
        image_id, caption, gt_ann, all_valid_anns = self.samples[idx]
        
        # Load image
        image_info = self.images[image_id]
        image_path = self.image_dir / image_info['file_name']
        image = Image.open(image_path).convert('RGB')
        
        # Select negatives
        negatives = self._select_negatives(gt_ann, all_valid_anns)
        
        # Build instance list: GT first, then negatives
        instances = [gt_ann] + negatives
        N = len(instances)
        
        # Extract boxes: [N, 4] in COCO format [x, y, w, h]
        boxes = torch.zeros(N, 4, dtype=torch.float32)
        for i, ann in enumerate(instances):
            bbox = ann.get('bbox', [0, 0, 0, 0])
            boxes[i] = torch.tensor(bbox[:4], dtype=torch.float32)
        
        # Extract masks: List of polygons
        masks = [ann.get('segmentation', []) for ann in instances]
        
        # Extract keypoints: [N, 17, 3]
        keypoints = torch.zeros(N, 17, 3, dtype=torch.float32)
        for i, ann in enumerate(instances):
            kp = ann.get('keypoints', [])
            if kp and len(kp) >= 51:  # 17 * 3
                kp_tensor = torch.tensor(kp[:51], dtype=torch.float32)
                keypoints[i] = kp_tensor.view(17, 3)
        
        # Valid mask: all True (already filtered)
        valid = torch.ones(N, dtype=torch.bool)
        
        # GT index is always 0 (GT is first in list)
        gt_index = 0
        
        return {
            "image": image,
            "caption": caption,
            "boxes": boxes,
            "masks": masks,
            "keypoints": keypoints,
            "valid": valid,
            "gt_index": gt_index
        }
    
    # =========================================================================
    # TASK 8: STATISTICS
    # =========================================================================
    
    def _print_statistics(self):
        """Print curation statistics."""
        print("\n" + "=" * 60)
        print("CURATION STATISTICS")
        print("=" * 60)
        
        print(f"\n  Dataset overview:")
        print(f"    Total images:       {self.stats['total_images']:,}")
        print(f"    Total annotations:  {self.stats['total_annotations']:,}")
        print(f"    Final samples:      {self.stats['final_samples']:,}")
        
        print(f"\n  Filtering breakdown:")
        total_dropped = (
            self.stats['dropped_by_language'] +
            self.stats['dropped_by_area'] +
            self.stats['dropped_by_dedup'] +
            self.stats['dropped_by_cap']
        )
        
        print(f"    Dropped by language:      {self.stats['dropped_by_language']:,} "
              f"({self.stats['dropped_by_language']/self.stats['total_annotations']*100:.1f}%)")
        print(f"    Dropped by area filter:   {self.stats['dropped_by_area']:,} "
              f"({self.stats['dropped_by_area']/self.stats['total_annotations']*100:.1f}%)")
        print(f"    Dropped by deduplication: {self.stats['dropped_by_dedup']:,} "
              f"({self.stats['dropped_by_dedup']/self.stats['total_annotations']*100:.1f}%)")
        print(f"    Dropped by caption cap:   {self.stats['dropped_by_cap']:,} "
              f"({self.stats['dropped_by_cap']/self.stats['total_annotations']*100:.1f}%)")
        print(f"    Total dropped:            {total_dropped:,} "
              f"({total_dropped/self.stats['total_annotations']*100:.1f}%)")
        
        print(f"\n  Top 10 caption frequencies (English only):")
        sorted_caps = sorted(
            self.caption_frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        for cap, count in sorted_caps:
            display_cap = cap[:40] + "..." if len(cap) > 40 else cap
            print(f"    [{count:4d}] \"{display_cap}\"")
        
        print("\n" + "=" * 60)
    
    def get_stats(self) -> Dict:
        """Return curation statistics dictionary."""
        return self.stats.copy()


# =============================================================================
# VALIDATION SCRIPT
# =============================================================================

def validate_dataset(json_path: str, image_dir: str):
    """
    Validate the curated dataset and print statistics.
    
    Args:
        json_path: Path to COCO JSON
        image_dir: Path to images directory
    """
    print("=" * 60)
    print("CURATED COCO DATASET VALIDATION")
    print("=" * 60)
    
    config = CurationConfig()
    print(f"\nConfiguration:\n{config}")
    
    # Create dataset
    dataset = CuratedCocoDataset(
        json_path=json_path,
        image_dir=image_dir,
        config=config,
        verbose=True
    )
    
    # Validate a few samples
    print("\n" + "=" * 60)
    print("SAMPLE VALIDATION")
    print("=" * 60)
    
    if len(dataset) > 0:
        # Check first sample
        sample = dataset[0]
        print(f"\n  Sample 0:")
        print(f"    Image size:    {sample['image'].size}")
        print(f"    Caption:       \"{sample['caption'][:50]}...\"")
        print(f"    Num instances: {sample['boxes'].shape[0]}")
        print(f"    Boxes shape:   {sample['boxes'].shape}")
        print(f"    Keypoints shape: {sample['keypoints'].shape}")
        print(f"    Valid shape:   {sample['valid'].shape}")
        print(f"    GT index:      {sample['gt_index']}")
        
        # Verify GT index
        assert sample['gt_index'] == 0, "GT index should be 0"
        assert sample['valid'].all(), "All instances should be valid"
        
        print(f"\n  [PASS] Sample structure validated")
    else:
        print(f"\n  [WARN] No samples in dataset")
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    
    return dataset


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Default paths
    DEFAULT_JSON = r"C:\Users\Eswar\Desktop\refyolo_human\_annotations_COCO_final.json"
    DEFAULT_IMAGE_DIR = r"C:\Users\Eswar\Desktop\refyolo_human\images"
    
    json_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JSON
    image_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_IMAGE_DIR
    
    # Check if image directory exists
    if not Path(image_dir).exists():
        print(f"[WARN] Image directory not found: {image_dir}")
        print(f"       Dataset will be created but samples cannot be loaded.")
        print(f"       Creating dataset in stats-only mode...\n")
        
        # Create dataset without loading images (stats only)
        config = CurationConfig()
        print(f"Configuration:\n{config}\n")
        
        # Load JSON and curate
        dataset = CuratedCocoDataset.__new__(CuratedCocoDataset)
        dataset.json_path = Path(json_path)
        dataset.image_dir = Path(image_dir)
        dataset.config = config
        dataset.verbose = True
        dataset.stats = {
            'total_images': 0,
            'total_annotations': 0,
            'dropped_by_language': 0,
            'dropped_by_area': 0,
            'dropped_by_dedup': 0,
            'dropped_by_cap': 0,
            'final_samples': 0,
            'final_instances': 0,
        }
        dataset._load_coco_json()
        dataset._curate_dataset()
        dataset._print_statistics()
    else:
        validate_dataset(json_path, image_dir)
