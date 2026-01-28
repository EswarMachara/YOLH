# -*- coding: utf-8 -*-
"""
Curated COCO Dataset for RefYOLO-Human Training

Implements:
- Language filtering (English-only)
- Instance-level filtering (min bbox area ratio)
- Image-level caption deduplication
- Global caption frequency cap
- Deterministic negative sampling
- SAMPLE-LEVEL train/val/test splits (NOT image-level)

A SAMPLE is defined as: (image_id, caption, gt_human_index, candidate_humans)
Splitting happens at sample level, meaning the same image may appear in 
multiple splits with different captions/instances. This is INTENTIONAL and 
CORRECT for referring expression grounding tasks.

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

import hashlib
import json
import random
import re
import string
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Literal

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


@dataclass
class SplitConfig:
    """Configuration for sample-level train/val/test splits."""
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
    seed: int = 42
    
    def __post_init__(self):
        """Validate that ratios sum to exactly 1.0."""
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.4f} "
                f"(train={self.train}, val={self.val}, test={self.test})"
            )
        if any(r < 0 for r in [self.train, self.val, self.test]):
            raise ValueError("Split ratios cannot be negative")
    
    def __repr__(self):
        return (
            f"SplitConfig(\n"
            f"  train={self.train:.2f},\n"
            f"  val={self.val:.2f},\n"
            f"  test={self.test:.2f},\n"
            f"  seed={self.seed}\n"
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


def compute_caption_hash(caption: str) -> str:
    """
    Compute a deterministic hash for a caption.
    
    Uses normalized caption to ensure consistency.
    
    Args:
        caption: Raw or normalized caption
        
    Returns:
        8-character hex hash
    """
    normalized = normalize_caption(caption)
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()[:8]


def compute_sample_id(image_id: int, annotation_id: int, caption: str) -> str:
    """
    Compute a unique, deterministic sample identity.
    
    Used ONLY for:
    1. Sample-level splitting
    2. Leakage detection
    
    Args:
        image_id: COCO image ID
        annotation_id: COCO annotation ID
        caption: Original caption string
        
    Returns:
        Unique sample ID string: "imgID_annID_capHash"
    """
    cap_hash = compute_caption_hash(caption)
    return f"{image_id}_{annotation_id}_{cap_hash}"


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

# Module-level cache for curated samples (shared across splits)
_CURATED_SAMPLES_CACHE: Dict[str, Tuple[List, Dict, Dict, Dict]] = {}


class CuratedCocoDataset(Dataset):
    """
    Curated COCO Dataset for RefYOLO-Human training.
    
    Applies:
    - Instance-level filtering (min bbox area)
    - Image-level caption deduplication
    - Global caption frequency cap
    - Deterministic negative sampling
    - SAMPLE-LEVEL train/val/test splitting
    
    IMPORTANT: Splitting is at SAMPLE (caption-instance) level, NOT image level.
    The same image may appear in multiple splits with different captions.
    This is intentional and correct for referring expression grounding.
    
    Args:
        json_path: Path to COCO format JSON file
        image_dir: Directory containing images
        split: Which split to load ("train", "val", "test", or None for all)
        config: CurationConfig with thresholds
        split_config: SplitConfig with train/val/test ratios
        verbose: Print curation statistics
    """
    
    def __init__(
        self,
        json_path: str,
        image_dir: str,
        split: Optional[Literal["train", "val", "test"]] = None,
        config: Optional[CurationConfig] = None,
        split_config: Optional[SplitConfig] = None,
        verbose: bool = True
    ):
        self.json_path = Path(json_path)
        self.image_dir = Path(image_dir)
        self.split = split
        self.config = config or CurationConfig()
        self.split_config = split_config or SplitConfig()
        self.verbose = verbose
        
        # Validate split parameter
        if split is not None and split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', 'test', or None, got '{split}'")
        
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
        
        # Use cache key for sharing curated samples across splits
        cache_key = str(self.json_path.resolve())
        
        if cache_key in _CURATED_SAMPLES_CACHE:
            # Reuse cached curated samples
            if self.verbose:
                print(f"[CACHE HIT] Reusing curated samples for split='{split}'")
            self._all_samples, self.images, self.stats, self._split_indices = _CURATED_SAMPLES_CACHE[cache_key]
        else:
            # First load - curate and cache
            if self.verbose:
                print(f"[CACHE MISS] Curating dataset (will be cached for other splits)")
            self._load_coco_json()
            self._curate_dataset()
            self._compute_splits()
            
            # Cache for other split instances
            _CURATED_SAMPLES_CACHE[cache_key] = (
                self._all_samples,
                self.images,
                self.stats,
                self._split_indices
            )
        
        # Filter samples by split
        if self.split is None:
            self.samples = self._all_samples
        else:
            indices = self._split_indices[self.split]
            self.samples = [self._all_samples[i] for i in indices]
        
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
        
        # Final samples list (stored in _all_samples for split filtering)
        # Each sample: (image_id, caption, gt_ann, all_valid_anns, sample_id)
        self._all_samples: List[Tuple[int, str, Dict, List[Dict], str]] = []
        
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
                
                # Compute stable sample ID for splitting
                original_caption = best_ann.get('caption', '')
                sample_id = compute_sample_id(image_id, best_ann['id'], original_caption)
                
                # Add sample with original caption (not normalized) and sample_id
                self._all_samples.append((
                    image_id,
                    original_caption,
                    best_ann,
                    valid_annotations,  # All valid instances for negative sampling
                    sample_id,
                ))
        
        self.stats['final_samples'] = len(self._all_samples)
        
        # Count unique instances in final samples
        seen_ann_ids = set()
        for _, _, gt_ann, valid_anns, _ in self._all_samples:
            seen_ann_ids.add(gt_ann['id'])
            for ann in valid_anns:
                seen_ann_ids.add(ann['id'])
        self.stats['final_instances'] = len(seen_ann_ids)
        
        # Store caption frequencies for stats
        self.caption_frequencies = caption_usage_count
        
        if self.verbose:
            print(f"  Final samples: {self.stats['final_samples']:,}")
    
    # =========================================================================
    # SAMPLE-LEVEL SPLITTING
    # =========================================================================
    
    def _compute_splits(self):
        """
        Compute sample-level train/val/test splits.
        
        IMPORTANT: This splits at SAMPLE level, NOT image level.
        The same image may appear in multiple splits with different captions.
        This is intentional for referring expression grounding.
        
        Process:
        1. Generate deterministic indices based on seed
        2. Shuffle ONCE using the provided seed
        3. Partition into train/val/test using exact ratios
        4. Validate no leakage
        """
        if self.verbose:
            print(f"Computing sample-level splits (seed={self.split_config.seed})...")
        
        n_total = len(self._all_samples)
        if n_total == 0:
            self._split_indices = {"train": [], "val": [], "test": []}
            return
        
        # Generate indices and shuffle deterministically
        indices = list(range(n_total))
        rng = random.Random(self.split_config.seed)
        rng.shuffle(indices)
        
        # Compute split boundaries
        n_train = int(n_total * self.split_config.train)
        n_val = int(n_total * self.split_config.val)
        # Test gets the remainder to ensure no samples are lost
        n_test = n_total - n_train - n_val
        
        # Partition
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        self._split_indices = {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }
        
        # =====================================================================
        # LEAKAGE VALIDATION (REQUIRED - HARD FAIL)
        # =====================================================================
        self._validate_no_leakage()
        
        # Update stats with split info
        self.stats['train_samples'] = len(train_indices)
        self.stats['val_samples'] = len(val_indices)
        self.stats['test_samples'] = len(test_indices)
    
    def _validate_no_leakage(self):
        """
        Validate that no sample appears in multiple splits.
        
        Hard-fails if:
        - Any sample ID appears in more than one split
        - Total split counts don't match total samples
        """
        train_ids = set()
        val_ids = set()
        test_ids = set()
        
        for idx in self._split_indices["train"]:
            sample_id = self._all_samples[idx][4]  # sample_id is 5th element
            train_ids.add(sample_id)
        
        for idx in self._split_indices["val"]:
            sample_id = self._all_samples[idx][4]
            val_ids.add(sample_id)
        
        for idx in self._split_indices["test"]:
            sample_id = self._all_samples[idx][4]
            test_ids.add(sample_id)
        
        # Check for overlap
        train_val_overlap = train_ids & val_ids
        train_test_overlap = train_ids & test_ids
        val_test_overlap = val_ids & test_ids
        
        if train_val_overlap:
            raise RuntimeError(
                f"[LEAKAGE DETECTED] {len(train_val_overlap)} samples in both train and val!"
            )
        if train_test_overlap:
            raise RuntimeError(
                f"[LEAKAGE DETECTED] {len(train_test_overlap)} samples in both train and test!"
            )
        if val_test_overlap:
            raise RuntimeError(
                f"[LEAKAGE DETECTED] {len(val_test_overlap)} samples in both val and test!"
            )
        
        # Check total count
        total_split = len(train_ids) + len(val_ids) + len(test_ids)
        if total_split != len(self._all_samples):
            raise RuntimeError(
                f"[SPLIT ERROR] Total split samples ({total_split}) != "
                f"total curated samples ({len(self._all_samples)})"
            )
        
        if self.verbose:
            print(f"  [PASS] No leakage detected")
            print(f"  [PASS] len(train) + len(val) + len(test) == total_samples")
    
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
            
        NOTE: gt_index is always 0 (GT is first in list).
              No additional fields are returned to preserve output contract.
        """
        # Sample format: (image_id, caption, gt_ann, all_valid_anns, sample_id)
        # sample_id is only used for splitting, not returned
        image_id, caption, gt_ann, all_valid_anns, _sample_id = self.samples[idx]
        
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
        """Print curation and split statistics."""
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
        
        if self.stats['total_annotations'] > 0:
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
        
        # =====================================================================
        # SPLIT STATISTICS (REQUIRED)
        # =====================================================================
        if hasattr(self, '_split_indices') and self._split_indices:
            total = self.stats['final_samples']
            train_n = len(self._split_indices['train'])
            val_n = len(self._split_indices['val'])
            test_n = len(self._split_indices['test'])
            
            print(f"\n  SAMPLE-LEVEL SPLIT STATISTICS:")
            print(f"    Total curated samples: {total:,}")
            print(f"    Train samples: {train_n:,} ({train_n/total*100:.1f}%)")
            print(f"    Val samples:   {val_n:,} ({val_n/total*100:.1f}%)")
            print(f"    Test samples:  {test_n:,} ({test_n/total*100:.1f}%)")
            print(f"\n    [PASS] Sample-level split complete")
            print(f"    [PASS] No leakage detected")
            
            if self.split:
                print(f"\n    Current split: '{self.split}' ({len(self.samples):,} samples)")
        
        # Show top caption frequencies only if available
        if hasattr(self, 'caption_frequencies') and self.caption_frequencies:
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
    
    def get_split_info(self) -> Dict[str, int]:
        """Return split statistics."""
        if not hasattr(self, '_split_indices'):
            return {}
        return {
            'total': self.stats['final_samples'],
            'train': len(self._split_indices['train']),
            'val': len(self._split_indices['val']),
            'test': len(self._split_indices['test']),
            'current_split': self.split,
            'current_split_samples': len(self.samples),
        }


# =============================================================================
# VALIDATION SCRIPT
# =============================================================================

def validate_dataset(json_path: str, image_dir: str):
    """
    Validate the curated dataset with sample-level splits.
    
    Args:
        json_path: Path to COCO JSON
        image_dir: Path to images directory
    """
    print("=" * 60)
    print("CURATED COCO DATASET VALIDATION (WITH SAMPLE-LEVEL SPLITS)")
    print("=" * 60)
    
    config = CurationConfig()
    split_config = SplitConfig()
    print(f"\nCuration Config:\n{config}")
    print(f"\nSplit Config:\n{split_config}")
    
    # Create all three splits to validate
    print("\n" + "-" * 60)
    print("Creating TRAIN split...")
    train_dataset = CuratedCocoDataset(
        json_path=json_path,
        image_dir=image_dir,
        split="train",
        config=config,
        split_config=split_config,
        verbose=True
    )
    
    print("\n" + "-" * 60)
    print("Creating VAL split...")
    val_dataset = CuratedCocoDataset(
        json_path=json_path,
        image_dir=image_dir,
        split="val",
        config=config,
        split_config=split_config,
        verbose=True
    )
    
    print("\n" + "-" * 60)
    print("Creating TEST split...")
    test_dataset = CuratedCocoDataset(
        json_path=json_path,
        image_dir=image_dir,
        split="test",
        config=config,
        split_config=split_config,
        verbose=True
    )
    
    # =========================================================================
    # VALIDATE OUTPUT CONTRACT
    # =========================================================================
    print("\n" + "=" * 60)
    print("OUTPUT CONTRACT VALIDATION")
    print("=" * 60)
    
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        
        # Check required fields
        required_fields = {"image", "caption", "boxes", "masks", "keypoints", "valid", "gt_index"}
        actual_fields = set(sample.keys())
        
        assert actual_fields == required_fields, \
            f"Output contract violated! Expected {required_fields}, got {actual_fields}"
        
        # Check types and shapes
        from PIL import Image as PILImage
        assert isinstance(sample['image'], PILImage.Image), "image must be PIL.Image"
        assert isinstance(sample['caption'], str), "caption must be str"
        assert isinstance(sample['boxes'], torch.Tensor), "boxes must be Tensor"
        assert sample['boxes'].dim() == 2 and sample['boxes'].shape[1] == 4, "boxes must be [N, 4]"
        assert isinstance(sample['masks'], list), "masks must be List"
        assert isinstance(sample['keypoints'], torch.Tensor), "keypoints must be Tensor"
        assert sample['keypoints'].dim() == 3 and sample['keypoints'].shape[1:] == (17, 3), \
            "keypoints must be [N, 17, 3]"
        assert isinstance(sample['valid'], torch.Tensor), "valid must be Tensor"
        assert sample['gt_index'] == 0, "gt_index must be 0"
        
        print(f"  [PASS] Output contract preserved (7 fields, correct types/shapes)")
        print(f"  [PASS] gt_index == 0 (GT is first)")
    
    # =========================================================================
    # FINAL VERIFICATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("FINAL VERIFICATION")
    print("=" * 60)
    
    total = len(train_dataset) + len(val_dataset) + len(test_dataset)
    print(f"\n  Train: {len(train_dataset):,} samples")
    print(f"  Val:   {len(val_dataset):,} samples")
    print(f"  Test:  {len(test_dataset):,} samples")
    print(f"  Total: {total:,} samples")
    
    # Verify sum
    split_info = train_dataset.get_split_info()
    assert total == split_info['total'], f"Split sum mismatch!"
    
    print(f"\n  [PASS] len(train) + len(val) + len(test) == total_samples")
    
    print("\n" + "=" * 60)
    print("The dataset is split at the sample (caption–instance) level,")
    print("not at image level.")
    print("This is intentional and correct for referring expression grounding.")
    print("=" * 60)
    
    return train_dataset, val_dataset, test_dataset


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
        print(f"       Creating dataset in stats-only mode (no sample loading)...\n")
        
        # Create dataset for split validation only
        config = CurationConfig()
        split_config = SplitConfig()
        print(f"Configuration:\n{config}\n")
        print(f"Split Config:\n{split_config}\n")
        
        # Create train split (this will curate and compute splits)
        train_ds = CuratedCocoDataset(
            json_path=json_path,
            image_dir=image_dir,
            split="train",
            config=config,
            split_config=split_config,
            verbose=True
        )
        
        # Create val and test (will use cache)
        val_ds = CuratedCocoDataset(
            json_path=json_path,
            image_dir=image_dir,
            split="val",
            config=config,
            split_config=split_config,
            verbose=True
        )
        
        test_ds = CuratedCocoDataset(
            json_path=json_path,
            image_dir=image_dir,
            split="test",
            config=config,
            split_config=split_config,
            verbose=True
        )
        
        print("\n" + "=" * 60)
        print("SPLIT SUMMARY")
        print("=" * 60)
        print(f"  Train: {len(train_ds):,} samples")
        print(f"  Val:   {len(val_ds):,} samples")
        print(f"  Test:  {len(test_ds):,} samples")
        total = len(train_ds) + len(val_ds) + len(test_ds)
        print(f"  Total: {total:,} samples")
        
        print("\n" + "=" * 60)
        print("The dataset is split at the sample (caption–instance) level,")
        print("not at image level.")
        print("This is intentional and correct for referring expression grounding.")
        print("=" * 60)
    else:
        validate_dataset(json_path, image_dir)
