# -*- coding: utf-8 -*-
"""
Memory-safe COCO JSON Analysis
Streaming analysis of large annotation files without loading into RAM.
"""

import ijson
import sys
from collections import defaultdict
from pathlib import Path
import hashlib

# =============================================================================
# CONFIGURATION
# =============================================================================

JSON_PATH = r"C:\Users\Eswar\Desktop\refyolo_human\_annotations_COCO_final.json"

# =============================================================================
# TASK 1: SAFE STRUCTURE INSPECTION
# =============================================================================

def task1_inspect_structure():
    """Identify top-level keys without loading full arrays."""
    print("\n" + "=" * 70)
    print("TASK 1: SAFE STRUCTURE INSPECTION")
    print("=" * 70)
    
    top_level_keys = []
    
    # Use ijson to get top-level keys only
    with open(JSON_PATH, 'rb') as f:
        parser = ijson.parse(f)
        current_depth = 0
        for prefix, event, value in parser:
            if prefix == '' and event == 'map_key':
                top_level_keys.append(value)
            # Stop after finding all top-level keys (when we start going deeper)
            if len(top_level_keys) > 0 and prefix != '' and '.' not in prefix:
                # We've seen enough
                if len(top_level_keys) >= 10:  # Safety limit
                    break
    
    print(f"\n  Top-level keys found: {top_level_keys}")
    
    # Check for standard COCO keys
    standard_keys = {'images', 'annotations', 'categories'}
    found_standard = {k for k in top_level_keys if k in standard_keys}
    non_standard = [k for k in top_level_keys if k not in standard_keys]
    
    print(f"\n  Standard COCO keys present:")
    print(f"    images:      {'YES' if 'images' in found_standard else 'NO'}")
    print(f"    annotations: {'YES' if 'annotations' in found_standard else 'NO'}")
    print(f"    categories:  {'YES' if 'categories' in found_standard else 'NO'}")
    
    if non_standard:
        print(f"\n  Non-standard keys: {non_standard}")
    else:
        print(f"\n  Non-standard keys: None")
    
    return top_level_keys


# =============================================================================
# TASK 2: IMAGE-LEVEL ANALYSIS (STREAMED)
# =============================================================================

def task2_image_statistics():
    """Stream through images and compute statistics."""
    print("\n" + "=" * 70)
    print("TASK 2: IMAGE-LEVEL ANALYSIS (STREAMED)")
    print("=" * 70)
    
    # Running statistics
    count = 0
    width_min, width_max, width_sum = float('inf'), 0, 0
    height_min, height_max, height_sum = float('inf'), 0, 0
    area_min, area_max, area_sum = float('inf'), 0, 0
    
    # Store image_id -> area for later use
    image_areas = {}
    
    print("\n  Streaming through images...")
    
    with open(JSON_PATH, 'rb') as f:
        for image in ijson.items(f, 'images.item'):
            count += 1
            
            w = image.get('width', 0)
            h = image.get('height', 0)
            area = w * h
            img_id = image.get('id')
            
            # Update running stats
            width_min = min(width_min, w)
            width_max = max(width_max, w)
            width_sum += w
            
            height_min = min(height_min, h)
            height_max = max(height_max, h)
            height_sum += h
            
            area_min = min(area_min, area)
            area_max = max(area_max, area)
            area_sum += area
            
            # Store for later annotation analysis
            if img_id is not None:
                image_areas[img_id] = area
            
            if count % 10000 == 0:
                print(f"    Processed {count:,} images...")
    
    print(f"\n  Total images: {count:,}")
    print(f"\n  Width statistics:")
    print(f"    min:  {width_min}")
    print(f"    max:  {width_max}")
    print(f"    mean: {width_sum / count:.1f}" if count > 0 else "    mean: N/A")
    
    print(f"\n  Height statistics:")
    print(f"    min:  {height_min}")
    print(f"    max:  {height_max}")
    print(f"    mean: {height_sum / count:.1f}" if count > 0 else "    mean: N/A")
    
    print(f"\n  Area statistics (pixels):")
    print(f"    min:  {area_min:,}")
    print(f"    max:  {area_max:,}")
    print(f"    mean: {area_sum / count:,.0f}" if count > 0 else "    mean: N/A")
    
    return image_areas, count


# =============================================================================
# TASK 3: INSTANCE-LEVEL ANALYSIS (STREAMED)
# =============================================================================

def task3_instance_statistics(image_areas):
    """Stream through annotations and compute instance statistics."""
    print("\n" + "=" * 70)
    print("TASK 3: INSTANCE-LEVEL ANALYSIS (STREAMED)")
    print("=" * 70)
    
    # Running statistics
    count = 0
    bbox_area_min, bbox_area_max, bbox_area_sum = float('inf'), 0, 0
    ratio_min, ratio_max, ratio_sum = float('inf'), 0, 0
    valid_ratio_count = 0
    
    # Area ratio distribution buckets
    ratio_buckets = {
        'lt_0.1%': 0,
        'lt_0.5%': 0,
        'lt_1%': 0,
        'lt_2%': 0,
        'gte_2%': 0
    }
    
    # Instances per image counter
    instances_per_image = defaultdict(int)
    
    # Category counts
    category_counts = defaultdict(int)
    
    print("\n  Streaming through annotations...")
    
    with open(JSON_PATH, 'rb') as f:
        for ann in ijson.items(f, 'annotations.item'):
            count += 1
            
            img_id = ann.get('image_id')
            cat_id = ann.get('category_id')
            bbox = ann.get('bbox', [0, 0, 0, 0])
            
            # Count instances per image
            instances_per_image[img_id] += 1
            
            # Count by category
            category_counts[cat_id] += 1
            
            # Bbox statistics (COCO format: [x, y, width, height])
            if len(bbox) >= 4:
                bbox_w, bbox_h = bbox[2], bbox[3]
                bbox_area = bbox_w * bbox_h
                
                bbox_area_min = min(bbox_area_min, bbox_area)
                bbox_area_max = max(bbox_area_max, bbox_area)
                bbox_area_sum += bbox_area
                
                # Area ratio
                img_area = image_areas.get(img_id, 0)
                if img_area > 0:
                    ratio = (bbox_area / img_area) * 100  # percentage
                    ratio_min = min(ratio_min, ratio)
                    ratio_max = max(ratio_max, ratio)
                    ratio_sum += ratio
                    valid_ratio_count += 1
                    
                    # Bucket
                    if ratio < 0.1:
                        ratio_buckets['lt_0.1%'] += 1
                    elif ratio < 0.5:
                        ratio_buckets['lt_0.5%'] += 1
                    elif ratio < 1.0:
                        ratio_buckets['lt_1%'] += 1
                    elif ratio < 2.0:
                        ratio_buckets['lt_2%'] += 1
                    else:
                        ratio_buckets['gte_2%'] += 1
            
            if count % 50000 == 0:
                print(f"    Processed {count:,} annotations...")
    
    print(f"\n  Total annotations: {count:,}")
    
    print(f"\n  Bbox area statistics (pixels):")
    print(f"    min:  {bbox_area_min:,.1f}")
    print(f"    max:  {bbox_area_max:,.1f}")
    print(f"    mean: {bbox_area_sum / count:,.1f}" if count > 0 else "    mean: N/A")
    
    print(f"\n  Bbox/Image area ratio statistics:")
    print(f"    min:  {ratio_min:.4f}%")
    print(f"    max:  {ratio_max:.2f}%")
    print(f"    mean: {ratio_sum / valid_ratio_count:.4f}%" if valid_ratio_count > 0 else "    mean: N/A")
    
    print(f"\n  Area ratio distribution:")
    for bucket, cnt in ratio_buckets.items():
        pct = (cnt / valid_ratio_count * 100) if valid_ratio_count > 0 else 0
        print(f"    {bucket}: {cnt:,} ({pct:.2f}%)")
    
    return count, instances_per_image, category_counts


# =============================================================================
# TASK 4: CATEGORY ANALYSIS
# =============================================================================

def task4_category_analysis(category_counts):
    """Analyze categories from the dataset."""
    print("\n" + "=" * 70)
    print("TASK 4: CATEGORY ANALYSIS")
    print("=" * 70)
    
    # Load categories (small array, safe to load)
    categories = {}
    
    with open(JSON_PATH, 'rb') as f:
        for cat in ijson.items(f, 'categories.item'):
            cat_id = cat.get('id')
            cat_name = cat.get('name', 'unknown')
            categories[cat_id] = cat
    
    print(f"\n  Total categories: {len(categories)}")
    
    print(f"\n  Category breakdown:")
    human_related = []
    non_human = []
    
    for cat_id, cat in categories.items():
        name = cat.get('name', 'unknown')
        count = category_counts.get(cat_id, 0)
        
        # Heuristic: check if human-related
        human_keywords = ['person', 'human', 'man', 'woman', 'people', 'pedestrian']
        is_human = any(kw in name.lower() for kw in human_keywords)
        
        if is_human:
            human_related.append((name, cat_id, count))
        else:
            non_human.append((name, cat_id, count))
        
        print(f"    {name} (id={cat_id}): {count:,} instances")
    
    print(f"\n  Human-related categories: {len(human_related)}")
    for name, cid, cnt in human_related:
        print(f"    - {name}: {cnt:,}")
    
    print(f"\n  Non-human categories: {len(non_human)}")
    for name, cid, cnt in non_human:
        print(f"    - {name}: {cnt:,}")
    
    return categories


# =============================================================================
# TASK 5: KEYPOINT & MASK ANALYSIS
# =============================================================================

def task5_keypoint_mask_analysis():
    """Analyze keypoints and segmentation masks."""
    print("\n" + "=" * 70)
    print("TASK 5: KEYPOINT & MASK ANALYSIS")
    print("=" * 70)
    
    # Keypoint statistics
    kp_count = 0
    kp_total_per_instance = []  # Will use running stats
    kp_sum = 0
    kp_visible_sum = 0
    
    # Visible keypoint distribution
    visible_buckets = {
        '0': 0,
        '1-4': 0,
        '5-9': 0,
        '10+': 0
    }
    
    # Segmentation statistics
    seg_count = 0
    seg_polygon = 0
    seg_rle = 0
    seg_empty = 0
    
    ann_count = 0
    
    print("\n  Streaming through annotations for keypoints/masks...")
    
    with open(JSON_PATH, 'rb') as f:
        for ann in ijson.items(f, 'annotations.item'):
            ann_count += 1
            
            # Keypoints analysis
            keypoints = ann.get('keypoints', [])
            if keypoints and len(keypoints) > 0:
                kp_count += 1
                num_kp = len(keypoints) // 3  # COCO format: [x1,y1,v1, x2,y2,v2, ...]
                kp_sum += num_kp
                
                # Count visible keypoints (v=2 visible, v=1 occluded, v=0 not labeled)
                visible = sum(1 for i in range(2, len(keypoints), 3) if keypoints[i] > 0)
                kp_visible_sum += visible
                
                # Bucket visible keypoints
                if visible == 0:
                    visible_buckets['0'] += 1
                elif visible <= 4:
                    visible_buckets['1-4'] += 1
                elif visible <= 9:
                    visible_buckets['5-9'] += 1
                else:
                    visible_buckets['10+'] += 1
            
            # Segmentation analysis
            seg = ann.get('segmentation')
            if seg is not None:
                seg_count += 1
                
                if isinstance(seg, dict):
                    # RLE format
                    seg_rle += 1
                elif isinstance(seg, list):
                    if len(seg) == 0:
                        seg_empty += 1
                    else:
                        # Check if polygon is degenerate
                        is_degenerate = all(len(poly) < 6 for poly in seg if isinstance(poly, list))
                        if is_degenerate:
                            seg_empty += 1
                        else:
                            seg_polygon += 1
            
            if ann_count % 50000 == 0:
                print(f"    Processed {ann_count:,} annotations...")
    
    print(f"\n  Keypoint statistics:")
    print(f"    Annotations with keypoints: {kp_count:,} ({kp_count/ann_count*100:.1f}%)")
    if kp_count > 0:
        print(f"    Mean keypoints per instance: {kp_sum / kp_count:.1f}")
        print(f"    Mean visible keypoints: {kp_visible_sum / kp_count:.1f}")
    
    print(f"\n  Visible keypoint distribution:")
    for bucket, cnt in visible_buckets.items():
        pct = (cnt / kp_count * 100) if kp_count > 0 else 0
        print(f"    {bucket}: {cnt:,} ({pct:.1f}%)")
    
    print(f"\n  Segmentation statistics:")
    print(f"    Annotations with segmentation: {seg_count:,} ({seg_count/ann_count*100:.1f}%)")
    print(f"    Polygon format: {seg_polygon:,}")
    print(f"    RLE format: {seg_rle:,}")
    print(f"    Empty/degenerate: {seg_empty:,}")
    
    return {
        'kp_count': kp_count,
        'visible_buckets': visible_buckets,
        'seg_count': seg_count,
        'seg_empty': seg_empty
    }


# =============================================================================
# TASK 6: CAPTION / REFERRING TEXT ANALYSIS
# =============================================================================

def task6_caption_analysis():
    """Analyze captions or referring text."""
    print("\n" + "=" * 70)
    print("TASK 6: CAPTION / REFERRING TEXT ANALYSIS")
    print("=" * 70)
    
    # Check for caption fields
    caption_fields = ['caption', 'captions', 'sentence', 'sentences', 'text', 'expression', 'ref']
    
    # Statistics
    total_captions = 0
    len_min, len_max, len_sum = float('inf'), 0, 0
    empty_count = 0
    
    # Duplicate detection (approximate via hashing)
    caption_hashes = set()
    duplicate_count = 0
    
    # Sample storage
    shortest_captions = []  # (length, text)
    longest_captions = []   # (length, text)
    
    ann_count = 0
    found_field = None
    
    print("\n  Scanning for caption/referring text fields...")
    
    with open(JSON_PATH, 'rb') as f:
        for ann in ijson.items(f, 'annotations.item'):
            ann_count += 1
            
            # Find caption field
            caption = None
            for field in caption_fields:
                if field in ann:
                    found_field = field
                    val = ann[field]
                    if isinstance(val, str):
                        caption = val
                    elif isinstance(val, list) and len(val) > 0:
                        # Take first if list
                        caption = val[0] if isinstance(val[0], str) else str(val[0])
                    break
            
            if caption is not None:
                total_captions += 1
                length = len(caption)
                
                # Update stats
                if length == 0:
                    empty_count += 1
                else:
                    len_min = min(len_min, length)
                    len_max = max(len_max, length)
                    len_sum += length
                
                # Duplicate check
                h = hashlib.md5(caption.encode()).hexdigest()
                if h in caption_hashes:
                    duplicate_count += 1
                else:
                    caption_hashes.add(h)
                
                # Track shortest/longest
                if length > 0:
                    if len(shortest_captions) < 5:
                        shortest_captions.append((length, caption))
                        shortest_captions.sort(key=lambda x: x[0])
                    elif length < shortest_captions[-1][0]:
                        shortest_captions[-1] = (length, caption)
                        shortest_captions.sort(key=lambda x: x[0])
                    
                    if len(longest_captions) < 5:
                        longest_captions.append((length, caption))
                        longest_captions.sort(key=lambda x: -x[0])
                    elif length > longest_captions[-1][0]:
                        longest_captions[-1] = (length, caption)
                        longest_captions.sort(key=lambda x: -x[0])
            
            if ann_count % 50000 == 0:
                print(f"    Processed {ann_count:,} annotations...")
    
    print(f"\n  Caption field found: {found_field if found_field else 'NONE'}")
    print(f"  Total captions: {total_captions:,}")
    
    if total_captions > 0:
        print(f"\n  Caption length statistics:")
        print(f"    min:  {len_min if len_min != float('inf') else 'N/A'}")
        print(f"    max:  {len_max}")
        print(f"    mean: {len_sum / (total_captions - empty_count):.1f}" if (total_captions - empty_count) > 0 else "    mean: N/A")
        
        print(f"\n  Quality metrics:")
        print(f"    Empty captions: {empty_count:,} ({empty_count/total_captions*100:.1f}%)")
        print(f"    Duplicate captions: {duplicate_count:,} ({duplicate_count/total_captions*100:.1f}%)")
        
        print(f"\n  5 Shortest captions:")
        for length, text in shortest_captions[:5]:
            print(f"    [{length}] \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
        
        print(f"\n  5 Longest captions:")
        for length, text in longest_captions[:5]:
            print(f"    [{length}] \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
    else:
        print(f"\n  No captions found in annotations.")
        print(f"  Checking if captions are in a separate structure...")
    
    return {
        'total': total_captions,
        'empty': empty_count,
        'duplicates': duplicate_count
    }


# =============================================================================
# TASK 7: AGGREGATED DISTRIBUTIONS
# =============================================================================

def task7_aggregated_distributions(instances_per_image, keypoint_stats, category_counts, total_instances):
    """Print aggregated distribution statistics."""
    print("\n" + "=" * 70)
    print("TASK 7: AGGREGATED DISTRIBUTIONS")
    print("=" * 70)
    
    # Instances per image
    ipi_values = list(instances_per_image.values())
    ipi_min = min(ipi_values) if ipi_values else 0
    ipi_max = max(ipi_values) if ipi_values else 0
    ipi_mean = sum(ipi_values) / len(ipi_values) if ipi_values else 0
    
    print(f"\n  Instances per image:")
    print(f"    min:  {ipi_min}")
    print(f"    max:  {ipi_max}")
    print(f"    mean: {ipi_mean:.2f}")
    
    # Distribution of instances per image
    ipi_dist = defaultdict(int)
    for v in ipi_values:
        if v == 1:
            ipi_dist['1'] += 1
        elif v <= 3:
            ipi_dist['2-3'] += 1
        elif v <= 5:
            ipi_dist['4-5'] += 1
        elif v <= 10:
            ipi_dist['6-10'] += 1
        else:
            ipi_dist['11+'] += 1
    
    print(f"\n  Instances per image distribution:")
    for bucket in ['1', '2-3', '4-5', '6-10', '11+']:
        cnt = ipi_dist.get(bucket, 0)
        pct = (cnt / len(ipi_values) * 100) if ipi_values else 0
        print(f"    {bucket}: {cnt:,} images ({pct:.1f}%)")
    
    # Keypoint-poor instances
    if keypoint_stats['kp_count'] > 0:
        kp_poor = keypoint_stats['visible_buckets']['0'] + keypoint_stats['visible_buckets']['1-4']
        kp_poor_pct = (kp_poor / keypoint_stats['kp_count']) * 100
        print(f"\n  Keypoint-poor instances (0-4 visible): {kp_poor:,} ({kp_poor_pct:.1f}%)")
    
    # Non-human instances (if any category is non-human)
    # Already handled in task 4
    
    return {
        'ipi_min': ipi_min,
        'ipi_max': ipi_max,
        'ipi_mean': ipi_mean
    }


# =============================================================================
# TASK 8: SUMMARY & CURATION IMPLICATIONS
# =============================================================================

def task8_summary(image_count, instance_count, caption_stats, keypoint_stats, category_counts):
    """Generate actionable summary for dataset curation."""
    print("\n" + "=" * 70)
    print("TASK 8: SUMMARY & CURATION IMPLICATIONS")
    print("=" * 70)
    
    print(f"""
    +---------------------------------------------------------------------+
    |                    DATASET ANALYSIS SUMMARY                         |
    +---------------------------------------------------------------------+
    
    DATASET OVERVIEW:
    -----------------
    Total images:      {image_count:,}
    Total instances:   {instance_count:,}
    Instances/image:   {instance_count / image_count:.2f} avg
    
    NOISE ASSESSMENT:
    -----------------
    """)
    
    # Calculate noise indicators
    noise_factors = []
    
    # 1. Caption quality
    if caption_stats['total'] > 0:
        empty_pct = caption_stats['empty'] / caption_stats['total'] * 100
        dup_pct = caption_stats['duplicates'] / caption_stats['total'] * 100
        if empty_pct > 5:
            noise_factors.append(f"Empty captions: {empty_pct:.1f}%")
        if dup_pct > 10:
            noise_factors.append(f"Duplicate captions: {dup_pct:.1f}%")
    
    # 2. Keypoint quality
    if keypoint_stats['kp_count'] > 0:
        kp_poor = keypoint_stats['visible_buckets']['0'] + keypoint_stats['visible_buckets']['1-4']
        kp_poor_pct = kp_poor / keypoint_stats['kp_count'] * 100
        if kp_poor_pct > 20:
            noise_factors.append(f"Keypoint-poor instances: {kp_poor_pct:.1f}%")
    
    # 3. Segmentation quality
    if keypoint_stats['seg_count'] > 0:
        seg_empty_pct = keypoint_stats['seg_empty'] / keypoint_stats['seg_count'] * 100
        if seg_empty_pct > 5:
            noise_factors.append(f"Empty segmentations: {seg_empty_pct:.1f}%")
    
    if noise_factors:
        print("    Potential noise sources:")
        for nf in noise_factors:
            print(f"      - {nf}")
    else:
        print("    No major noise indicators detected.")
    
    print(f"""
    FILTERING SIGNAL STRENGTH:
    --------------------------
    """)
    
    signals = []
    
    # Rank signals by usefulness
    if caption_stats['total'] > 0:
        signals.append(("Captions", "HIGH", "Filter empty/duplicate"))
    
    if keypoint_stats['kp_count'] > 0:
        signals.append(("Keypoints", "HIGH", "Filter 0-4 visible"))
    
    signals.append(("Bbox area ratio", "MEDIUM", "Filter < 1% image area"))
    
    if keypoint_stats['seg_count'] > 0:
        signals.append(("Segmentation", "MEDIUM", "Filter empty/degenerate"))
    
    signals.append(("Category", "LOW", "Already human-focused"))
    
    for name, strength, action in signals:
        print(f"    {name}: {strength} - {action}")
    
    print(f"""
    SUGGESTED INITIAL THRESHOLDS:
    -----------------------------
    (For curation design, not final decisions)
    
    - Min bbox area ratio:     1% of image
    - Min visible keypoints:   5
    - Caption requirements:    Non-empty, unique per instance
    - Segmentation:            Non-empty polygon or valid RLE
    
    ESTIMATED USABLE DATA:
    ----------------------
    """)
    
    # Rough estimate
    usable_pct = 100
    
    if caption_stats['total'] > 0:
        usable_pct -= (caption_stats['empty'] / caption_stats['total'] * 100)
        usable_pct -= (caption_stats['duplicates'] / caption_stats['total'] * 100) * 0.5  # Some dups may be OK
    
    if keypoint_stats['kp_count'] > 0:
        kp_poor_pct = (keypoint_stats['visible_buckets']['0'] + keypoint_stats['visible_buckets']['1-4']) / keypoint_stats['kp_count'] * 100
        usable_pct -= kp_poor_pct * 0.3  # Partial penalty
    
    usable_pct = max(0, min(100, usable_pct))
    
    print(f"    Rough estimate of usable instances: {usable_pct:.0f}%")
    print(f"    Estimated usable count: ~{int(instance_count * usable_pct / 100):,}")
    
    print(f"""
    NEXT STEP: CURATION DESIGN
    --------------------------
    Use these findings to define:
    1. Hard filters (area, keypoints, empty captions)
    2. Soft filters (duplicates, low-quality masks)
    3. Sampling strategy (balance by difficulty)
    
    +---------------------------------------------------------------------+
    """)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 70)
    print("MEMORY-SAFE COCO DATASET ANALYSIS")
    print(f"File: {JSON_PATH}")
    print(f"Size: {Path(JSON_PATH).stat().st_size / (1024**3):.2f} GB")
    print("=" * 70)
    
    # Task 1: Structure inspection
    top_keys = task1_inspect_structure()
    
    # Task 2: Image statistics
    image_areas, image_count = task2_image_statistics()
    
    # Task 3: Instance statistics
    instance_count, instances_per_image, category_counts = task3_instance_statistics(image_areas)
    
    # Free image_areas if not needed
    # (keeping for now in case needed)
    
    # Task 4: Category analysis
    categories = task4_category_analysis(category_counts)
    
    # Task 5: Keypoint & mask analysis
    keypoint_stats = task5_keypoint_mask_analysis()
    
    # Task 6: Caption analysis
    caption_stats = task6_caption_analysis()
    
    # Task 7: Aggregated distributions
    agg_stats = task7_aggregated_distributions(instances_per_image, keypoint_stats, category_counts, instance_count)
    
    # Task 8: Summary
    task8_summary(image_count, instance_count, caption_stats, keypoint_stats, category_counts)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
