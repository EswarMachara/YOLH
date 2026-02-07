# -*- coding: utf-8 -*-
"""
Caption Paraphrasing Utilities for RefYOLO-Human

Provides simple rule-based paraphrasing and infrastructure for 
more advanced neural paraphrasing.

USAGE:
    from core.paraphrase import CaptionParaphraser
    
    paraphraser = CaptionParaphraser()
    paraphrased = paraphraser.paraphrase("man in red shirt on the left")
"""

import re
import random
from typing import Dict, List, Optional, Tuple


class CaptionParaphraser:
    """
    Simple rule-based caption paraphraser for data augmentation.
    
    Applies various transformations:
    - Synonym replacement (red → crimson, left → leftmost)
    - Word order variations (the man in red → the red-clad man)
    - Article variations (the → a/the)
    
    For better results, use neural paraphrasing (T5, PEGASUS, etc.)
    and store pre-computed paraphrases.
    """
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        
        # Color synonyms
        self.color_synonyms = {
            "red": ["crimson", "scarlet", "red-colored", "reddish"],
            "blue": ["navy", "azure", "blue-colored", "bluish"],
            "green": ["emerald", "olive", "green-colored", "greenish"],
            "yellow": ["golden", "amber", "yellow-colored", "yellowish"],
            "white": ["ivory", "cream", "white-colored", "pale"],
            "black": ["dark", "ebony", "black-colored", "darkened"],
            "orange": ["tangerine", "orange-colored", "amber"],
            "pink": ["rose", "pink-colored", "rosy"],
            "purple": ["violet", "lavender", "purple-colored"],
            "brown": ["tan", "beige", "brown-colored", "bronze"],
            "gray": ["grey", "silver", "gray-colored", "greyish"],
            "grey": ["gray", "silver", "grey-colored", "grayish"],
        }
        
        # Spatial synonyms
        self.spatial_synonyms = {
            "left": ["leftmost", "on the left side", "to the left"],
            "right": ["rightmost", "on the right side", "to the right"],
            "middle": ["center", "central", "in the center"],
            "center": ["middle", "central", "in the middle"],
            "front": ["foreground", "in front", "closest"],
            "back": ["background", "behind", "furthest"],
            "top": ["upper", "highest", "at the top"],
            "bottom": ["lower", "lowest", "at the bottom"],
            "nearest": ["closest", "front", "foreground"],
            "farthest": ["furthest", "back", "background"],
        }
        
        # Clothing synonyms
        self.clothing_synonyms = {
            "shirt": ["top", "t-shirt", "blouse"],
            "t-shirt": ["shirt", "tee", "top"],
            "pants": ["trousers", "jeans", "bottoms"],
            "jeans": ["pants", "denim pants", "trousers"],
            "dress": ["gown", "frock", "outfit"],
            "jacket": ["coat", "blazer", "outerwear"],
            "hat": ["cap", "headwear", "head covering"],
            "shorts": ["short pants", "bermudas"],
        }
        
        # Person synonyms
        self.person_synonyms = {
            "man": ["male", "guy", "gentleman", "person"],
            "woman": ["female", "lady", "girl", "person"],
            "person": ["individual", "human", "figure"],
            "people": ["individuals", "persons", "humans"],
            "child": ["kid", "youngster", "young person"],
            "boy": ["male child", "young man", "kid"],
            "girl": ["female child", "young woman", "kid"],
        }
        
        # Action synonyms
        self.action_synonyms = {
            "standing": ["upright", "on their feet", "vertical"],
            "sitting": ["seated", "sat down", "in a seated position"],
            "walking": ["moving", "strolling", "going"],
            "running": ["jogging", "sprinting", "moving quickly"],
            "holding": ["carrying", "grasping", "with"],
            "wearing": ["dressed in", "in", "with"],
        }
        
        # Phrase templates for restructuring
        self.restructure_patterns = [
            # "X in Y" → "Y-clad X" (for clothing)
            (r"(\w+)\s+in\s+(red|blue|green|yellow|white|black)\s+(\w+)",
             lambda m: f"{m.group(2)}-clad {m.group(1)} in {m.group(3)}"),
            
            # "the X" → "a X" (article variation)
            (r"^the\s+", lambda m: self.rng.choice(["the ", "a ", ""])),
            
            # "wearing X" → "dressed in X"
            (r"wearing\s+", lambda m: self.rng.choice(["wearing ", "dressed in ", "in "])),
        ]
    
    def _replace_word(self, text: str, word: str, synonyms: List[str]) -> str:
        """Replace a word with a random synonym."""
        if not synonyms:
            return text
        
        pattern = rf'\b{re.escape(word)}\b'
        synonym = self.rng.choice(synonyms)
        return re.sub(pattern, synonym, text, flags=re.IGNORECASE)
    
    def _apply_synonym_replacement(self, text: str) -> str:
        """Randomly replace words with synonyms."""
        text_lower = text.lower()
        
        # Color replacement (50% chance)
        for color, synonyms in self.color_synonyms.items():
            if color in text_lower and self.rng.random() < 0.5:
                text = self._replace_word(text, color, synonyms)
        
        # Spatial replacement (50% chance)
        for spatial, synonyms in self.spatial_synonyms.items():
            if spatial in text_lower and self.rng.random() < 0.5:
                text = self._replace_word(text, spatial, synonyms)
        
        # Clothing replacement (40% chance)
        for clothing, synonyms in self.clothing_synonyms.items():
            if clothing in text_lower and self.rng.random() < 0.4:
                text = self._replace_word(text, clothing, synonyms)
        
        # Person replacement (40% chance)
        for person, synonyms in self.person_synonyms.items():
            if person in text_lower and self.rng.random() < 0.4:
                text = self._replace_word(text, person, synonyms)
        
        return text
    
    def paraphrase(self, caption: str, variation_strength: float = 0.5) -> str:
        """
        Generate a paraphrase of the caption.
        
        Args:
            caption: Original caption string
            variation_strength: How much to vary (0.0 = no change, 1.0 = maximum)
        
        Returns:
            Paraphrased caption
        """
        if not caption or self.rng.random() > variation_strength:
            return caption
        
        text = caption
        
        # Apply synonym replacement
        text = self._apply_synonym_replacement(text)
        
        return text
    
    def paraphrase_batch(
        self, 
        captions: List[str], 
        paraphrase_prob: float = 0.3
    ) -> Tuple[List[str], List[bool]]:
        """
        Paraphrase a batch of captions with given probability.
        
        Args:
            captions: List of caption strings
            paraphrase_prob: Probability of paraphrasing each caption
        
        Returns:
            Tuple of (paraphrased_captions, was_paraphrased_mask)
        """
        results = []
        masks = []
        
        for caption in captions:
            if self.rng.random() < paraphrase_prob:
                results.append(self.paraphrase(caption, variation_strength=0.8))
                masks.append(True)
            else:
                results.append(caption)
                masks.append(False)
        
        return results, masks


def generate_paraphrases_for_dataset(
    coco_json_path: str,
    output_path: str,
    num_paraphrases: int = 3,
    use_neural: bool = False,
) -> None:
    """
    Generate paraphrases for all captions in a COCO dataset.
    
    Pre-generates paraphrases for faster training.
    
    Args:
        coco_json_path: Path to COCO annotations JSON
        output_path: Path to save paraphrase mappings
        num_paraphrases: Number of paraphrases per caption
        use_neural: Whether to use neural paraphrasing (requires T5)
    """
    import json
    from pathlib import Path
    from tqdm import tqdm
    
    print(f"Loading annotations from: {coco_json_path}")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    paraphraser = CaptionParaphraser()
    paraphrase_map: Dict[str, List[str]] = {}
    
    # Collect unique captions
    unique_captions = set()
    for ann in coco_data['annotations']:
        if 'caption' in ann and ann['caption']:
            unique_captions.add(ann['caption'])
    
    print(f"Found {len(unique_captions)} unique captions")
    
    # Generate paraphrases
    for caption in tqdm(unique_captions, desc="Generating paraphrases"):
        paraphrases = []
        for i in range(num_paraphrases):
            paraphraser.rng.seed(hash(caption) + i)  # Deterministic per caption
            paraphrase = paraphraser.paraphrase(caption, variation_strength=0.9)
            if paraphrase != caption:
                paraphrases.append(paraphrase)
        
        # Deduplicate and ensure we have variations
        paraphrases = list(set(paraphrases))
        if paraphrases:
            paraphrase_map[caption] = paraphrases
    
    # Save mappings
    print(f"Saving {len(paraphrase_map)} caption paraphrase mappings to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(paraphrase_map, f, indent=2)
    
    print(f"Done! Average paraphrases per caption: {sum(len(v) for v in paraphrase_map.values()) / max(len(paraphrase_map), 1):.1f}")


if __name__ == "__main__":
    # Demo usage
    paraphraser = CaptionParaphraser(seed=123)
    
    test_captions = [
        "man in red shirt on the left",
        "woman wearing blue dress in the middle",
        "person standing in the background",
        "the child in yellow jacket nearest to camera",
        "man holding umbrella on the right side",
    ]
    
    print("Caption Paraphrasing Demo:")
    print("=" * 60)
    for caption in test_captions:
        print(f"\nOriginal: {caption}")
        for i in range(3):
            # Use different seeds for each variation
            paraphraser.rng.seed(hash(caption) + i * 100)
            paraphrased = paraphraser.paraphrase(caption, variation_strength=1.0)
            if paraphrased != caption:
                print(f"  → {paraphrased}")
            else:
                print(f"  → (unchanged)")
