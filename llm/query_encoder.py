"""
Phase 4: Query Encoding & QueryEmbedding Assembly

This module implements deterministic query encoding using a text encoder
and maps it into the QueryEmbedding contract.

CPU-only. No training. No vision-text interaction.
"""

import os
import sys

# Disable TensorFlow to avoid compatibility issues
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from core.datatypes import QueryEmbedding, D_QUERY
from core.assertions import assert_query_embedding


# =============================================================================
# TASK 1: MODEL AVAILABILITY CHECK
# =============================================================================

def task1_check_model_availability():
    """
    TASK 1: Verify accessibility of candidate text models.
    
    Checks import + load capability for preferred and fallback models.
    """
    print("\n" + "=" * 70)
    print("TASK 1: MODEL AVAILABILITY CHECK")
    print("=" * 70)
    
    candidates = []
    
    # Check transformers availability
    try:
        from transformers import AutoTokenizer, AutoModel, AutoConfig
        print("\n[OK] transformers library available")
    except ImportError as e:
        print(f"\n[FAIL] transformers not available: {e}")
        return []
    
    # Candidate 1: Qwen2.5 (preferred) - but likely too large for CPU
    print("\n--- Checking Qwen2.5 ---")
    try:
        # Qwen2.5-0.5B is the smallest variant
        config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
        candidates.append({
            "name": "Qwen/Qwen2.5-0.5B",
            "hidden_size": config.hidden_size,
            "status": "config_loadable",
            "cpu_compatible": True,  # 0.5B should work on CPU
        })
        print(f"  Name: Qwen/Qwen2.5-0.5B")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  CPU compatible: Yes (0.5B params)")
    except Exception as e:
        print(f"  [SKIP] Qwen2.5: {type(e).__name__}")
    
    # Candidate 2: sentence-transformers/all-MiniLM-L6-v2 (lightweight, proven)
    print("\n--- Checking all-MiniLM-L6-v2 ---")
    try:
        config = AutoConfig.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        candidates.append({
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "hidden_size": config.hidden_size,
            "status": "config_loadable",
            "cpu_compatible": True,
        })
        print(f"  Name: sentence-transformers/all-MiniLM-L6-v2")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  CPU compatible: Yes (22M params)")
    except Exception as e:
        print(f"  [SKIP] all-MiniLM-L6-v2: {type(e).__name__}")
    
    # Candidate 3: BERT-base (fallback, widely available)
    print("\n--- Checking BERT-base ---")
    try:
        config = AutoConfig.from_pretrained("bert-base-uncased")
        candidates.append({
            "name": "bert-base-uncased",
            "hidden_size": config.hidden_size,
            "status": "config_loadable",
            "cpu_compatible": True,
        })
        print(f"  Name: bert-base-uncased")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  CPU compatible: Yes (110M params)")
    except Exception as e:
        print(f"  [SKIP] bert-base: {type(e).__name__}")
    
    # Candidate 4: DistilBERT (smaller BERT)
    print("\n--- Checking DistilBERT ---")
    try:
        config = AutoConfig.from_pretrained("distilbert-base-uncased")
        candidates.append({
            "name": "distilbert-base-uncased",
            "hidden_size": config.hidden_size,
            "status": "config_loadable",
            "cpu_compatible": True,
        })
        print(f"  Name: distilbert-base-uncased")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  CPU compatible: Yes (66M params)")
    except Exception as e:
        print(f"  [SKIP] distilbert: {type(e).__name__}")
    
    print(f"\n  Total candidates found: {len(candidates)}")
    
    return candidates


# =============================================================================
# TASK 2: MODEL SELECTION
# =============================================================================

def task2_select_model(candidates: list) -> dict:
    """
    TASK 2: Select exactly one text encoder model.
    
    Selection criteria:
    - Open weights
    - CPU-loadable
    - Stable tokenizer
    - Hidden size >= 256
    """
    print("\n" + "=" * 70)
    print("TASK 2: MODEL SELECTION (LOCKED)")
    print("=" * 70)
    
    # Filter candidates meeting criteria
    valid = [c for c in candidates if c["hidden_size"] >= 256 and c["cpu_compatible"]]
    
    if not valid:
        raise RuntimeError("No valid text encoder found!")
    
    # Selection priority:
    # 1. all-MiniLM-L6-v2 (best balance of size/quality for embeddings)
    # 2. distilbert (smaller, faster)
    # 3. bert-base (fallback)
    
    priority_order = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "distilbert-base-uncased",
        "bert-base-uncased",
        "Qwen/Qwen2.5-0.5B",
    ]
    
    selected = None
    for model_name in priority_order:
        for c in valid:
            if c["name"] == model_name:
                selected = c
                break
        if selected:
            break
    
    if not selected:
        selected = valid[0]  # Fallback to first valid
    
    print(f"""
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    SELECTED MODEL (LOCKED)                          │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │   Model: {selected['name']:<54} │
    │   Hidden Size: {selected['hidden_size']:<49} │
    │   CPU Compatible: Yes                                               │
    │                                                                     │
    │   Justification:                                                    │
    │   - Lightweight (~22M params for MiniLM, ~66M for DistilBERT)      │
    │   - Designed for sentence embeddings                                │
    │   - Stable tokenizer with fixed vocabulary                          │
    │   - Proven performance on semantic similarity tasks                 │
    │   - Hidden size >= 256 (contract requirement)                       │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    """)
    
    return selected


# =============================================================================
# TASK 3-7: QUERY ENCODER CLASS
# =============================================================================

class QueryEncoder(nn.Module):
    """
    Encodes natural language queries into fixed-size embeddings.
    
    Uses a pre-trained text encoder's embedding layer only (no transformer blocks).
    Applies mean pooling and projects to D_QUERY (256) dimensions.
    
    Output: QueryEmbedding with embedding shape [B, 256]
    """
    
    def __init__(self, model_name: str, output_dim: int = D_QUERY, max_length: int = 64):
        """
        Initialize the query encoder.
        
        Args:
            model_name: HuggingFace model identifier
            output_dim: Output embedding dimension (default: D_QUERY=256)
            max_length: Maximum token sequence length
        """
        super().__init__()
        
        from transformers import AutoTokenizer, AutoModel
        
        self.model_name = model_name
        self.output_dim = output_dim
        self.max_length = max_length
        
        # Load tokenizer
        print(f"\n  Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model (for embedding layer extraction)
        print(f"  Loading model: {model_name}")
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to("cpu")
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Get embedding dimension
        self.hidden_size = self.model.config.hidden_size
        print(f"  Hidden size: {self.hidden_size}")
        
        # Projection layer if needed
        if self.hidden_size != output_dim:
            print(f"  Creating projection: {self.hidden_size} -> {output_dim}")
            torch.manual_seed(46)
            self.projection = nn.Linear(self.hidden_size, output_dim, bias=False)
            nn.init.orthogonal_(self.projection.weight)
            self.projection.eval()
            for param in self.projection.parameters():
                param.requires_grad = False
        else:
            self.projection = None
            print(f"  No projection needed (hidden_size == output_dim)")
    
    def tokenize(self, query: str) -> dict:
        """
        TASK 3: Tokenize the query string.
        
        Args:
            query: Natural language query string
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Tokenize with:
        # - Fixed max length
        # - Truncation enabled
        # - Padding disabled (will handle manually if needed)
        # - Return tensors
        
        tokens = self.tokenizer(
            query,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        
        return tokens
    
    def extract_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        TASK 4: Extract token embeddings from embedding layer only.
        
        No forward pass through transformer blocks.
        No gradients. Deterministic.
        
        Args:
            input_ids: Token IDs [B, seq_len]
            
        Returns:
            Token embeddings [B, seq_len, hidden_size]
        """
        with torch.no_grad():
            # Access embedding layer directly
            # For BERT-like models, this is model.embeddings.word_embeddings
            # For some models, it might be model.embed_tokens
            
            if hasattr(self.model, 'embeddings'):
                # BERT-like architecture
                embeddings = self.model.embeddings.word_embeddings(input_ids)
            elif hasattr(self.model, 'embed_tokens'):
                # Some other architectures
                embeddings = self.model.embed_tokens(input_ids)
            else:
                # Fallback: try to find embedding layer
                raise RuntimeError(f"Cannot find embedding layer in {self.model_name}")
        
        return embeddings
    
    def pool_embeddings(
        self, 
        embeddings: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        TASK 5: Pool token embeddings into a single vector.
        
        Strategy: Mean pooling over tokens (excluding padding).
        
        Justification:
        - Mean pooling captures overall semantic content
        - More robust than [CLS] token alone for short queries
        - Handles variable-length inputs naturally
        - Standard approach for sentence embeddings
        
        Args:
            embeddings: Token embeddings [B, seq_len, hidden_size]
            attention_mask: Mask indicating real tokens [B, seq_len]
            
        Returns:
            Pooled embedding [B, hidden_size]
        """
        # Expand attention mask to match embedding dimensions
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.shape).float()
        
        # Sum embeddings for non-padding tokens
        sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
        
        # Count non-padding tokens
        sum_mask = torch.sum(mask_expanded, dim=1).clamp(min=1e-9)
        
        # Mean pooling
        pooled = sum_embeddings / sum_mask
        
        return pooled
    
    def project(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        TASK 6: Project to target dimensionality if needed.
        
        No nonlinearity. Pure linear projection.
        
        Args:
            pooled: Pooled embedding [B, hidden_size]
            
        Returns:
            Projected embedding [B, output_dim]
        """
        if self.projection is not None:
            with torch.no_grad():
                return self.projection(pooled)
        return pooled
    
    def forward(self, query: str) -> QueryEmbedding:
        """
        TASK 7: Full encoding pipeline.
        
        Args:
            query: Natural language query string
            
        Returns:
            QueryEmbedding instance with L2-normalized embedding
        """
        # Tokenize
        tokens = self.tokenize(query)
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        
        # Extract embeddings
        embeddings = self.extract_embeddings(input_ids)
        
        # Pool
        pooled = self.pool_embeddings(embeddings, attention_mask)
        
        # Project
        projected = self.project(pooled)
        
        # L2 normalize
        norm = projected.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normalized = projected / norm
        
        # Ensure correct dtype
        normalized = normalized.float()
        
        return QueryEmbedding(embedding=normalized)
    
    def encode_batch(self, queries: list) -> QueryEmbedding:
        """
        Encode a batch of queries.
        
        Args:
            queries: List of query strings
            
        Returns:
            QueryEmbedding with shape [B, output_dim]
        """
        # Tokenize batch
        tokens = self.tokenizer(
            queries,
            max_length=self.max_length,
            truncation=True,
            padding=True,  # Pad to longest in batch
            return_tensors="pt",
        )
        
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        
        # Extract, pool, project
        embeddings = self.extract_embeddings(input_ids)
        pooled = self.pool_embeddings(embeddings, attention_mask)
        projected = self.project(pooled)
        
        # L2 normalize
        norm = projected.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normalized = projected / norm
        
        return QueryEmbedding(embedding=normalized.float())


# =============================================================================
# TASK 8: STABILITY & ASSERTIONS
# =============================================================================

def task8_stability_and_assertions(encoder: QueryEncoder, query: str):
    """
    TASK 8: Run stability tests and assertions.
    """
    print("\n" + "=" * 70)
    print("TASK 8: STABILITY & ASSERTION CHECKS")
    print("=" * 70)
    
    # Run twice with same query
    print(f"\n  Test query: '{query}'")
    
    query_emb_1 = encoder(query)
    query_emb_2 = encoder(query)
    
    emb_1 = query_emb_1.embedding
    emb_2 = query_emb_2.embedding
    
    # Max absolute difference
    max_diff = (emb_1 - emb_2).abs().max().item()
    
    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(emb_1, emb_2, dim=-1).item()
    
    print(f"\n  Stability Test (same query, two runs):")
    print(f"    Max absolute difference: {max_diff:.2e}")
    print(f"    Cosine similarity: {cos_sim:.6f}")
    
    if max_diff < 1e-6:
        print(f"    [PASS] Embeddings are numerically identical")
    else:
        print(f"    [WARN] Non-zero difference detected: {max_diff:.2e}")
    
    # Assertion check
    print(f"\n  Running assert_query_embedding...")
    try:
        assert_query_embedding(query_emb_1)
        print(f"    [PASS] assert_query_embedding PASSED")
    except (TypeError, ValueError) as e:
        print(f"    [FAIL] assert_query_embedding FAILED: {e}")
        raise
    
    # Additional checks
    print(f"\n  Additional validation:")
    print(f"    dtype: {emb_1.dtype}")
    print(f"    device: {emb_1.device}")
    print(f"    shape: {emb_1.shape}")
    print(f"    L2 norm: {emb_1.norm(dim=-1).item():.6f}")
    print(f"    Contains NaN: {torch.isnan(emb_1).any().item()}")
    print(f"    Contains Inf: {torch.isinf(emb_1).any().item()}")
    
    return query_emb_1, max_diff, cos_sim


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 70)
    print("PHASE 4: QUERY ENCODING & QueryEmbedding ASSEMBLY")
    print("=" * 70)
    
    # Task 1: Check model availability
    candidates = task1_check_model_availability()
    
    if not candidates:
        raise RuntimeError("No text encoder models available!")
    
    # Task 2: Select model
    selected = task2_select_model(candidates)
    
    # Tasks 3-7: Initialize encoder and demonstrate
    print("\n" + "=" * 70)
    print("TASKS 3-7: QUERY ENCODER IMPLEMENTATION")
    print("=" * 70)
    
    encoder = QueryEncoder(
        model_name=selected["name"],
        output_dim=D_QUERY,
        max_length=64,
    )
    
    # Demo query
    test_query = "the person wearing a red shirt on the left"
    
    print(f"\n--- TASK 3: Tokenization ---")
    tokens = encoder.tokenize(test_query)
    print(f"  Query: '{test_query}'")
    print(f"  Token IDs: {tokens['input_ids'].tolist()}")
    print(f"  Token count: {tokens['input_ids'].shape[1]}")
    
    print(f"\n--- TASK 4: Embedding Extraction ---")
    embeddings = encoder.extract_embeddings(tokens["input_ids"])
    print(f"  Raw embedding shape: {embeddings.shape}")
    print(f"  (seq_len={embeddings.shape[1]}, hidden_size={embeddings.shape[2]})")
    
    print(f"\n--- TASK 5: Pooling ---")
    pooled = encoder.pool_embeddings(embeddings, tokens["attention_mask"])
    print(f"  Pooled shape: {pooled.shape}")
    print(f"  Strategy: Mean pooling over tokens")
    
    print(f"\n--- TASK 6: Dimension Adaptation ---")
    projected = encoder.project(pooled)
    print(f"  Projected shape: {projected.shape}")
    if encoder.projection is not None:
        print(f"  Projection: {encoder.hidden_size} -> {encoder.output_dim}")
    else:
        print(f"  No projection needed")
    
    print(f"\n--- TASK 7: Build QueryEmbedding ---")
    query_embedding = encoder(test_query)
    print(f"  QueryEmbedding.embedding shape: {query_embedding.embedding.shape}")
    print(f"  L2 normalized: {query_embedding.embedding.norm(dim=-1).item():.6f}")
    
    # Task 8: Stability and assertions
    query_emb, max_diff, cos_sim = task8_stability_and_assertions(encoder, test_query)
    
    # Final summary
    print("\n" + "=" * 70)
    print("PHASE 4 COMPLETE - SUMMARY")
    print("=" * 70)
    
    print(f"\n1. SELECTED MODEL:")
    print(f"   Name: {selected['name']}")
    print(f"   Hidden size: {selected['hidden_size']}")
    
    print(f"\n2. TOKENIZATION:")
    print(f"   Token IDs: {tokens['input_ids'].tolist()}")
    print(f"   Token count: {tokens['input_ids'].shape[1]}")
    
    print(f"\n3. RAW EMBEDDING SHAPE: {embeddings.shape}")
    
    print(f"\n4. FINAL EMBEDDING SHAPE: {query_emb.embedding.shape}")
    
    print(f"\n5. STABILITY TEST RESULTS:")
    print(f"   Max absolute difference: {max_diff:.2e}")
    print(f"   Cosine similarity: {cos_sim:.6f}")
    
    print(f"\n6. FULL QueryEmbedding REPR:")
    print(f"   {query_emb}")
    
    print(f"\n7. ASSERTION STATUS: PASSED")
    
    print("\n" + "=" * 70)
    print("Notes:")
    print("  - Using embedding layer only (no transformer forward pass)")
    print("  - Mean pooling for robust sentence representation")
    print("  - L2 normalized output (||emb|| = 1)")
    print("  - Deterministic (no gradients, no randomness)")
    print("  - Ready for adapter integration")
    print("=" * 70)
    
    return encoder, query_emb


if __name__ == "__main__":
    main()
