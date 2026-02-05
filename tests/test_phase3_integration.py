# -*- coding: utf-8 -*-
"""
Phase-3 Integration Test

Tests the full TextVisualAlignmentAdapter pipeline:
1. Token-level encoding from SimpleQueryEncoder
2. TextVisualAlignmentAdapter forward pass
3. Scorer with grounded tokens
4. Gradient flow verification
5. Backward compatibility with Phase-0/1
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from training.grounding_train_v2 import SimpleQueryEncoder, TrainableScorer, TrainableAdapter
from adapter.cross_attention_adapter import create_grounding_adapter
from core.datatypes import D_TOKEN, D_QUERY


def test_phase3_integration():
    """Test full Phase-3 pipeline."""
    print("\n" + "="*70)
    print("Phase-3 Integration Test")
    print("="*70)
    
    device = 'cpu'
    
    # Create components
    print("\n--- Creating Components ---")
    query_encoder = SimpleQueryEncoder()
    query_encoder.eval()
    print("✓ SimpleQueryEncoder created")
    
    adapter = create_grounding_adapter(
        adapter_type="text_visual_alignment",
        token_dim=D_TOKEN,
        query_dim=D_QUERY,
        num_heads=4,
        num_layers=1,
        dim_feedforward=512,
        dropout=0.0,  # Deterministic for testing
        bidirectional=True,
    )
    adapter.to(device)
    adapter.train()
    print("✓ TextVisualAlignmentAdapter created")
    
    scorer = TrainableScorer(token_dim=D_TOKEN, query_dim=D_QUERY)
    scorer.to(device)
    scorer.train()
    print("✓ TrainableScorer created")
    
    # Test data
    print("\n--- Preparing Test Data ---")
    B, N = 2, 5  # 2 samples, 5 humans each
    captions = ['person on the left side', 'man wearing black shirt near table']
    
    visual_embeddings = torch.randn(B, N, D_TOKEN, device=device, requires_grad=True)
    print(f"Visual embeddings: {visual_embeddings.shape}")
    
    # Encode captions
    print("\n--- Forward Pass ---")
    with torch.no_grad():
        caption_tokens, caption_mask = query_encoder.forward_tokens_batch(captions)
        query_embeddings = query_encoder.forward_batch(captions)
    
    print(f"Caption tokens: {caption_tokens.shape}")
    print(f"Caption mask: {caption_mask.shape}")
    print(f"Query embeddings: {query_embeddings.shape}")
    
    # Phase-3 forward pass
    grounded_tokens = adapter(visual_embeddings, caption_tokens, caption_mask)
    print(f"Grounded tokens: {grounded_tokens.shape}")
    assert grounded_tokens.shape == (B, N, D_TOKEN), f"Shape mismatch: {grounded_tokens.shape}"
    
    # Scorer
    scores = scorer(grounded_tokens, query_embeddings)
    print(f"Scores: {scores.shape}")
    assert scores.shape == (B, N), f"Score shape mismatch: {scores.shape}"
    
    # Gradient flow
    print("\n--- Gradient Flow Test ---")
    loss = scores.sum()
    loss.backward()
    
    assert visual_embeddings.grad is not None, "No gradient for visual embeddings"
    print(f"✓ Visual embeddings grad norm: {visual_embeddings.grad.norm().item():.4f}")
    
    # Check adapter parameters have gradients
    adapter_params_with_grad = sum(1 for p in adapter.parameters() if p.grad is not None)
    print(f"✓ Adapter params with gradients: {adapter_params_with_grad}")
    
    scorer_params_with_grad = sum(1 for p in scorer.parameters() if p.grad is not None)
    print(f"✓ Scorer params with gradients: {scorer_params_with_grad}")
    
    # Verify query encoder is frozen
    query_encoder_grads = sum(1 for p in query_encoder.parameters() if p.grad is not None)
    assert query_encoder_grads == 0, f"Query encoder should be frozen, got {query_encoder_grads} grads"
    print(f"✓ Query encoder frozen (0 gradients)")
    
    print("\n" + "="*70)
    print("✅ Phase-3 Integration Test PASSED")
    print("="*70)


def test_backward_compatibility():
    """Test Phase-0/1 still work."""
    print("\n" + "="*70)
    print("Backward Compatibility Test")
    print("="*70)
    
    device = 'cpu'
    B, N = 2, 5
    
    # Phase-0: FiLM adapter
    print("\n--- Phase-0 (FiLM) ---")
    adapter_film = create_grounding_adapter(
        adapter_type="film",
        token_dim=D_TOKEN,
        query_dim=D_QUERY,
    )
    adapter_film.to(device)
    
    visual = torch.randn(B, N, D_TOKEN, device=device)
    query = torch.randn(B, D_QUERY, device=device)
    
    out_film = adapter_film(visual, query)
    assert out_film.shape == (B, N, D_TOKEN), f"FiLM output shape: {out_film.shape}"
    print(f"✓ FiLM adapter works: {out_film.shape}")
    
    # Phase-1: Cross-attention
    print("\n--- Phase-1 (Cross-Attention) ---")
    adapter_ca = create_grounding_adapter(
        adapter_type="cross_attention",
        token_dim=D_TOKEN,
        query_dim=D_QUERY,
        num_heads=4,
        num_layers=1,
    )
    adapter_ca.to(device)
    
    out_ca = adapter_ca(visual, query)
    assert out_ca.shape == (B, N, D_TOKEN), f"CrossAttention output shape: {out_ca.shape}"
    print(f"✓ CrossAttention adapter works: {out_ca.shape}")
    
    # Phase-3: Text-visual alignment
    print("\n--- Phase-3 (Text-Visual Alignment) ---")
    adapter_tva = create_grounding_adapter(
        adapter_type="text_visual_alignment",
        token_dim=D_TOKEN,
        query_dim=D_QUERY,
        num_heads=4,
        num_layers=1,
        bidirectional=True,
    )
    adapter_tva.to(device)
    
    T = 10  # caption length
    caption_tokens = torch.randn(B, T, D_TOKEN, device=device)
    caption_mask = torch.ones(B, T, dtype=torch.bool, device=device)
    
    out_tva = adapter_tva(visual, caption_tokens, caption_mask)
    assert out_tva.shape == (B, N, D_TOKEN), f"TVA output shape: {out_tva.shape}"
    print(f"✓ TextVisualAlignment adapter works: {out_tva.shape}")
    
    print("\n" + "="*70)
    print("✅ Backward Compatibility Test PASSED")
    print("="*70)


def test_numerical_stability():
    """Test for NaN/Inf in outputs."""
    print("\n" + "="*70)
    print("Numerical Stability Test")
    print("="*70)
    
    device = 'cpu'
    
    adapter = create_grounding_adapter(
        adapter_type="text_visual_alignment",
        token_dim=D_TOKEN,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )
    adapter.to(device)
    adapter.eval()
    
    # Test various input sizes
    test_cases = [
        (1, 1, 1),   # Single human, single token
        (2, 5, 10),  # Normal case
        (4, 20, 50), # Many humans, long caption
        (8, 3, 5),   # Large batch, few humans
    ]
    
    for B, N, T in test_cases:
        visual = torch.randn(B, N, D_TOKEN, device=device)
        caption = torch.randn(B, T, D_TOKEN, device=device)
        mask = torch.ones(B, T, dtype=torch.bool, device=device)
        
        with torch.no_grad():
            output = adapter(visual, caption, mask)
        
        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()
        
        status = "✓" if not (has_nan or has_inf) else "✗"
        print(f"  {status} B={B}, N={N}, T={T}: NaN={has_nan}, Inf={has_inf}")
        
        assert not has_nan, f"NaN detected for B={B}, N={N}, T={T}"
        assert not has_inf, f"Inf detected for B={B}, N={N}, T={T}"
    
    print("\n" + "="*70)
    print("✅ Numerical Stability Test PASSED")
    print("="*70)


if __name__ == "__main__":
    test_phase3_integration()
    test_backward_compatibility()
    test_numerical_stability()
    
    print("\n" + "="*70)
    print("🎉 ALL PHASE-3 TESTS PASSED")
    print("="*70)
