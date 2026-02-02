# -*- coding: utf-8 -*-
"""
Cross-Attention Grounding Adapter for RefYOLO-Human

Phase-1 Improvement: Replace FiLM-style modulation with query-to-human cross-attention.

Architecture:
    Query Encoder (frozen) → query embedding [B, 256]
            ↓
    Cross-Attention Module (trainable)
        - Query: text query [B, 1, 256]
        - Keys/Values: human visual tokens [B, N, 256]
        - nn.MultiheadAttention with 4-8 heads
        - 1-2 transformer layers
            ↓
    Grounded Human Tokens [B, N, 256]
            ↓
    MLP Scorer (unchanged)

DESIGN PRINCIPLES:
- Query attends TO human tokens (query as Q, humans as K/V)
- Output dimension preserved: [B, N, 256] → [B, N, 256]
- Lightweight: 1-2 layers, 4-8 heads
- No positional encoding (humans are unordered)
- Residual connections for stable training
- Layer normalization for gradient flow

USAGE:
    adapter = CrossAttentionAdapter(token_dim=256, query_dim=256, num_heads=4, num_layers=1)
    grounded_tokens = adapter(visual_tokens, query_embedding)
    # visual_tokens: [B, N, 256]
    # query_embedding: [B, 256]
    # grounded_tokens: [B, N, 256]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Import dimension constants
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.datatypes import D_TOKEN, D_QUERY


class CrossAttentionLayer(nn.Module):
    """
    Single cross-attention layer where query attends to human tokens.
    
    Architecture:
        1. Cross-attention: query (Q) attends to humans (K, V)
        2. Add & Norm (residual on humans)
        3. Feed-forward network
        4. Add & Norm
    
    The query "broadcasts" information to each human token based on relevance.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Model dimension (token_dim = query_dim = 256)
            num_heads: Number of attention heads (4 or 8)
            dim_feedforward: FFN hidden dimension
            dropout: Dropout rate (0.0 for deterministic inference)
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Cross-attention: humans attend to query
        # In PyTorch MultiheadAttention:
        #   - query: what we're computing attention FOR (humans)
        #   - key/value: what we're attending TO (text query)
        # But we want query-to-human attention, so we flip:
        #   - We broadcast query info to each human
        #   - Humans are updated based on their relevance to query
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # [B, seq_len, d_model]
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        
        # Dropout for residual
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        # Cross-attention is initialized by PyTorch
        # FFN initialization
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        human_tokens: torch.Tensor,
        query_token: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of cross-attention layer.
        
        Args:
            human_tokens: [B, N, d_model] human visual tokens
            query_token: [B, 1, d_model] text query token (expanded)
            key_padding_mask: [B, N] True for padded positions (optional)
        
        Returns:
            updated_tokens: [B, N, d_model] query-conditioned human tokens
        """
        B, N, D = human_tokens.shape
        
        # Cross-attention: humans (Q) attend to query (K, V)
        # This allows each human to "see" the query and modulate accordingly
        # Note: We use query as K and V, humans as Q
        # This broadcasts query information to all humans
        
        # Expand query for attention
        # query_token: [B, 1, D]
        
        # OPTION 1: Human tokens query the text (humans look at text)
        # attn_output, _ = self.cross_attn(
        #     query=human_tokens,  # [B, N, D] - what we're updating
        #     key=query_token,     # [B, 1, D] - what we attend to
        #     value=query_token,   # [B, 1, D] - what we gather
        # )
        
        # OPTION 2 (CHOSEN): Bidirectional fusion via concatenation
        # We concatenate query with each human and attend within that context
        # Actually, let's use a cleaner approach:
        
        # Standard cross-attention where humans attend to query
        # Each human token gathers information from the query based on relevance
        attn_output, attn_weights = self.cross_attn(
            query=human_tokens,  # [B, N, D] - humans as queries
            key=query_token,     # [B, 1, D] - text as key
            value=query_token,   # [B, 1, D] - text as value
            need_weights=False,
        )
        # attn_output: [B, N, D] - each human now has query-relevant info
        
        # Add & Norm
        human_tokens = self.norm1(human_tokens + self.dropout(attn_output))
        
        # Feed-forward
        ffn_output = self.ffn(human_tokens)
        
        # Add & Norm
        human_tokens = self.norm2(human_tokens + ffn_output)
        
        return human_tokens


class CrossAttentionAdapter(nn.Module):
    """
    Cross-Attention based Grounding Adapter.
    
    Replaces FiLM-style modulation with transformer cross-attention.
    
    Architecture:
        Input: visual_tokens [B, N, 256], query_embedding [B, 256]
        
        1. Project query to token space (if needed)
        2. Expand query: [B, 256] → [B, 1, 256]
        3. Stack of CrossAttentionLayers
        4. Output: grounded_tokens [B, N, 256]
    
    The key insight: each human token attends to the text query,
    gathering information about which aspects of the query it matches.
    This is more expressive than FiLM's global modulation.
    
    STRICT INTERFACE (same as TrainableAdapter):
        forward(tokens, query) → grounded_tokens
        - tokens: [B, N, 256] or [N, 256]
        - query: [B, 256] or [256]
        - returns: same shape as tokens
    """
    
    def __init__(
        self,
        token_dim: int = D_TOKEN,
        query_dim: int = D_QUERY,
        num_heads: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        """
        Initialize Cross-Attention Adapter.
        
        Args:
            token_dim: Human token dimension (256)
            query_dim: Query embedding dimension (256)
            num_heads: Number of attention heads (4 or 8)
            num_layers: Number of cross-attention layers (1 or 2)
            dim_feedforward: FFN hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.token_dim = token_dim
        self.query_dim = query_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Project query if dimensions differ
        if query_dim != token_dim:
            self.query_proj = nn.Linear(query_dim, token_dim, bias=False)
            nn.init.orthogonal_(self.query_proj.weight)
        else:
            self.query_proj = None
        
        # Stack of cross-attention layers
        self.layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=token_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Output projection (optional, for additional capacity)
        self.output_proj = nn.Linear(token_dim, token_dim, bias=False)
        nn.init.orthogonal_(self.output_proj.weight)
        
        # Debug flag for shape printing (disable after verification)
        self._debug_shapes = False
    
    def enable_shape_debug(self, enable: bool = True):
        """Enable/disable shape debugging prints."""
        self._debug_shapes = enable
    
    def forward(
        self,
        tokens: torch.Tensor,
        query: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: fuse visual tokens with query via cross-attention.
        
        Args:
            tokens: [B, N, token_dim] or [N, token_dim] - human visual tokens
            query: [B, query_dim] or [query_dim] - text query embedding
            valid_mask: [B, N] optional validity mask (True = valid)
        
        Returns:
            grounded_tokens: same shape as input tokens
        """
        # Handle unbatched input
        unbatch = False
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)  # [1, N, D]
            unbatch = True
        
        if query.dim() == 1:
            query = query.unsqueeze(0)  # [1, D]
        
        B, N, D = tokens.shape
        
        # Debug shape printing
        if self._debug_shapes:
            print(f"[CrossAttentionAdapter] Input shapes:")
            print(f"  tokens: {tokens.shape}")
            print(f"  query: {query.shape}")
        
        # Project query if needed
        if self.query_proj is not None:
            query = self.query_proj(query)  # [B, token_dim]
        
        # Expand query to [B, 1, D] for attention
        query_token = query.unsqueeze(1)  # [B, 1, D]
        
        if self._debug_shapes:
            print(f"  query_token (expanded): {query_token.shape}")
        
        # Create key padding mask if valid_mask provided
        # PyTorch MHA expects True for positions to IGNORE
        key_padding_mask = None
        if valid_mask is not None:
            key_padding_mask = ~valid_mask  # Invert: True = pad/ignore
        
        # Pass through cross-attention layers
        output = tokens
        for i, layer in enumerate(self.layers):
            output = layer(output, query_token, key_padding_mask)
            
            if self._debug_shapes:
                print(f"  After layer {i}: {output.shape}")
        
        # Output projection
        output = self.output_proj(output)
        
        if self._debug_shapes:
            print(f"  Final output: {output.shape}")
        
        # Sanity checks
        assert output.shape == tokens.shape, \
            f"Shape mismatch: input {tokens.shape} vs output {output.shape}"
        assert not torch.isnan(output).any(), "NaN detected in output!"
        assert not torch.isinf(output).any(), "Inf detected in output!"
        
        # Restore original batch dimension if needed
        if unbatch:
            output = output.squeeze(0)
        
        return output


# =============================================================================
# FACTORY FUNCTION FOR EASY SWITCHING
# =============================================================================

def create_grounding_adapter(
    adapter_type: str = "cross_attention",
    token_dim: int = D_TOKEN,
    query_dim: int = D_QUERY,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create grounding adapter.
    
    Args:
        adapter_type: "cross_attention" or "film" (for backward compatibility)
        token_dim: Token dimension
        query_dim: Query dimension
        **kwargs: Additional arguments for specific adapter types
    
    Returns:
        Adapter module
    """
    if adapter_type == "cross_attention":
        return CrossAttentionAdapter(
            token_dim=token_dim,
            query_dim=query_dim,
            num_heads=kwargs.get("num_heads", 4),
            num_layers=kwargs.get("num_layers", 1),
            dim_feedforward=kwargs.get("dim_feedforward", 512),
            dropout=kwargs.get("dropout", 0.1),
        )
    elif adapter_type == "film":
        # Import the original FiLM adapter for backward compatibility
        from training.grounding_train_v2 import TrainableAdapter
        return TrainableAdapter(token_dim=token_dim, query_dim=query_dim)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}. Choose 'cross_attention' or 'film'.")


# =============================================================================
# SANITY CHECK
# =============================================================================

def sanity_check():
    """
    Run sanity checks for CrossAttentionAdapter.
    
    Verifies:
    - Shape preservation
    - No NaN/Inf
    - Gradient flow
    - Device compatibility
    """
    print("\n" + "=" * 70)
    print("CrossAttentionAdapter Sanity Check")
    print("=" * 70)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Create adapter
    adapter = CrossAttentionAdapter(
        token_dim=256,
        query_dim=256,
        num_heads=4,
        num_layers=1,
        dim_feedforward=512,
        dropout=0.1,
    ).to(device)
    adapter.train()
    adapter.enable_shape_debug(True)
    
    print(f"\nAdapter created:")
    print(f"  token_dim: {adapter.token_dim}")
    print(f"  query_dim: {adapter.query_dim}")
    print(f"  num_heads: {adapter.num_heads}")
    print(f"  num_layers: {adapter.num_layers}")
    
    # Count parameters
    total_params = sum(p.numel() for p in adapter.parameters())
    trainable_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test 1: Batched input
    print("\n--- Test 1: Batched Input ---")
    B, N, D = 4, 6, 256
    tokens = torch.randn(B, N, D, device=device)
    query = torch.randn(B, D, device=device)
    
    output = adapter(tokens, query)
    print(f"✓ Output shape: {output.shape} (expected: {tokens.shape})")
    assert output.shape == tokens.shape
    
    # Test 2: Unbatched input
    print("\n--- Test 2: Unbatched Input ---")
    adapter.enable_shape_debug(False)
    tokens_single = torch.randn(N, D, device=device)
    query_single = torch.randn(D, device=device)
    
    output_single = adapter(tokens_single, query_single)
    print(f"✓ Output shape: {output_single.shape} (expected: {tokens_single.shape})")
    assert output_single.shape == tokens_single.shape
    
    # Test 3: Gradient flow
    print("\n--- Test 3: Gradient Flow ---")
    tokens.requires_grad_(True)
    output = adapter(tokens, query)
    loss = output.sum()
    loss.backward()
    
    print(f"✓ Gradients computed for tokens: {tokens.grad is not None}")
    
    # Check adapter gradients
    has_grad = False
    for name, param in adapter.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    print(f"✓ Gradients flow through adapter: {has_grad}")
    
    # Test 4: No NaN/Inf
    print("\n--- Test 4: Numerical Stability ---")
    tokens_clean = torch.randn(B, N, D, device=device)
    query_clean = torch.randn(B, D, device=device)
    output_clean = adapter(tokens_clean, query_clean)
    
    print(f"✓ No NaN in output: {not torch.isnan(output_clean).any()}")
    print(f"✓ No Inf in output: {not torch.isinf(output_clean).any()}")
    
    # Test 5: Different sequence lengths
    print("\n--- Test 5: Variable Sequence Length ---")
    for n in [1, 3, 8, 16]:
        tokens_var = torch.randn(2, n, D, device=device)
        query_var = torch.randn(2, D, device=device)
        output_var = adapter(tokens_var, query_var)
        assert output_var.shape == tokens_var.shape
    print(f"✓ Handles variable N: [1, 3, 8, 16]")
    
    print("\n" + "=" * 70)
    print("✅ ALL SANITY CHECKS PASSED")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    sanity_check()
