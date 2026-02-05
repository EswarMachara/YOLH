# -*- coding: utf-8 -*-
"""
Text-Visual Alignment Adapter (Phase-3)

Token-level cross-modal attention for fine-grained grounding.
Replaces sentence-level embeddings with word-level alignment.

MOTIVATION:
- Phase-1: Cross-attention uses single query vector per caption
- Phase-3: Multi-token alignment captures fine-grained descriptions
  e.g., "person [on the left]" → spatial tokens attend to location words

ARCHITECTURE:
    Input:  visual_tokens [B, N, 256], caption_tokens [B, T, 256], caption_mask [B, T]
    
    Step 1: Text → Visual Cross-Attention
        - Each human token attends to caption tokens
        - Masked attention (ignore padding)
    
    Step 2: Visual → Text Refinement (optional)
        - Caption-aware visual enhancement
        - Can be disabled via config
    
    Step 3: Fusion & Projection
        - Residual connection + LayerNorm
    
    Output: aligned_tokens [B, N, 256]

USAGE:
    adapter = TextVisualAlignmentAdapter(
        token_dim=256,
        num_heads=4,
        num_layers=1,
        dropout=0.1,
        bidirectional=True,
    )
    aligned = adapter(visual_tokens, caption_tokens, caption_mask)
"""

import torch
import torch.nn as nn
from typing import Optional


class TextVisualAlignmentAdapter(nn.Module):
    """
    Token-level cross-modal alignment for grounding.
    
    Implements bi-directional attention between:
    - Visual tokens (humans from YOLO)
    - Text tokens (words from caption)
    """
    
    def __init__(
        self,
        token_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        """
        Args:
            token_dim: Dimension of visual and text tokens (must match)
            num_heads: Number of attention heads
            num_layers: Number of cross-attention layers
            dim_feedforward: FFN hidden dimension
            dropout: Dropout rate
            bidirectional: If True, use visual→text refinement
        """
        super().__init__()
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Text → Visual Cross-Attention Layers
        # Query: visual tokens, Key/Value: text tokens
        self.text_to_visual_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=token_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        
        # Optional: Visual → Text Refinement
        if bidirectional:
            self.visual_to_text_layers = nn.ModuleList([
                nn.TransformerDecoderLayer(
                    d_model=token_dim,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ])
        else:
            self.visual_to_text_layers = None
        
        # Final projection (optional refinement)
        self.output_proj = nn.Linear(token_dim, token_dim, bias=False)
        nn.init.orthogonal_(self.output_proj.weight)
    
    def forward(
        self,
        visual_tokens: torch.Tensor,
        caption_tokens: torch.Tensor,
        caption_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            visual_tokens: [B, N, token_dim] - Human visual embeddings
            caption_tokens: [B, T, token_dim] - Caption token embeddings
            caption_mask: [B, T] - Boolean mask (True = valid token)
        
        Returns:
            aligned_tokens: [B, N, token_dim] - Text-aligned visual tokens
        """
        B, N, D = visual_tokens.shape
        _, T, _ = caption_tokens.shape
        
        # Convert mask format for TransformerDecoderLayer
        # TransformerDecoderLayer expects: True = IGNORE, False = ATTEND
        # Our input: True = VALID, False = PADDING
        # So we need to invert
        if caption_mask is not None:
            # Invert: True (valid) → False (attend), False (pad) → True (ignore)
            memory_key_padding_mask = ~caption_mask  # [B, T]
        else:
            memory_key_padding_mask = None
        
        # Step 1: Text → Visual Cross-Attention
        # Each human token queries caption tokens
        aligned_visual = visual_tokens  # Start with original
        
        for layer in self.text_to_visual_layers:
            aligned_visual = layer(
                tgt=aligned_visual,                    # [B, N, D] - visual tokens (query)
                memory=caption_tokens,                  # [B, T, D] - text tokens (key/value)
                memory_key_padding_mask=memory_key_padding_mask,  # [B, T] - mask padding
            )
        
        # Step 2: Visual → Text Refinement (optional)
        if self.bidirectional and self.visual_to_text_layers is not None:
            refined_text = caption_tokens
            
            # No padding mask needed for visual tokens (all valid)
            for layer in self.visual_to_text_layers:
                refined_text = layer(
                    tgt=refined_text,                  # [B, T, D] - text tokens
                    memory=aligned_visual,              # [B, N, D] - visual tokens
                )
            
            # Re-align visual with refined text
            for layer in self.text_to_visual_layers:
                aligned_visual = layer(
                    tgt=aligned_visual,
                    memory=refined_text,
                    memory_key_padding_mask=memory_key_padding_mask,
                )
        
        # Step 3: Final projection
        output = self.output_proj(aligned_visual)
        
        return output
    
    def get_attention_entropy(
        self,
        visual_tokens: torch.Tensor,
        caption_tokens: torch.Tensor,
        caption_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute average attention entropy for debugging.
        Higher entropy = more diffuse attention across words.
        Lower entropy = focused on specific words.
        
        Returns:
            entropy: [B] - Average entropy per sample
        """
        B, N, D = visual_tokens.shape
        _, T, _ = caption_tokens.shape
        
        # Quick entropy estimate using first layer
        layer = self.text_to_visual_layers[0]
        
        # Get attention weights (this is a simplification)
        # In practice, we'd need to hook into MultiheadAttention
        # For now, return a placeholder
        
        # Placeholder: uniform entropy
        return torch.ones(B, device=visual_tokens.device) * 1.0


def create_text_visual_alignment_adapter(
    token_dim: int = 256,
    num_heads: int = 4,
    num_layers: int = 1,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
    bidirectional: bool = True,
) -> TextVisualAlignmentAdapter:
    """
    Factory function for creating text-visual alignment adapter.
    
    Args:
        token_dim: Token dimension (must match visual & text)
        num_heads: Number of attention heads
        num_layers: Number of cross-attention layers
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
        bidirectional: Enable visual→text refinement
    
    Returns:
        TextVisualAlignmentAdapter instance
    """
    return TextVisualAlignmentAdapter(
        token_dim=token_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        bidirectional=bidirectional,
    )


# =============================================================================
# SANITY CHECK
# =============================================================================

def sanity_check():
    """Built-in sanity check for TextVisualAlignmentAdapter."""
    print("\n" + "="*70)
    print("Text-Visual Alignment Adapter Sanity Check (Phase-3)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create adapter
    adapter = TextVisualAlignmentAdapter(
        token_dim=256,
        num_heads=4,
        num_layers=1,
        dropout=0.1,
        bidirectional=True,
    ).to(device)
    
    print(f"\n✓ Adapter created")
    print(f"  Device: {device}")
    print(f"  Bidirectional: True")
    print(f"  Num heads: 4")
    print(f"  Num layers: 1")
    
    # Test 1: Variable batch size, humans, caption length
    print("\n--- Test 1: Variable Shapes ---")
    B, N, T, D = 2, 5, 10, 256
    
    visual_tokens = torch.randn(B, N, D, device=device)
    caption_tokens = torch.randn(B, T, D, device=device)
    caption_mask = torch.ones(B, T, dtype=torch.bool, device=device)
    
    # Add padding to second sample
    caption_mask[1, 7:] = False  # Last 3 tokens are padding
    
    with torch.no_grad():
        output = adapter(visual_tokens, caption_tokens, caption_mask)
    
    print(f"  Input visual: {visual_tokens.shape}")
    print(f"  Input caption: {caption_tokens.shape}")
    print(f"  Caption mask: {caption_mask.shape}")
    print(f"  Output: {output.shape}")
    assert output.shape == (B, N, D), f"Expected {(B, N, D)}, got {output.shape}"
    print(f"  ✓ Shape preserved: [B={B}, N={N}, D={D}]")
    
    # Test 2: Gradient flow
    print("\n--- Test 2: Gradient Flow ---")
    adapter.train()
    visual_tokens.requires_grad = True
    caption_tokens.requires_grad = True
    
    output = adapter(visual_tokens, caption_tokens, caption_mask)
    loss = output.sum()
    loss.backward()
    
    assert visual_tokens.grad is not None, "No gradient for visual tokens"
    assert caption_tokens.grad is not None, "No gradient for caption tokens"
    print(f"  ✓ Gradients flow through adapter")
    print(f"    Visual grad norm: {visual_tokens.grad.norm().item():.4f}")
    print(f"    Caption grad norm: {caption_tokens.grad.norm().item():.4f}")
    
    # Test 3: No NaN/Inf
    print("\n--- Test 3: Numerical Stability ---")
    adapter.eval()
    with torch.no_grad():
        output = adapter(visual_tokens, caption_tokens, caption_mask)
    
    assert not torch.isnan(output).any(), "NaN detected in output"
    assert not torch.isinf(output).any(), "Inf detected in output"
    print(f"  ✓ No NaN/Inf in output")
    print(f"    Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # Test 4: Unidirectional mode
    print("\n--- Test 4: Unidirectional Mode ---")
    adapter_unidirectional = TextVisualAlignmentAdapter(
        token_dim=256,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        bidirectional=False,
    ).to(device)
    
    with torch.no_grad():
        output_uni = adapter_unidirectional(visual_tokens, caption_tokens, caption_mask)
    
    assert output_uni.shape == (B, N, D)
    print(f"  ✓ Unidirectional mode works")
    print(f"    Output shape: {output_uni.shape}")
    
    # Test 5: No mask (all tokens valid)
    print("\n--- Test 5: No Padding Mask ---")
    with torch.no_grad():
        output_no_mask = adapter(visual_tokens, caption_tokens, caption_mask=None)
    
    assert output_no_mask.shape == (B, N, D)
    print(f"  ✓ Works without mask")
    
    # Test 6: Single human, single caption token (edge case)
    print("\n--- Test 6: Edge Cases ---")
    visual_single = torch.randn(1, 1, D, device=device)
    caption_single = torch.randn(1, 1, D, device=device)
    mask_single = torch.ones(1, 1, dtype=torch.bool, device=device)
    
    with torch.no_grad():
        output_single = adapter(visual_single, caption_single, mask_single)
    
    assert output_single.shape == (1, 1, D)
    print(f"  ✓ Edge case (N=1, T=1): {output_single.shape}")
    
    print("\n" + "="*70)
    print("✅ ALL SANITY CHECKS PASSED")
    print("="*70)


if __name__ == "__main__":
    sanity_check()
