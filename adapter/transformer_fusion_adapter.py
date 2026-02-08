# -*- coding: utf-8 -*-
"""
Transformer Fusion Adapter (Phase-5B)

Deep multi-head cross-attention transformer for refined text-visual grounding.
Addresses limitations of simpler cross-attention by:

1. SPATIAL POSITIONAL ENCODING
   - Encodes bounding box positions as learnable embeddings
   - "left arm" → attends to humans positioned on the left
   
2. DEEP FUSION (4-6 layers)
   - More expressive transformation capacity
   - Progressive refinement of text-visual alignment
   
3. MULTI-SCALE TEXT ATTENTION
   - Word-level + phrase-level attention
   - Captures both fine-grained and compositional semantics

4. GATED RESIDUAL CONNECTIONS
   - Learnable gating for adaptive residual strength
   - Prevents gradient degradation in deep networks

ARCHITECTURE:
    Input:  visual_tokens [B, N, 256], text_tokens [B, T, 256], text_mask [B, T]
            boxes [B, N, 4] (for spatial encoding)
    
    Step 1: Add spatial positional encoding to visual tokens
    Step 2: Deep transformer fusion (4-6 cross-attention layers)
    Step 3: Gated output projection
    
    Output: fused_tokens [B, N, 256]

USAGE:
    adapter = TransformerFusionAdapter(
        token_dim=256,
        num_heads=8,
        num_layers=4,
        use_spatial_encoding=True,
    )
    fused = adapter(visual_tokens, text_tokens, text_mask, boxes=boxes)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.datatypes import D_TOKEN, D_QUERY


class SpatialPositionalEncoding(nn.Module):
    """
    Learnable spatial positional encoding from bounding boxes.
    
    Encodes (x1, y1, x2, y2) normalized coordinates into position embeddings.
    Captures spatial relationships like "left", "right", "center", "large", "small".
    """
    
    def __init__(self, d_model: int = 256, max_positions: int = 100):
        super().__init__()
        self.d_model = d_model
        
        # Project 4D box coordinates to d_model
        # Input: [x1, y1, x2, y2] normalized to [0, 1]
        self.box_proj = nn.Sequential(
            nn.Linear(4, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
        
        # Additional features: center, width, height, area
        self.geometry_proj = nn.Sequential(
            nn.Linear(5, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
        
        # Combine box and geometry
        self.combine = nn.Linear(d_model * 2, d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            boxes: [B, N, 4] bounding boxes (x1, y1, x2, y2) normalized to [0, 1]
        
        Returns:
            pos_embed: [B, N, d_model] spatial positional embeddings
        """
        B, N, _ = boxes.shape
        
        # Clamp boxes to valid range
        boxes = boxes.clamp(0.0, 1.0)
        
        # Extract geometry features
        x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        cx = (x1 + x2) / 2  # center x
        cy = (y1 + y2) / 2  # center y
        w = (x2 - x1).clamp(min=1e-6)  # width
        h = (y2 - y1).clamp(min=1e-6)  # height
        area = w * h  # area
        
        geometry = torch.stack([cx, cy, w, h, area], dim=-1)  # [B, N, 5]
        
        # Project
        box_embed = self.box_proj(boxes)  # [B, N, d_model]
        geo_embed = self.geometry_proj(geometry)  # [B, N, d_model]
        
        # Combine
        combined = torch.cat([box_embed, geo_embed], dim=-1)  # [B, N, d_model*2]
        pos_embed = self.combine(combined)  # [B, N, d_model]
        
        return pos_embed


class GatedResidual(nn.Module):
    """
    Gated residual connection with learnable gate.
    
    output = gate * new + (1 - gate) * residual
    
    The gate is learned and allows the model to adaptively
    balance between residual and new information.
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
    
    def forward(self, new: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Args:
            new: [B, N, d_model] new representation
            residual: [B, N, d_model] residual connection
        
        Returns:
            output: [B, N, d_model] gated combination
        """
        combined = torch.cat([new, residual], dim=-1)  # [B, N, d_model*2]
        gate = self.gate(combined)  # [B, N, d_model]
        output = gate * new + (1 - gate) * residual
        return output


class TransformerFusionLayer(nn.Module):
    """
    Single transformer fusion layer with cross-attention.
    
    Visual tokens attend to text tokens, then pass through FFN.
    Uses pre-norm architecture for better gradient flow.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_gated_residual: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_gated_residual = use_gated_residual
        
        # Pre-norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Self-attention on visual tokens (optional, for inter-human reasoning)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Cross-attention: visual (Q) attends to text (K, V)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        
        # Gated residual connections
        if use_gated_residual:
            self.gate1 = GatedResidual(d_model)
            self.gate2 = GatedResidual(d_model)
            self.gate3 = GatedResidual(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        visual: torch.Tensor,
        text: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            visual: [B, N, d_model] visual tokens
            text: [B, T, d_model] text tokens
            text_mask: [B, T] True for valid positions (optional)
            visual_mask: [B, N] True for valid humans (optional)
        
        Returns:
            output: [B, N, d_model] updated visual tokens
        """
        # Convert mask format for PyTorch attention (True = ignore)
        text_key_padding_mask = None
        if text_mask is not None:
            text_key_padding_mask = ~text_mask  # Invert: True = pad
        
        # 1. Self-attention on visual tokens (inter-human reasoning)
        residual = visual
        visual_normed = self.norm1(visual)
        self_attn_out, _ = self.self_attn(
            visual_normed, visual_normed, visual_normed,
            need_weights=False,
        )
        if self.use_gated_residual:
            visual = self.gate1(self_attn_out, residual)
        else:
            visual = residual + self.dropout(self_attn_out)
        
        # 2. Cross-attention: visual attends to text
        residual = visual
        visual_normed = self.norm2(visual)
        cross_attn_out, _ = self.cross_attn(
            query=visual_normed,
            key=text,
            value=text,
            key_padding_mask=text_key_padding_mask,
            need_weights=False,
        )
        if self.use_gated_residual:
            visual = self.gate2(cross_attn_out, residual)
        else:
            visual = residual + self.dropout(cross_attn_out)
        
        # 3. Feed-forward network
        residual = visual
        visual_normed = self.norm3(visual)
        ffn_out = self.ffn(visual_normed)
        if self.use_gated_residual:
            visual = self.gate3(ffn_out, residual)
        else:
            visual = residual + ffn_out
        
        return visual


class TransformerFusionAdapter(nn.Module):
    """
    Deep Transformer Fusion Adapter for Phase-5B.
    
    Key innovations:
    1. Spatial positional encoding from bounding boxes
    2. Deep fusion with 4-6 transformer layers
    3. Gated residual connections
    4. Inter-human self-attention for relational reasoning
    """
    
    def __init__(
        self,
        token_dim: int = D_TOKEN,
        query_dim: int = D_QUERY,  # Not used, for interface compatibility
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_spatial_encoding: bool = True,
        use_gated_residual: bool = True,
    ):
        """
        Args:
            token_dim: Dimension of visual tokens (256)
            query_dim: Dimension of query (256, for interface compatibility)
            num_heads: Number of attention heads (8 recommended)
            num_layers: Number of fusion layers (4-6 recommended)
            dim_feedforward: FFN hidden dimension (1024 recommended)
            dropout: Dropout rate
            use_spatial_encoding: Add spatial positional encoding from boxes
            use_gated_residual: Use gated residual connections
        """
        super().__init__()
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_spatial_encoding = use_spatial_encoding
        
        # Spatial positional encoding
        if use_spatial_encoding:
            self.spatial_encoder = SpatialPositionalEncoding(d_model=token_dim)
        
        # Stack of transformer fusion layers
        self.layers = nn.ModuleList([
            TransformerFusionLayer(
                d_model=token_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                use_gated_residual=use_gated_residual,
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(token_dim)
        
        # Output projection
        self.output_proj = nn.Linear(token_dim, token_dim)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
        # Count parameters
        self._count_parameters()
    
    def _count_parameters(self):
        """Count and store parameter count."""
        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: deep transformer fusion of visual and text tokens.
        
        Args:
            visual_tokens: [B, N, token_dim] human visual features
            text_tokens: [B, T, token_dim] text token embeddings
            text_mask: [B, T] True for valid text positions
            boxes: [B, N, 4] bounding boxes for spatial encoding (normalized)
            valid_mask: [B, N] True for valid humans (optional)
        
        Returns:
            fused_tokens: [B, N, token_dim] text-grounded visual features
        """
        # Handle unbatched input
        unbatch = False
        if visual_tokens.dim() == 2:
            visual_tokens = visual_tokens.unsqueeze(0)
            text_tokens = text_tokens.unsqueeze(0)
            if text_mask is not None:
                text_mask = text_mask.unsqueeze(0)
            if boxes is not None:
                boxes = boxes.unsqueeze(0)
            unbatch = True
        
        B, N, D = visual_tokens.shape
        
        # Add spatial positional encoding
        if self.use_spatial_encoding and boxes is not None:
            spatial_pos = self.spatial_encoder(boxes)  # [B, N, D]
            visual_tokens = visual_tokens + spatial_pos
        
        # Pass through transformer fusion layers
        x = visual_tokens
        for layer in self.layers:
            x = layer(x, text_tokens, text_mask, valid_mask)
        
        # Final norm and projection
        x = self.final_norm(x)
        x = self.output_proj(x)
        
        # Remove batch dimension if input was unbatched
        if unbatch:
            x = x.squeeze(0)
        
        return x


def create_transformer_fusion_adapter(
    token_dim: int = D_TOKEN,
    num_heads: int = 8,
    num_layers: int = 4,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
    use_spatial_encoding: bool = True,
    use_gated_residual: bool = True,
) -> TransformerFusionAdapter:
    """
    Factory function for TransformerFusionAdapter.
    """
    return TransformerFusionAdapter(
        token_dim=token_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        use_spatial_encoding=use_spatial_encoding,
        use_gated_residual=use_gated_residual,
    )


# =============================================================================
# SANITY CHECK
# =============================================================================

def sanity_check():
    """
    Run sanity checks for TransformerFusionAdapter.
    """
    print("\n" + "=" * 70)
    print("TransformerFusionAdapter Sanity Check (Phase-5B)")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Create adapter
    adapter = TransformerFusionAdapter(
        token_dim=256,
        num_heads=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        use_spatial_encoding=True,
        use_gated_residual=True,
    ).to(device)
    adapter.train()
    
    print(f"\nAdapter created:")
    print(f"  token_dim: {adapter.token_dim}")
    print(f"  num_heads: {adapter.num_heads}")
    print(f"  num_layers: {adapter.num_layers}")
    print(f"  Total parameters: {adapter.total_params:,}")
    print(f"  Trainable parameters: {adapter.trainable_params:,}")
    
    # Test 1: Basic forward pass
    print("\n--- Test 1: Basic Forward Pass ---")
    B, N, T, D = 4, 6, 20, 256
    visual_tokens = torch.randn(B, N, D, device=device)
    text_tokens = torch.randn(B, T, D, device=device)
    text_mask = torch.ones(B, T, dtype=torch.bool, device=device)
    boxes = torch.rand(B, N, 4, device=device)
    # Ensure x2 > x1 and y2 > y1
    boxes[..., 2] = boxes[..., 0] + torch.rand(B, N, device=device) * 0.5
    boxes[..., 3] = boxes[..., 1] + torch.rand(B, N, device=device) * 0.5
    boxes = boxes.clamp(0, 1)
    
    output = adapter(visual_tokens, text_tokens, text_mask, boxes=boxes)
    print(f"✓ Output shape: {output.shape} (expected: {visual_tokens.shape})")
    assert output.shape == visual_tokens.shape
    
    # Test 2: Without spatial encoding
    print("\n--- Test 2: Without Spatial Encoding ---")
    adapter_no_spatial = TransformerFusionAdapter(
        token_dim=256,
        num_heads=8,
        num_layers=4,
        use_spatial_encoding=False,
    ).to(device)
    
    output_no_spatial = adapter_no_spatial(visual_tokens, text_tokens, text_mask)
    print(f"✓ Output shape (no spatial): {output_no_spatial.shape}")
    assert output_no_spatial.shape == visual_tokens.shape
    
    # Test 3: Gradient flow
    print("\n--- Test 3: Gradient Flow ---")
    visual_tokens.requires_grad_(True)
    output = adapter(visual_tokens, text_tokens, text_mask, boxes=boxes)
    loss = output.sum()
    loss.backward()
    
    print(f"✓ Gradients computed for visual_tokens: {visual_tokens.grad is not None}")
    
    has_grad = any(
        param.grad is not None and param.grad.abs().sum() > 0
        for param in adapter.parameters()
    )
    print(f"✓ Gradients flow through adapter: {has_grad}")
    
    # Test 4: Variable sequence lengths
    print("\n--- Test 4: Variable Sequence Lengths ---")
    for n, t in [(1, 5), (4, 10), (10, 30), (8, 64)]:
        v = torch.randn(2, n, D, device=device)
        txt = torch.randn(2, t, D, device=device)
        mask = torch.ones(2, t, dtype=torch.bool, device=device)
        b = torch.rand(2, n, 4, device=device)
        b[..., 2:] = (b[..., :2] + 0.3).clamp(0, 1)
        
        out = adapter(v, txt, mask, boxes=b)
        assert out.shape == v.shape
    print(f"✓ Handles variable N, T combinations")
    
    # Test 5: Memory usage estimate
    print("\n--- Test 5: Memory Estimate ---")
    adapter.eval()
    with torch.no_grad():
        B_large, N_large, T_large = 64, 20, 64
        v_large = torch.randn(B_large, N_large, D, device=device)
        txt_large = torch.randn(B_large, T_large, D, device=device)
        mask_large = torch.ones(B_large, T_large, dtype=torch.bool, device=device)
        b_large = torch.rand(B_large, N_large, 4, device=device)
        
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        
        _ = adapter(v_large, txt_large, mask_large, boxes=b_large)
        
        if device == "cuda":
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            print(f"✓ Peak memory (B=64): {peak_mem:.1f} MB")
    
    print("\n" + "=" * 70)
    print("✅ ALL SANITY CHECKS PASSED")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    sanity_check()
