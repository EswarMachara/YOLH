"""
Phase 5: Dynamic Grounding Adapter

Fuses HumanToken and QueryEmbedding into query-aware human tokens.
No LLM usage. CPU-only. Deterministic. No training.
"""

import os
import sys

# Disable TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from core.datatypes import HumanToken, QueryEmbedding, D_TOKEN, D_QUERY
from core.assertions import assert_human_token


# =============================================================================
# TASK 1: CONTRACT RECONFIRMATION
# =============================================================================

def task1_contract_reconfirmation():
    """
    TASK 1: Explicitly restate input/output contracts.
    """
    print("\n" + "=" * 70)
    print("TASK 1: CONTRACT RECONFIRMATION")
    print("=" * 70)
    
    print(f"""
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    ADAPTER I/O CONTRACTS                            │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │   INPUT:                                                            │
    │   ───────                                                           │
    │   HumanToken.tokens:       [B, N, {D_TOKEN}]  (float32)                  │
    │   HumanToken.valid:        [B, N]       (bool)                      │
    │   QueryEmbedding.embedding: [B, {D_QUERY}]   (float32, L2-normalized)   │
    │                                                                     │
    │   OUTPUT:                                                           │
    │   ────────                                                          │
    │   GroundedHumanToken.tokens: [B, N, {D_TOKEN}]  (float32)                │
    │   GroundedHumanToken.valid:  [B, N]       (bool, unchanged)         │
    │                                                                     │
    │   CONSTRAINTS:                                                      │
    │   ────────────                                                      │
    │   - No dimensional changes (256 → 256)                              │
    │   - Valid mask preserved from input                                 │
    │   - Output is query-conditioned                                     │
    │   - Deterministic: same inputs → same outputs                       │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    """)
    
    # Verify constants match
    assert D_TOKEN == 256, f"D_TOKEN mismatch: {D_TOKEN} != 256"
    assert D_QUERY == 256, f"D_QUERY mismatch: {D_QUERY} != 256"
    
    print(f"  D_TOKEN = {D_TOKEN} ✓")
    print(f"  D_QUERY = {D_QUERY} ✓")
    print(f"  [PASS] Contracts confirmed")
    
    return True


# =============================================================================
# TASK 2: FUSION MECHANISM SELECTION
# =============================================================================

def task2_fusion_selection():
    """
    TASK 2: Select and lock fusion mechanism.
    """
    print("\n" + "=" * 70)
    print("TASK 2: FUSION MECHANISM SELECTION (LOCKED)")
    print("=" * 70)
    
    print(f"""
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    FUSION MECHANISM OPTIONS                         │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │   Option 1: Concatenation + MLP                                     │
    │   ─────────────────────────────────────────────────────────────────│
    │   - Concat([human, query]) → MLP → output                          │
    │   - Simple, deterministic                                           │
    │   - Requires projection back to 256D                                │
    │   - No explicit attention                                           │
    │                                                                     │
    │   Option 2: Cross-Attention (single-head)                           │
    │   ─────────────────────────────────────────────────────────────────│
    │   - Query attends to human tokens                                   │
    │   - Learnable Q, K, V projections                                   │
    │   - More expressive but heavier                                     │
    │   - Still deterministic (no dropout)                                │
    │                                                                     │
    │   Option 3: Additive Modulation (FiLM-style) ← SELECTED            │
    │   ─────────────────────────────────────────────────────────────────│
    │   - gamma, beta = f(query)                                          │
    │   - output = gamma * human + beta                                   │
    │   - Lightweight, deterministic                                      │
    │   - Query directly modulates each human token                       │
    │   - Feature-wise Linear Modulation proven effective                 │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │                    SELECTED: FiLM-STYLE MODULATION                  │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │   JUSTIFICATION:                                                    │
    │                                                                     │
    │   1. LIGHTWEIGHT: Only two linear layers to generate gamma/beta    │
    │      - No large attention matrices                                  │
    │      - CPU-friendly computation                                     │
    │                                                                     │
    │   2. DETERMINISTIC: Pure linear operations                          │
    │      - gamma = Linear(query)                                        │
    │      - beta = Linear(query)                                         │
    │      - No softmax normalization that could cause numerical issues   │
    │                                                                     │
    │   3. QUERY-CONDITIONED: Each query produces unique gamma/beta      │
    │      - Different queries → different modulation → different outputs │
    │      - Same query → identical modulation → reproducible outputs     │
    │                                                                     │
    │   4. DIMENSION PRESERVING: Output = gamma * input + beta            │
    │      - Input: [N, 256], gamma: [256], beta: [256]                   │
    │      - Output: [N, 256] (no shape change)                           │
    │                                                                     │
    │   5. GATING COMPATIBLE: Can add scalar gate on top                  │
    │      - gate = sigmoid(f(query))                                     │
    │      - final = gate * (gamma * human + beta)                        │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    """)
    
    return "FiLM-style additive modulation"


# =============================================================================
# TASKS 3-4: ADAPTER IMPLEMENTATION WITH GATING
# =============================================================================

class DynamicGroundingAdapter(nn.Module):
    """
    Fuses HumanToken and QueryEmbedding using FiLM-style modulation.
    
    Architecture:
        1. Query → gamma, beta (FiLM parameters)
        2. Query → gate (scalar gating)
        3. modulated = gamma * human_token + beta
        4. output = gate * modulated + (1 - gate) * human_token
    
    The residual connection ensures graceful degradation when gate → 0.
    
    Input:
        HumanToken.tokens: [B, N, 256]
        QueryEmbedding.embedding: [B, 256]
    
    Output:
        GroundedHumanToken.tokens: [B, N, 256]
    """
    
    def __init__(self, token_dim: int = D_TOKEN, query_dim: int = D_QUERY):
        """
        Initialize the adapter.
        
        Args:
            token_dim: Human token dimension (default: 256)
            query_dim: Query embedding dimension (default: 256)
        """
        super().__init__()
        
        self.token_dim = token_dim
        self.query_dim = query_dim
        
        # FiLM parameter generators
        # gamma: multiplicative modulation
        torch.manual_seed(47)
        self.gamma_generator = nn.Linear(query_dim, token_dim, bias=True)
        nn.init.xavier_uniform_(self.gamma_generator.weight)
        nn.init.ones_(self.gamma_generator.bias)  # Initialize gamma ≈ 1
        
        # beta: additive modulation
        torch.manual_seed(48)
        self.beta_generator = nn.Linear(query_dim, token_dim, bias=True)
        nn.init.xavier_uniform_(self.beta_generator.weight)
        nn.init.zeros_(self.beta_generator.bias)  # Initialize beta ≈ 0
        
        # Scalar gate generator
        torch.manual_seed(49)
        self.gate_generator = nn.Sequential(
            nn.Linear(query_dim, 64, bias=True),
            nn.Tanh(),  # Non-linearity for expressiveness
            nn.Linear(64, 1, bias=True),
        )
        # Initialize gate layers
        nn.init.xavier_uniform_(self.gate_generator[0].weight)
        nn.init.zeros_(self.gate_generator[0].bias)
        nn.init.xavier_uniform_(self.gate_generator[2].weight)
        nn.init.zeros_(self.gate_generator[2].bias)
        
        # Freeze all parameters (no training)
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def compute_film_params(self, query_emb: torch.Tensor) -> tuple:
        """
        Compute FiLM parameters (gamma, beta) from query embedding.
        
        Args:
            query_emb: [B, query_dim]
            
        Returns:
            gamma: [B, 1, token_dim] - multiplicative factor
            beta: [B, 1, token_dim] - additive bias
        """
        # Generate gamma and beta
        gamma = self.gamma_generator(query_emb)  # [B, token_dim]
        beta = self.beta_generator(query_emb)    # [B, token_dim]
        
        # Add dimension for broadcasting over N humans
        gamma = gamma.unsqueeze(1)  # [B, 1, token_dim]
        beta = beta.unsqueeze(1)    # [B, 1, token_dim]
        
        return gamma, beta
    
    def compute_gate(self, query_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute scalar gate from query embedding.
        
        Args:
            query_emb: [B, query_dim]
            
        Returns:
            gate: [B, 1, 1] - scalar gate in (0, 1)
        """
        gate_logit = self.gate_generator(query_emb)  # [B, 1]
        gate = torch.sigmoid(gate_logit)  # [B, 1], range (0, 1)
        gate = gate.unsqueeze(-1)  # [B, 1, 1] for broadcasting
        
        return gate
    
    def forward(
        self, 
        human_token: HumanToken, 
        query_emb: QueryEmbedding
    ) -> HumanToken:
        """
        Fuse HumanToken with QueryEmbedding.
        
        Args:
            human_token: HumanToken with tokens [B, N, 256] and valid [B, N]
            query_emb: QueryEmbedding with embedding [B, 256]
            
        Returns:
            GroundedHumanToken (as HumanToken) with query-conditioned tokens
        """
        tokens = human_token.tokens      # [B, N, 256]
        valid = human_token.valid        # [B, N]
        query = query_emb.embedding      # [B, 256]
        
        B, N, D = tokens.shape
        
        # Handle zero-human case
        if N == 0:
            return HumanToken(tokens=tokens, valid=valid)
        
        with torch.no_grad():
            # Compute FiLM parameters
            gamma, beta = self.compute_film_params(query)  # [B, 1, D], [B, 1, D]
            
            # Compute gate
            gate = self.compute_gate(query)  # [B, 1, 1]
            
            # Apply FiLM modulation
            modulated = gamma * tokens + beta  # [B, N, D]
            
            # Apply gating with residual connection
            # output = gate * modulated + (1 - gate) * tokens
            output = gate * modulated + (1.0 - gate) * tokens  # [B, N, D]
            
            # Mask invalid tokens (set to zero)
            valid_mask = valid.unsqueeze(-1).float()  # [B, N, 1]
            output = output * valid_mask
        
        return HumanToken(tokens=output, valid=valid)
    
    def get_gate_value(self, query_emb: QueryEmbedding) -> float:
        """
        Get the scalar gate value for a query (for inspection).
        
        Args:
            query_emb: QueryEmbedding
            
        Returns:
            Gate value as float
        """
        with torch.no_grad():
            gate = self.compute_gate(query_emb.embedding)
            return gate.squeeze().item()


# =============================================================================
# TASK 5: SHAPE & DETERMINISM CHECK
# =============================================================================

def task5_shape_and_determinism(adapter: DynamicGroundingAdapter):
    """
    TASK 5: Verify shape correctness and determinism.
    """
    print("\n" + "=" * 70)
    print("TASK 5: SHAPE & DETERMINISM CHECK")
    print("=" * 70)
    
    # Create test inputs
    B, N = 1, 4
    torch.manual_seed(100)
    tokens = torch.randn(B, N, D_TOKEN, dtype=torch.float32)
    valid = torch.tensor([[True, True, True, False]], dtype=torch.bool)
    
    torch.manual_seed(101)
    query = torch.randn(B, D_QUERY, dtype=torch.float32)
    query = query / query.norm(dim=-1, keepdim=True)  # L2 normalize
    
    human_token = HumanToken(tokens=tokens, valid=valid)
    query_emb = QueryEmbedding(embedding=query)
    
    # Run twice
    output_1 = adapter(human_token, query_emb)
    output_2 = adapter(human_token, query_emb)
    
    # Shape check
    print(f"\n  Input shapes:")
    print(f"    HumanToken.tokens: {tokens.shape}")
    print(f"    HumanToken.valid:  {valid.shape}")
    print(f"    QueryEmbedding:    {query.shape}")
    
    print(f"\n  Output shapes:")
    print(f"    GroundedHumanToken.tokens: {output_1.tokens.shape}")
    print(f"    GroundedHumanToken.valid:  {output_1.valid.shape}")
    
    shape_ok = (
        output_1.tokens.shape == tokens.shape and
        output_1.valid.shape == valid.shape
    )
    print(f"\n  Shape preservation: {'[PASS]' if shape_ok else '[FAIL]'}")
    
    # Determinism check
    max_diff = (output_1.tokens - output_2.tokens).abs().max().item()
    print(f"\n  Determinism Test (two runs, same input):")
    print(f"    Max absolute difference: {max_diff:.2e}")
    determinism_ok = max_diff < 1e-6
    print(f"    {'[PASS]' if determinism_ok else '[FAIL]'} Outputs are {'identical' if determinism_ok else 'different'}")
    
    # NaN/Inf check
    has_nan = torch.isnan(output_1.tokens).any().item()
    has_inf = torch.isinf(output_1.tokens).any().item()
    print(f"\n  Numerical stability:")
    print(f"    Contains NaN: {has_nan}")
    print(f"    Contains Inf: {has_inf}")
    print(f"    {'[PASS]' if not (has_nan or has_inf) else '[FAIL]'}")
    
    # Valid mask preserved
    valid_preserved = torch.equal(output_1.valid, valid)
    print(f"\n  Valid mask preserved: {'[PASS]' if valid_preserved else '[FAIL]'}")
    
    return max_diff, shape_ok and determinism_ok and not has_nan and not has_inf


# =============================================================================
# TASK 6: QUERY SENSITIVITY TEST
# =============================================================================

def task6_query_sensitivity(adapter: DynamicGroundingAdapter):
    """
    TASK 6: Verify different queries produce different outputs.
    """
    print("\n" + "=" * 70)
    print("TASK 6: QUERY SENSITIVITY TEST")
    print("=" * 70)
    
    # Create test human tokens
    B, N = 1, 3
    torch.manual_seed(200)
    tokens = torch.randn(B, N, D_TOKEN, dtype=torch.float32)
    valid = torch.ones(B, N, dtype=torch.bool)
    human_token = HumanToken(tokens=tokens, valid=valid)
    
    # Query A: "person on the left"
    torch.manual_seed(201)
    query_a = torch.randn(B, D_QUERY, dtype=torch.float32)
    query_a = query_a / query_a.norm(dim=-1, keepdim=True)
    query_emb_a = QueryEmbedding(embedding=query_a)
    
    # Query B: "person wearing red" (different seed = different semantics)
    torch.manual_seed(202)
    query_b = torch.randn(B, D_QUERY, dtype=torch.float32)
    query_b = query_b / query_b.norm(dim=-1, keepdim=True)
    query_emb_b = QueryEmbedding(embedding=query_b)
    
    # Run adapter with both queries
    output_a = adapter(human_token, query_emb_a)
    output_b = adapter(human_token, query_emb_b)
    
    # Gate values
    gate_a = adapter.get_gate_value(query_emb_a)
    gate_b = adapter.get_gate_value(query_emb_b)
    
    print(f"\n  Query A gate value: {gate_a:.4f}")
    print(f"  Query B gate value: {gate_b:.4f}")
    
    # Difference metrics
    token_diff = (output_a.tokens - output_b.tokens).abs()
    max_diff = token_diff.max().item()
    mean_diff = token_diff.mean().item()
    
    print(f"\n  Output difference (Query A vs Query B):")
    print(f"    Max absolute difference: {max_diff:.4f}")
    print(f"    Mean absolute difference: {mean_diff:.4f}")
    
    # Cosine similarity between outputs (per human)
    flat_a = output_a.tokens.view(N, -1)  # [N, D]
    flat_b = output_b.tokens.view(N, -1)  # [N, D]
    
    cos_sim = torch.nn.functional.cosine_similarity(flat_a, flat_b, dim=-1)
    
    print(f"\n  Cosine similarity (per human):")
    for i, sim in enumerate(cos_sim):
        print(f"    Human {i}: {sim.item():.4f}")
    
    mean_cos_sim = cos_sim.mean().item()
    print(f"    Mean: {mean_cos_sim:.4f}")
    
    # Check that outputs are different
    outputs_differ = max_diff > 1e-6
    print(f"\n  Outputs numerically different: {'[PASS]' if outputs_differ else '[FAIL]'}")
    
    # Check norms are bounded
    norm_a = output_a.tokens.norm(dim=-1)
    norm_b = output_b.tokens.norm(dim=-1)
    
    print(f"\n  Output norms (Query A): {norm_a.squeeze().tolist()}")
    print(f"  Output norms (Query B): {norm_b.squeeze().tolist()}")
    
    norms_bounded = (norm_a.max() < 100) and (norm_b.max() < 100)
    print(f"  Norms bounded: {'[PASS]' if norms_bounded else '[FAIL]'}")
    
    return mean_cos_sim, outputs_differ


# =============================================================================
# TASK 7: ZERO-HUMAN HANDLING
# =============================================================================

def task7_zero_human(adapter: DynamicGroundingAdapter):
    """
    TASK 7: Verify adapter handles zero humans gracefully.
    """
    print("\n" + "=" * 70)
    print("TASK 7: ZERO-HUMAN HANDLING")
    print("=" * 70)
    
    # Create empty human token
    B = 1
    N = 0  # Zero humans
    tokens = torch.zeros(B, N, D_TOKEN, dtype=torch.float32)
    valid = torch.zeros(B, N, dtype=torch.bool)
    human_token = HumanToken(tokens=tokens, valid=valid)
    
    # Create query
    torch.manual_seed(300)
    query = torch.randn(B, D_QUERY, dtype=torch.float32)
    query = query / query.norm(dim=-1, keepdim=True)
    query_emb = QueryEmbedding(embedding=query)
    
    print(f"\n  Input:")
    print(f"    HumanToken.tokens shape: {tokens.shape}")
    print(f"    HumanToken.valid shape:  {valid.shape}")
    print(f"    QueryEmbedding shape:    {query.shape}")
    
    # Run adapter
    try:
        output = adapter(human_token, query_emb)
        no_crash = True
    except Exception as e:
        print(f"\n  [FAIL] Adapter crashed: {e}")
        return False
    
    print(f"\n  Output:")
    print(f"    GroundedHumanToken.tokens shape: {output.tokens.shape}")
    print(f"    GroundedHumanToken.valid shape:  {output.valid.shape}")
    
    # Verify output is empty
    output_empty = output.tokens.shape[1] == 0
    valid_preserved = output.valid.shape[1] == 0
    
    print(f"\n  Zero-human handling:")
    print(f"    No crash: {'[PASS]' if no_crash else '[FAIL]'}")
    print(f"    Output empty (N=0): {'[PASS]' if output_empty else '[FAIL]'}")
    print(f"    Valid flag preserved: {'[PASS]' if valid_preserved else '[FAIL]'}")
    
    return no_crash and output_empty and valid_preserved


# =============================================================================
# TASK 8: ASSERTION CHECKS
# =============================================================================

def task8_assertions(adapter: DynamicGroundingAdapter):
    """
    TASK 8: Run assert_human_token on output.
    """
    print("\n" + "=" * 70)
    print("TASK 8: ASSERTION CHECKS")
    print("=" * 70)
    
    # Create valid input
    B, N = 2, 5
    torch.manual_seed(400)
    tokens = torch.randn(B, N, D_TOKEN, dtype=torch.float32)
    valid = torch.tensor([
        [True, True, True, True, False],
        [True, True, False, False, False]
    ], dtype=torch.bool)
    human_token = HumanToken(tokens=tokens, valid=valid)
    
    torch.manual_seed(401)
    query = torch.randn(B, D_QUERY, dtype=torch.float32)
    query = query / query.norm(dim=-1, keepdim=True)
    query_emb = QueryEmbedding(embedding=query)
    
    # Run adapter
    output = adapter(human_token, query_emb)
    
    print(f"\n  Running assert_human_token on GroundedHumanToken...")
    
    try:
        assert_human_token(output)
        print(f"  [PASS] assert_human_token PASSED")
        assertion_ok = True
    except (TypeError, ValueError) as e:
        print(f"  [FAIL] assert_human_token FAILED: {e}")
        assertion_ok = False
    
    print(f"\n  Additional checks:")
    print(f"    Output dtype: {output.tokens.dtype}")
    print(f"    Output device: {output.tokens.device}")
    print(f"    Contains NaN: {torch.isnan(output.tokens).any().item()}")
    print(f"    Contains Inf: {torch.isinf(output.tokens).any().item()}")
    
    return assertion_ok


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 70)
    print("PHASE 5: DYNAMIC GROUNDING ADAPTER")
    print("=" * 70)
    
    # Task 1: Contract reconfirmation
    task1_contract_reconfirmation()
    
    # Task 2: Fusion mechanism selection
    fusion_mechanism = task2_fusion_selection()
    
    # Tasks 3-4: Create adapter (includes gating)
    print("\n" + "=" * 70)
    print("TASKS 3-4: ADAPTER IMPLEMENTATION WITH GATING")
    print("=" * 70)
    
    adapter = DynamicGroundingAdapter(token_dim=D_TOKEN, query_dim=D_QUERY)
    
    print(f"\n  Adapter module summary:")
    print(f"    Fusion: FiLM-style modulation")
    print(f"    gamma_generator: Linear({D_QUERY} → {D_TOKEN})")
    print(f"    beta_generator:  Linear({D_QUERY} → {D_TOKEN})")
    print(f"    gate_generator:  Linear({D_QUERY} → 64) → Tanh → Linear(64 → 1) → Sigmoid")
    print(f"\n  Total parameters: {sum(p.numel() for p in adapter.parameters())}")
    print(f"  All parameters frozen (requires_grad=False)")
    
    # Sample gate value
    torch.manual_seed(999)
    sample_query = torch.randn(1, D_QUERY, dtype=torch.float32)
    sample_query = sample_query / sample_query.norm(dim=-1, keepdim=True)
    sample_query_emb = QueryEmbedding(embedding=sample_query)
    sample_gate = adapter.get_gate_value(sample_query_emb)
    print(f"\n  Sample gate value (random query): {sample_gate:.4f}")
    
    # Task 5: Shape & determinism
    max_diff, task5_ok = task5_shape_and_determinism(adapter)
    
    # Task 6: Query sensitivity
    mean_cos_sim, task6_ok = task6_query_sensitivity(adapter)
    
    # Task 7: Zero-human handling
    task7_ok = task7_zero_human(adapter)
    
    # Task 8: Assertions
    task8_ok = task8_assertions(adapter)
    
    # Final summary
    print("\n" + "=" * 70)
    print("PHASE 5 COMPLETE - SUMMARY")
    print("=" * 70)
    
    print(f"\n1. FUSION MECHANISM CHOSEN:")
    print(f"   {fusion_mechanism}")
    
    print(f"\n2. ADAPTER MODULE SUMMARY:")
    print(f"   - FiLM parameters: gamma, beta ∈ R^{D_TOKEN}")
    print(f"   - Gate: scalar ∈ (0, 1)")
    print(f"   - Formula: output = gate * (gamma * tokens + beta) + (1-gate) * tokens")
    
    print(f"\n3. GATE VALUE FOR SAMPLE QUERY:")
    print(f"   {sample_gate:.4f}")
    
    print(f"\n4. OUTPUT TOKEN SHAPE:")
    print(f"   [B, N, {D_TOKEN}] (unchanged from input)")
    
    print(f"\n5. DETERMINISM TEST RESULT:")
    print(f"   Max diff = {max_diff:.2e} {'[PASS]' if max_diff < 1e-6 else '[FAIL]'}")
    
    print(f"\n6. QUERY SENSITIVITY METRICS:")
    print(f"   Mean cosine similarity between Query A/B outputs: {mean_cos_sim:.4f}")
    print(f"   {'[PASS]' if task6_ok else '[FAIL]'} Outputs differ for different queries")
    
    print(f"\n7. ZERO-HUMAN TEST:")
    print(f"   {'[PASS]' if task7_ok else '[FAIL]'} Handles N=0 gracefully")
    
    print(f"\n8. ASSERTION CONFIRMATION:")
    print(f"   {'[PASS]' if task8_ok else '[FAIL]'} assert_human_token PASSED")
    
    all_pass = task5_ok and task6_ok and task7_ok and task8_ok
    print(f"\n" + "=" * 70)
    print(f"ALL TESTS: {'[PASS]' if all_pass else '[FAIL]'}")
    print("=" * 70)
    
    return adapter


if __name__ == "__main__":
    main()
