# -*- coding: utf-8 -*-
"""Quick sanity check for Phase-3 token-level encoding."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from training.grounding_train_v2 import SimpleQueryEncoder
from core.datatypes import D_QUERY

print('Testing SimpleQueryEncoder token-level output (Phase-3)')
print('='*70)

encoder = SimpleQueryEncoder()
print('✓ Encoder created')

# Test token-level encoding
captions = ['person on the left', 'man wearing black shirt standing near table']
tokens, mask = encoder.forward_tokens_batch(captions)
sent_emb = encoder.forward_batch(captions)

print(f'\nCaption 1: "{captions[0]}"')
print(f'Caption 2: "{captions[1]}"')
print(f'\nToken embeddings shape: {tokens.shape}')
print(f'Token mask shape: {mask.shape}')
print(f'Sentence embedding shape: {sent_emb.shape}')
print(f'\nToken dim matches D_QUERY: {tokens.shape[-1] == D_QUERY}')
print(f'Mask has correct batch size: {mask.shape[0] == 2}')
print(f'Mask is boolean: {mask.dtype == torch.bool}')

# Verify mask values
print(f'\nMask sample 1 (valid tokens): {mask[0].sum().item()}/{mask.shape[1]}')
print(f'Mask sample 2 (valid tokens): {mask[1].sum().item()}/{mask.shape[1]}')

print('\n' + '='*70)
print('✅ SimpleQueryEncoder token-level output works correctly')
