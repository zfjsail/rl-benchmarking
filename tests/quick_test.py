#!/usr/bin/env python
"""
Quick test to verify SGLangRolloutWithLogit can be imported and basic functions work.

Usage:
    python tests/quick_test.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all necessary imports work"""
    print("Testing imports...")
    
    try:
        from verl.workers.rollout.sglang_rollout.sglang_rollout_with_logit import (
            SGLangRolloutWithLogit,
            _extract_logits_from_output,
        )
        print("✓ Successfully imported SGLangRolloutWithLogit")
        print("✓ Successfully imported _extract_logits_from_output")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    return True


def test_mock_output():
    """Test logit extraction with mock data"""
    print("\nTesting logit extraction with mock data...")
    
    try:
        import numpy as np
        import torch
        from verl.workers.rollout.sglang_rollout.sglang_rollout_with_logit import (
            _extract_logits_from_output,
        )
        
        vocab_size = 128256
        seq_len = 5
        
        # Create mock output
        output_token_logprobs = []
        for i in range(seq_len):
            log_prob = np.random.uniform(-10, 0)
            token_id = np.random.randint(0, vocab_size)
            logits = np.random.normal(0, 1, vocab_size).astype(np.float32)
            output_token_logprobs.append((log_prob, token_id, logits))
        
        mock_output = {
            "text": "Test response",
            "output_ids": list(range(100, 100 + seq_len)),
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": output_token_logprobs,
            },
        }
        
        # Test extraction
        output_token_ids, logits = _extract_logits_from_output(mock_output)
        
        if logits is not None:
            print(f"✓ Successfully extracted logits")
            print(f"  - Shape: {logits.shape}")
            print(f"  - Expected: ({seq_len}, {vocab_size})")
            
            if logits.shape == (seq_len, vocab_size):
                print("✓ Shape is correct!")
                return True
            else:
                print(f"✗ Shape mismatch!")
                return False
        else:
            print(f"✗ Logits extraction returned None")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_registry():
    """Test that rollout can be registered"""
    print("\nTesting rollout registration...")
    
    try:
        from verl.workers.rollout.base import _ROLLOUT_REGISTRY
        
        # Check if sglang_with_logit is registered
        key = ("sglang_with_logit", "async")
        
        if key in _ROLLOUT_REGISTRY:
            print(f"✓ Rollout class is registered in registry")
            print(f"  - Path: {_ROLLOUT_REGISTRY[key]}")
            return True
        else:
            print(f"✗ Rollout class NOT registered in registry")
            print(f"  - Available entries: {list(_ROLLOUT_REGISTRY.keys())}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all quick tests"""
    print("="*70)
    print("Quick Test for SGLangRolloutWithLogit")
    print("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("Mock Output", test_mock_output),
        ("Registry", test_registry),
    ]
    
    results = {}
    for test_name, test_func in tests:
        result = test_func()
        results[test_name] = result
    
    # Summary
    print("\n" + "="*70)
    print("Quick Test Summary")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*70)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

