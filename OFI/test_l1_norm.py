#!/usr/bin/env python3
"""
Test L1 normalization for PCA components
"""

import numpy as np
import sys
import os
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from OFI import OrderbookFlowImbalance


class MockOrderBookData:
    """Mock OrderBookData for testing."""
    
    def __init__(self, n_samples, n_levels):
        self.numSnapshots = n_samples
        self.numLevels = n_levels
        
        # Generate timestamps with 100ms intervals
        start_dt = datetime(2024, 1, 1, 10, 30, 0, tzinfo=timezone.utc)
        start_time_ms = int(start_dt.timestamp() * 1000)
        self.timestamps = start_time_ms + np.arange(n_samples, dtype=np.int64) * 100
        
        # Generate synthetic data
        np.random.seed(42)
        self.buyPrices = np.zeros((n_samples, n_levels))
        self.sellPrices = np.zeros((n_samples, n_levels))
        self.buyVolumes = np.zeros((n_samples, n_levels))
        self.sellVolumes = np.zeros((n_samples, n_levels))
        
        # Create patterns
        trend = np.sin(np.linspace(0, 4*np.pi, n_samples)) * 5
        
        for i in range(n_samples):
            for j in range(n_levels):
                self.buyPrices[i, j] = 100.0 + trend[i] - j * 0.1
                self.sellPrices[i, j] = 100.0 + trend[i] + 0.1 + j * 0.1
                self.buyVolumes[i, j] = np.exp(-j * 0.3) * (1 + 0.2 * np.random.rand())
                self.sellVolumes[i, j] = np.exp(-j * 0.3) * (1 + 0.2 * np.random.rand())
        
        self.price_tick = 0.1
        self.volume_tick = 0.001


def test_l1_normalization():
    """Test that PCA components are L1 normalized."""
    print("Testing L1 Normalization of PCA Components")
    print("=" * 60)
    
    # Create data
    n_samples = 200
    n_levels = 5
    mock_data = MockOrderBookData(n_samples, n_levels)
    
    # Create OFI with integrated OFI enabled
    ofi = OrderbookFlowImbalance(
        mock_data,
        mode='event_driven',
        lookback_interval=10,
        train_interval=50,  # 4 groups total
        use_numba=False,
        is_integrated_ofi=True
    )
    
    print(f"\nNumber of groups: {ofi.num_groups}")
    print(f"Number of levels: {n_levels}")
    
    # Check L1 norm of each principal component
    print("\nL1 Norm of Principal Components:")
    print("-" * 40)
    
    all_norms_correct = True
    for i in range(ofi.num_groups):
        pc1 = ofi.pca_w1[i]
        l1_norm = np.sum(np.abs(pc1))
        l2_norm = np.linalg.norm(pc1)
        
        print(f"Group {i}:")
        print(f"  PC1: {pc1}")
        print(f"  L1 norm: {l1_norm:.6f} (should be 1.0)")
        print(f"  L2 norm: {l2_norm:.6f}")
        
        # Check if L1 norm is approximately 1
        if abs(l1_norm - 1.0) > 1e-6:
            all_norms_correct = False
            print(f"  ❌ L1 norm is not 1.0!")
        else:
            print(f"  ✓ L1 norm is correct")
        print()
    
    # Test integrated OFI computation with L1 normalized components
    print("\nIntegrated OFI Statistics:")
    print("-" * 40)
    
    out_ofi = ofi.integrated_ofi_outsample
    in_ofi = ofi.integrated_ofi_insample
    
    print(f"Out-of-sample range: [{np.min(out_ofi):.3f}, {np.max(out_ofi):.3f}]")
    print(f"In-sample range: [{np.min(in_ofi):.3f}, {np.max(in_ofi):.3f}]")
    
    # Compare with manual calculation for verification
    print("\nManual Verification (Group 1, Sample 0):")
    group_id = 1
    sample_idx = 50  # First sample of group 1
    
    # Get OFI vector
    ofi_vec, _ = ofi.get_ofi_at(sample_idx)
    
    # Manual calculation with previous group's PC (out-of-sample)
    prev_pc1 = ofi.pca_w1[group_id - 1]
    manual_out = np.dot(prev_pc1, ofi_vec)
    actual_out, _ = ofi.get_integrated_ofi_outsample_at(sample_idx)
    
    print(f"Manual out-of-sample: {manual_out:.6f}")
    print(f"Actual out-of-sample: {actual_out:.6f}")
    print(f"Match: {abs(manual_out - actual_out) < 1e-6}")
    
    # Manual calculation with current group's PC (in-sample)
    curr_pc1 = ofi.pca_w1[group_id]
    manual_in = np.dot(curr_pc1, ofi_vec)
    actual_in, _ = ofi.get_integrated_ofi_insample_at(sample_idx)
    
    print(f"\nManual in-sample: {manual_in:.6f}")
    print(f"Actual in-sample: {actual_in:.6f}")
    print(f"Match: {abs(manual_in - actual_in) < 1e-6}")
    
    return all_norms_correct


if __name__ == "__main__":
    all_correct = test_l1_normalization()
    if all_correct:
        print("\n✅ All PCA components are correctly L1 normalized!")
    else:
        print("\n❌ Some PCA components are not L1 normalized!")