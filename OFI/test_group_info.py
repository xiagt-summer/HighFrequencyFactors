#!/usr/bin/env python3
"""
Test get_group_info function
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
        self.buyPrices = np.random.randn(n_samples, n_levels) + 100
        self.sellPrices = self.buyPrices + 0.1
        self.buyVolumes = np.random.exponential(1.0, (n_samples, n_levels))
        self.sellVolumes = np.random.exponential(1.0, (n_samples, n_levels))
        
        self.price_tick = 0.1
        self.volume_tick = 0.001


def test_group_info():
    """Test get_group_info function."""
    print("Testing get_group_info Function")
    print("=" * 60)
    
    # Test 1: Event-driven mode
    print("\n1. Event-Driven Mode:")
    print("-" * 40)
    
    n_samples = 200
    mock_data = MockOrderBookData(n_samples, 5)
    
    ofi_event = OrderbookFlowImbalance(
        mock_data,
        mode='event_driven',
        lookback_interval=10,
        train_interval=50,  # 50 samples per group
        use_numba=False
    )
    
    print(f"Total samples: {n_samples}")
    print(f"Number of groups: {ofi_event.num_groups}")
    
    # Get info for each group
    for i in range(ofi_event.num_groups):
        left_index, right_index = ofi_event.get_group_info(i)
        print(f"\nGroup {i}:")
        print(f"  Sample indices: [{left_index}, {right_index}]")
        if left_index >= 0:
            print(f"  Timestamps: [{ofi_event._timestamps[left_index]}, {ofi_event._timestamps[right_index]}]")
            print(f"  Number of samples: {right_index - left_index + 1}")
        else:
            print(f"  Empty group")
    
    # Test 2: Time-driven mode
    print("\n\n2. Time-Driven Mode:")
    print("-" * 40)
    
    # Create data spanning 1 hour
    n_samples = 3600  # 1 hour at 100ms intervals
    mock_data_time = MockOrderBookData(n_samples, 5)
    
    ofi_time = OrderbookFlowImbalance(
        mock_data_time,
        mode='time_driven',
        lookback_interval=1000,    # 1 second lookback
        train_interval=300000,     # 5 minute groups
        use_numba=False
    )
    
    print(f"Total samples: {n_samples}")
    print(f"Number of groups: {ofi_time.num_groups}")
    
    # Show first 3 groups
    for i in range(min(3, ofi_time.num_groups)):
        left_index, right_index = ofi_time.get_group_info(i)
        
        if left_index >= 0:
            # Convert timestamps to readable format
            left_ts = ofi_time._timestamps[left_index]
            right_ts = ofi_time._timestamps[right_index]
            left_dt = datetime.fromtimestamp(left_ts / 1000, tz=timezone.utc)
            right_dt = datetime.fromtimestamp(right_ts / 1000, tz=timezone.utc)
            
            print(f"\nGroup {i}:")
            print(f"  Sample indices: [{left_index}, {right_index}]")
            print(f"  Timestamps: [{left_ts}, {right_ts}]")
            print(f"  Timestamp times: [{left_dt.strftime('%H:%M:%S')}, {right_dt.strftime('%H:%M:%S')}]")
            print(f"  Number of samples: {right_index - left_index + 1}")
        else:
            print(f"\nGroup {i}: Empty group")
    
    # Test 3: Accessing specific group info
    print("\n\n3. Accessing Specific Group:")
    print("-" * 40)
    
    group_id = 1
    left_index, right_index = ofi_event.get_group_info(group_id)
    
    print(f"Group {group_id} info:")
    print(f"  Left index: {left_index}")
    print(f"  Right index: {right_index}")
    if left_index >= 0:
        print(f"  Number of samples: {right_index - left_index + 1}")
        print(f"  First timestamp: {ofi_event._timestamps[left_index]}")
        print(f"  Last timestamp: {ofi_event._timestamps[right_index]}")


if __name__ == "__main__":
    test_group_info()
    print("\n✅ get_group_info test completed!")