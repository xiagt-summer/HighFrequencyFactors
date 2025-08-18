#!/usr/bin/env python3
"""
OrderbookFlowImbalance - Optimized Version
==========================================

High-performance Order Flow Imbalance (OFI) calculation for orderbook data.
Optimized with vectorized operations, pre-computed boundaries, and optional numba JIT.

Classes
-------
OrderbookFlowImbalance : Compute OFI from OrderBookData
    Calculates per-level OFI values over time windows with optimized performance.
"""

import numpy as np
from typing import Literal, Optional, Tuple
from datetime import datetime, timezone

# Import OrderBookData from the user's codebase
try:
    from OrderBookDS.OrderBookDS import OrderBookData
except ImportError:
    try:
        from OrderBookDS import OrderBookData
    except ImportError as e:
        raise ImportError(
            "Failed to import OrderBookData. Ensure OrderBookDS/OrderBookDS.py is on sys.path."
        ) from e

# Optional numba import for JIT compilation
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Define no-op decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


@jit(nopython=True, parallel=True, cache=True)
def _compute_ofi_vectorized_numba(
    buy_prices: np.ndarray,
    buy_volumes: np.ndarray, 
    sell_prices: np.ndarray,
    sell_volumes: np.ndarray,
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    eps: float = 1e-12
) -> np.ndarray:
    """
    Numba-optimized OFI computation for all windows.
    
    Parameters
    ----------
    buy_prices, buy_volumes, sell_prices, sell_volumes : ndarray
        Price and volume data, shape (n_samples, n_levels)
    window_starts, window_ends : ndarray
        Start and end indices for each window
    eps : float
        Small value to avoid division by zero
        
    Returns
    -------
    ofi : ndarray
        OFI values, shape (n_windows, n_levels)
    """
    n_windows = len(window_starts)
    n_levels = buy_prices.shape[1]
    ofi_result = np.zeros((n_windows, n_levels), dtype=np.float32)
    
    for i in prange(n_windows):
        start = window_starts[i]
        end = window_ends[i]
        
        if end - start < 2:
            continue
            
        # Calculate price changes
        dpb = buy_prices[start+1:end] - buy_prices[start:end-1]
        dps = sell_prices[start+1:end] - sell_prices[start:end-1]
        
        # Calculate OFI components
        ofi_buy_pos = np.sum(buy_volumes[start+1:end] * (dpb >= 0), axis=0)
        ofi_buy_neg = np.sum(buy_volumes[start:end-1] * (dpb <= 0), axis=0)
        ofi_sell_neg = np.sum(sell_volumes[start+1:end] * (dps <= 0), axis=0)
        ofi_sell_pos = np.sum(sell_volumes[start:end-1] * (dps >= 0), axis=0)
        
        ofi = ofi_buy_pos - ofi_buy_neg - ofi_sell_neg + ofi_sell_pos
        
        # Normalization
        total_volume = np.sum(buy_volumes[start:end]) + np.sum(sell_volumes[start:end])
        Q = total_volume / (2.0 * n_levels)
        
        if Q > eps:
            ofi_result[i] = ofi / Q
            
    return ofi_result


def _compute_ofi_vectorized_numpy(
    buy_prices: np.ndarray,
    buy_volumes: np.ndarray,
    sell_prices: np.ndarray,
    sell_volumes: np.ndarray,
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    eps: float = 1e-12
) -> np.ndarray:
    """
    Numpy-vectorized OFI computation for all windows.
    """
    n_windows = len(window_starts)
    n_levels = buy_prices.shape[1]
    ofi_result = np.zeros((n_windows, n_levels), dtype=np.float32)
    
    # Process in batches for better cache efficiency
    batch_size = min(1000, n_windows)
    
    for batch_start in range(0, n_windows, batch_size):
        batch_end = min(batch_start + batch_size, n_windows)
        
        for i in range(batch_start, batch_end):
            start = window_starts[i]
            end = window_ends[i]
            
            if end - start < 2:
                continue
            
            # Slice once and reuse
            buy_p_window = buy_prices[start:end]
            buy_v_window = buy_volumes[start:end]
            sell_p_window = sell_prices[start:end]
            sell_v_window = sell_volumes[start:end]
            
            # Calculate price changes
            dpb = buy_p_window[1:] - buy_p_window[:-1]
            dps = sell_p_window[1:] - sell_p_window[:-1]
            
            # Vectorized OFI calculation
            ofi = (
                np.sum(buy_v_window[1:] * (dpb >= 0), axis=0) -
                np.sum(buy_v_window[:-1] * (dpb <= 0), axis=0) -
                np.sum(sell_v_window[1:] * (dps <= 0), axis=0) +
                np.sum(sell_v_window[:-1] * (dps >= 0), axis=0)
            )
            
            # Normalization
            total_volume = np.sum(buy_v_window) + np.sum(sell_v_window)
            Q = total_volume / (2.0 * n_levels)
            
            if Q > eps:
                ofi_result[i] = ofi / Q
                
    return ofi_result


class OrderbookFlowImbalance:
    """
    Optimized Order Flow Imbalance (OFI) calculator with performance enhancements.
    
    Parameters
    ----------
    obData : OrderBookData
        Input orderbook data with timestamps, prices, and volumes.
    mode : {'time_driven', 'event_driven'}
        Computation mode:
        - 'time_driven': Use time-based windows (lookback_interval in milliseconds)
        - 'event_driven': Use snapshot count windows (lookback_interval in number of snapshots)
    lookback_interval : int
        Window size: milliseconds for time_driven mode, snapshot count for event_driven mode.
    train_interval : int, optional
        Grouping interval for OFI vectors:
        - time_driven: milliseconds for time-based grouping
        - event_driven: number of samples per group
    dtype : np.dtype, optional
        Data type for OFI calculations. Default is np.float32.
    use_numba : bool, optional
        Whether to use numba JIT compilation if available. Default is True.
    
    Attributes
    ----------
    ofi_vectors : ndarray
        Computed OFI values, shape (num_samples, num_levels).
    num_samples : int
        Number of samples in the input data.
    
    Examples
    --------
    >>> ob = OrderBookData('BTCUSDT.csv.gz', numLevels=10)
    >>> ofi = OrderbookFlowImbalance(ob, mode='time_driven', lookback_interval=1000)
    >>> ofi_values = ofi.ofi_vectors
    """
    
    def __init__(self, 
                 obData: OrderBookData,
                 mode: Literal['time_driven', 'event_driven'] = 'time_driven',
                 lookback_interval: int = 1000,
                 train_interval: Optional[int] = None,
                 dtype: np.dtype = np.float32,
                 use_numba: bool = True):
        """Initialize OFI computation with optimizations."""
        
        # Validate mode
        if mode not in ['time_driven', 'event_driven']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'time_driven' or 'event_driven'")
        
        # Store parameters
        self._obData = obData
        self._mode = mode
        self._lookback_interval = lookback_interval  # milliseconds or snapshots
        self._train_interval = train_interval  # milliseconds or snapshots for grouping
        self._dtype = dtype
        self._use_numba = use_numba and HAS_NUMBA
        
        # Extract data from OrderBookData - ensure contiguous arrays for performance
        self._timestamps = np.ascontiguousarray(obData.timestamps, dtype=np.int64)
        self._buy_prices = np.ascontiguousarray(obData.buyPrices, dtype=dtype)
        self._buy_volumes = np.ascontiguousarray(obData.buyVolumes, dtype=dtype)
        self._sell_prices = np.ascontiguousarray(obData.sellPrices, dtype=dtype)
        self._sell_volumes = np.ascontiguousarray(obData.sellVolumes, dtype=dtype)
        self._num_samples = obData.numSnapshots
        self._num_levels = obData.numLevels
        
        # Pre-allocate OFI storage
        self._ofi = np.zeros((self._num_samples, self._num_levels), dtype=self._dtype)
        
        # Track sampled indices for event-driven mode
        self._sampled_indices = np.array([], dtype=np.int64)
        
        # Pre-compute window boundaries for efficiency
        self._window_starts = None
        self._window_ends = None
        
        # Grouping information
        self._groups = None
        self._group_boundaries = None
        self._grid_start_time = None
        self._grid_end_time = None
        
        # Compute OFI
        self._compute_ofi()
        
        # Compute grouping if train_interval is specified
        if self._train_interval is not None:
            self._compute_groups()
        
        # Log summary
        unit = "ms" if mode == 'time_driven' else "snapshots"
        numba_status = "enabled" if self._use_numba else "disabled"
        print(f"OFI: mode={mode}, L={self._num_samples}, n={self._num_levels}, "
              f"lookback_interval={lookback_interval}{unit}, numba={numba_status}")
    
    def _compute_ofi(self):
        """Main OFI computation based on mode."""
        if self._mode == 'time_driven':
            self._compute_ofi_time_driven_optimized()
        elif self._mode == 'event_driven':
            self._compute_ofi_event_driven_optimized()
        else:
            raise ValueError(f"Unsupported mode: {self._mode}")
    
    def _compute_ofi_time_driven_optimized(self):
        """
        Optimized time-driven OFI computation with pre-computed boundaries.
        """
        # Pre-compute all window boundaries at once
        window_indices = np.arange(self._num_samples)
        window_start_times = self._timestamps - self._lookback_interval
        
        # Vectorized searchsorted for all windows
        self._window_starts = np.searchsorted(self._timestamps, window_start_times, side='right')
        self._window_ends = window_indices + 1
        
        # Choose computation method
        if self._use_numba:
            ofi_values = _compute_ofi_vectorized_numba(
                self._buy_prices, self._buy_volumes,
                self._sell_prices, self._sell_volumes,
                self._window_starts, self._window_ends
            )
        else:
            ofi_values = _compute_ofi_vectorized_numpy(
                self._buy_prices, self._buy_volumes,
                self._sell_prices, self._sell_volumes,
                self._window_starts, self._window_ends
            )
        
        self._ofi = ofi_values
    
    def _compute_ofi_event_driven_optimized(self):
        """
        Optimized event-driven OFI computation.
        """
        K = self._lookback_interval
        
        # Calculate sampled indices: K-1, 2K-1, 3K-1, ...
        self._sampled_indices = np.arange(K - 1, self._num_samples, K)
        
        if len(self._sampled_indices) == 0:
            return
        
        # Pre-compute window boundaries
        self._window_starts = np.maximum(0, self._sampled_indices - K + 1)
        self._window_ends = self._sampled_indices + 1
        
        # Compute OFI for sampled windows
        if self._use_numba:
            sampled_ofi = _compute_ofi_vectorized_numba(
                self._buy_prices, self._buy_volumes,
                self._sell_prices, self._sell_volumes,
                self._window_starts, self._window_ends
            )
        else:
            sampled_ofi = _compute_ofi_vectorized_numpy(
                self._buy_prices, self._buy_volumes,
                self._sell_prices, self._sell_volumes,
                self._window_starts, self._window_ends
            )
        
        # Place results at sampled indices
        self._ofi[self._sampled_indices] = sampled_ofi
    
    def _process_time_boundaries(self):
        """
        Process start and end times to align with day boundaries.
        
        Returns
        -------
        start_time : int
            Start of the earliest day at 00:00:00.000 UTC
        end_time : int
            Start of the day after the latest timestamp at 00:00:00.000 UTC
        """
        # Convert timestamps to datetime for processing
        min_ts = self._timestamps.min()
        max_ts = self._timestamps.max()
        
        # Convert to datetime (assuming milliseconds)
        min_dt = datetime.fromtimestamp(min_ts / 1000, tz=timezone.utc)
        max_dt = datetime.fromtimestamp(max_ts / 1000, tz=timezone.utc)
        
        # Get start of earliest day
        start_dt = min_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        start_time = int(start_dt.timestamp() * 1000)
        
        # Get start of day after latest timestamp
        end_dt = max_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        if end_dt <= max_dt:
            # Move to next day
            from datetime import timedelta
            end_dt = end_dt + timedelta(days=1)
        end_time = int(end_dt.timestamp() * 1000)
        
        return start_time, end_time
    
    @property
    def ofi_vectors(self) -> np.ndarray:
        """Return computed OFI vectors."""
        return self._ofi
    
    @property
    def num_samples(self) -> int:
        """Return number of samples."""
        return self._num_samples
    
    @property
    def sampled_indices(self) -> np.ndarray:
        """Return sampled indices where OFI is computed in event-driven mode."""
        return self._sampled_indices
    
    def get_ofi_at(self, i: int) -> Tuple[np.ndarray, int]:
        """
        Get OFI vector and timestamp at index i.
        
        Parameters
        ----------
        i : int
            Index of the sample.
            
        Returns
        -------
        ofi_vector : ndarray
            OFI values at index i, shape (num_levels,).
        timestamp : int
            Timestamp in milliseconds at index i.
        """
        if i < 0 or i >= self._num_samples:
            raise IndexError(f"Index {i} out of range [0, {self._num_samples})")
        
        return self._ofi[i], self._timestamps[i]
    
    def recompute(self, 
                  mode: Optional[str] = None,
                  lookback_interval: Optional[int] = None) -> None:
        """
        Re-run the OFI computation with possibly updated parameters.
        
        Parameters
        ----------
        mode : str, optional
            New computation mode.
        lookback_interval : int, optional
            New lookback interval in milliseconds.
        """
        # Update parameters if provided
        if mode is not None:
            self._mode = mode
        
        if lookback_interval is not None:
            self._lookback_interval = lookback_interval
        
        # Reset OFI array and sampled indices
        self._ofi = np.zeros((self._num_samples, self._num_levels), dtype=self._dtype)
        self._sampled_indices = np.array([], dtype=np.int64)
        self._window_starts = None
        self._window_ends = None
        self._groups = None
        self._group_boundaries = None
        self._grid_start_time = None
        self._grid_end_time = None
        
        # Recompute
        self._compute_ofi()
        
        # Recompute grouping if train_interval is specified
        if self._train_interval is not None:
            self._compute_groups()
        
        # Log summary
        unit = "ms" if self._mode == 'time_driven' else "snapshots"
        print(f"OFI recomputed: mode={self._mode}, L={self._num_samples}, "
              f"n={self._num_levels}, lookback_interval={self._lookback_interval}{unit}")
    
    def _compute_groups(self):
        """
        Compute grouping of OFI vectors based on train_interval.
        
        For time_driven mode:
        - Aligns to day boundaries (00:00:00.000)
        - Creates time grid with train_interval spacing
        
        For event_driven mode:
        - Groups by fixed number of samples
        """
        if self._mode == 'time_driven':
            self._compute_time_driven_groups()
        else:
            self._compute_event_driven_groups()
    
    def _compute_time_driven_groups(self):
        """
        Compute time-based grouping aligned to day boundaries.
        """
        # Get day-aligned boundaries
        self._grid_start_time, self._grid_end_time = self._process_time_boundaries()
        
        # Create time grid
        grid_times = np.arange(
            self._grid_start_time,
            self._grid_end_time + self._train_interval,
            self._train_interval,
            dtype=np.int64
        )
        
        # Assign each sample to a group
        self._groups = np.searchsorted(grid_times, self._timestamps, side='right') - 1
        self._group_boundaries = grid_times
        
        # Log grouping info
        n_groups = len(grid_times) - 1
        print(f"Time-driven grouping: {n_groups} groups, "
              f"grid from {self._grid_start_time} to {self._grid_end_time}, "
              f"interval={self._train_interval}ms")
    
    def _compute_event_driven_groups(self):
        """
        Compute event-based grouping by sample count.
        """
        # Group by fixed number of samples
        self._groups = np.arange(self._num_samples) // self._train_interval
        
        # Compute group boundaries (sample indices)
        n_groups = self._groups[-1] + 1 if self._num_samples > 0 else 0
        self._group_boundaries = np.arange(n_groups + 1) * self._train_interval
        
        # Log grouping info
        print(f"Event-driven grouping: {n_groups} groups, "
              f"{self._train_interval} samples per group")
    
    def get_group_by_timestamp(self, timestamp: int) -> Optional[int]:
        """
        Get the group number for a given timestamp.
        
        Parameters
        ----------
        timestamp : int
            Timestamp in milliseconds
            
        Returns
        -------
        group : int or None
            Group number (0-indexed), or None if timestamp is out of range
        """
        if self._groups is None:
            raise ValueError("No grouping computed. Set train_interval to enable grouping.")
        
        if self._mode == 'time_driven':
            # For time-driven, use the time grid
            if timestamp < self._grid_start_time or timestamp >= self._grid_end_time:
                return None
            group = (timestamp - self._grid_start_time) // self._train_interval
            return int(group)
        else:
            # For event-driven, find the sample index first
            idx = np.searchsorted(self._timestamps, timestamp, side='left')
            if idx >= self._num_samples:
                return None
            # If exact match or within tolerance
            if idx < self._num_samples and abs(self._timestamps[idx] - timestamp) < 1:
                return int(self._groups[idx])
            return None
    
    def get_group_ofi(self, group_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all OFI vectors and timestamps for a specific group.
        
        Parameters
        ----------
        group_id : int
            Group number (0-indexed)
            
        Returns
        -------
        ofi_vectors : ndarray
            OFI values for the group, shape (n_samples_in_group, n_levels)
        timestamps : ndarray
            Timestamps for the group
        """
        if self._groups is None:
            raise ValueError("No grouping computed. Set train_interval to enable grouping.")
        
        # Find samples in this group
        mask = self._groups == group_id
        group_indices = np.where(mask)[0]
        
        if len(group_indices) == 0:
            return np.array([]), np.array([])
        
        return self._ofi[group_indices], self._timestamps[group_indices]
    
    @property
    def groups(self) -> Optional[np.ndarray]:
        """Return group assignments for each sample."""
        return self._groups
    
    @property
    def group_boundaries(self) -> Optional[np.ndarray]:
        """Return group boundaries (time points or sample indices)."""
        return self._group_boundaries
    
    @property
    def num_groups(self) -> Optional[int]:
        """Return total number of groups."""
        if self._groups is None:
            return None
        return int(np.max(self._groups) + 1) if len(self._groups) > 0 else 0