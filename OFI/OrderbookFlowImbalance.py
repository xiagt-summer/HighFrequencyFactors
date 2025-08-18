#!/usr/bin/env python3
"""
OrderbookFlowImbalance (Optimized Version)
==========================================

High-performance Order Flow Imbalance (OFI) calculation for orderbook data.
Optimized for speed with vectorized operations and optional numba acceleration.

Classes
-------
OrderbookFlowImbalance : Compute OFI from OrderBookData
    Calculates per-level OFI values over time windows.
"""

import numpy as np
from typing import Literal, Optional, Tuple
from datetime import datetime, timezone

# Try to import numba for JIT compilation
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Define dummy decorators if numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

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


@jit(nopython=True, cache=True, parallel=True)
def _compute_window_ofi_vectorized(
    buy_prices: np.ndarray,
    buy_volumes: np.ndarray, 
    sell_prices: np.ndarray,
    sell_volumes: np.ndarray,
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    out_ofi: np.ndarray,
    eps: float = 1e-12
) -> None:
    """
    Vectorized OFI computation for multiple windows using numba.
    
    Parameters
    ----------
    buy_prices, buy_volumes, sell_prices, sell_volumes : ndarray
        Full data arrays
    window_starts, window_ends : ndarray
        Start and end indices for each window
    out_ofi : ndarray
        Output array to store OFI values (modified in-place)
    eps : float
        Small value to avoid division by zero
    """
    n_windows = len(window_starts)
    n_levels = buy_prices.shape[1]
    
    for w in prange(n_windows):
        start_idx = window_starts[w]
        end_idx = window_ends[w]
        
        if end_idx - start_idx < 2:
            # Not enough data points
            for j in range(n_levels):
                out_ofi[w, j] = 0.0
            continue
        
        # Calculate OFI for each level
        for j in range(n_levels):
            ofi_sum = 0.0
            total_volume = 0.0
            
            # Process each consecutive pair in the window
            for i in range(start_idx, end_idx - 1):
                # Price changes
                dpb = buy_prices[i + 1, j] - buy_prices[i, j]
                dps = sell_prices[i + 1, j] - sell_prices[i, j]
                
                # Buy side contributions
                if dpb >= 0:
                    ofi_sum += buy_volumes[i + 1, j]
                if dpb <= 0:
                    ofi_sum -= buy_volumes[i, j]
                
                # Sell side contributions
                if dps <= 0:
                    ofi_sum -= sell_volumes[i + 1, j]
                if dps >= 0:
                    ofi_sum += sell_volumes[i, j]
            
            # Calculate total volume for normalization
            for i in range(start_idx, end_idx):
                total_volume += buy_volumes[i, j] + sell_volumes[i, j]
            
            # Normalize
            Q = total_volume / (2.0 * n_levels)
            if Q > eps:
                out_ofi[w, j] = ofi_sum / Q
            else:
                out_ofi[w, j] = 0.0


@jit(nopython=True, cache=True)
def _precompute_window_boundaries_time_driven(
    timestamps: np.ndarray,
    lookback_interval: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute window start indices for all timestamps.
    
    Parameters
    ----------
    timestamps : ndarray
        Array of timestamps
    lookback_interval : int
        Lookback window in milliseconds
        
    Returns
    -------
    window_starts : ndarray
        Start indices for each window
    window_ends : ndarray
        End indices for each window (exclusive)
    """
    n_samples = len(timestamps)
    window_starts = np.empty(n_samples, dtype=np.int64)
    window_ends = np.arange(1, n_samples + 1, dtype=np.int64)
    
    # Use two-pointer technique for O(n) complexity
    start_idx = 0
    
    for i in range(n_samples):
        window_start_time = timestamps[i] - lookback_interval
        
        # Move start_idx forward until we find the window start
        while start_idx < i and timestamps[start_idx] < window_start_time:
            start_idx += 1
        
        window_starts[i] = start_idx
    
    return window_starts, window_ends


class OrderbookFlowImbalance:
    """
    Compute per-level Order Flow Imbalance (OFI) vectors from orderbook data.
    
    Optimized version with vectorized operations and optional numba acceleration.
    
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
        Reserved for future use. Same units as lookback_interval.
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
                 round_volume_down: bool = True,
                 use_numba: bool = True):
        """Initialize OFI computation."""
        
        # Validate mode
        if mode not in ['time_driven', 'event_driven']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'time_driven' or 'event_driven'")
        
        # Store parameters
        self._obData = obData
        self._mode = mode
        self._lookback_interval = lookback_interval  # milliseconds or snapshots
        self._train_interval = train_interval  # reserved for future use
        self._dtype = dtype
        self._round_volume_down = round_volume_down
        self._use_numba = use_numba and HAS_NUMBA
        
        # Extract data from OrderBookData - convert to contiguous arrays for better performance
        self._timestamps = np.ascontiguousarray(obData.timestamps, dtype=np.int64)
        self._buy_prices = np.ascontiguousarray(obData.buyPrices, dtype=self._dtype)
        self._buy_volumes = np.ascontiguousarray(obData.buyVolumes, dtype=self._dtype)
        self._sell_prices = np.ascontiguousarray(obData.sellPrices, dtype=self._dtype)
        self._sell_volumes = np.ascontiguousarray(obData.sellVolumes, dtype=self._dtype)
        self._num_samples = obData.numSnapshots
        self._num_levels = obData.numLevels
        
        # Initialize OFI storage
        self._ofi = np.zeros((self._num_samples, self._num_levels), dtype=self._dtype)
        
        # Track sampled indices for event-driven mode
        self._sampled_indices = np.array([], dtype=np.int64)
        
        # Compute OFI
        self._compute_ofi()
        
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
        Optimized OFI computation for time-driven mode.
        Precomputes window boundaries and uses vectorized operations.
        """
        if self._use_numba:
            # Use numba-optimized version
            window_starts, window_ends = _precompute_window_boundaries_time_driven(
                self._timestamps, self._lookback_interval
            )
            
            _compute_window_ofi_vectorized(
                self._buy_prices,
                self._buy_volumes,
                self._sell_prices,
                self._sell_volumes,
                window_starts,
                window_ends,
                self._ofi
            )
        else:
            # Fall back to numpy vectorized version
            self._compute_ofi_time_driven_numpy()
    
    def _compute_ofi_time_driven_numpy(self):
        """
        Numpy-based vectorized OFI computation for time-driven mode.
        """
        # Precompute window boundaries
        window_starts = np.searchsorted(
            self._timestamps, 
            self._timestamps - self._lookback_interval, 
            side='right'
        )
        
        # Process in batches for better memory efficiency
        batch_size = min(1000, self._num_samples)
        
        for batch_start in range(0, self._num_samples, batch_size):
            batch_end = min(batch_start + batch_size, self._num_samples)
            
            for i in range(batch_start, batch_end):
                start_idx = window_starts[i]
                end_idx = i + 1
                
                if end_idx - start_idx < 2:
                    self._ofi[i, :] = 0.0
                    continue
                
                # Vectorized OFI computation for all levels at once
                self._ofi[i, :] = self._compute_window_ofi_numpy(
                    start_idx, end_idx
                )
    
    def _compute_window_ofi_numpy(self, start_idx: int, end_idx: int) -> np.ndarray:
        """
        Numpy-vectorized OFI computation for a single window.
        """
        # Extract window data (views, no copy)
        buy_prices = self._buy_prices[start_idx:end_idx]
        buy_volumes = self._buy_volumes[start_idx:end_idx]
        sell_prices = self._sell_prices[start_idx:end_idx]
        sell_volumes = self._sell_volumes[start_idx:end_idx]
        
        # Vectorized price changes
        dpb = np.diff(buy_prices, axis=0)
        dps = np.diff(sell_prices, axis=0)
        
        # Vectorized OFI calculation
        ofi = (
            np.sum(buy_volumes[1:] * (dpb >= 0), axis=0) -
            np.sum(buy_volumes[:-1] * (dpb <= 0), axis=0) -
            np.sum(sell_volumes[1:] * (dps <= 0), axis=0) +
            np.sum(sell_volumes[:-1] * (dps >= 0), axis=0)
        )
        
        # Normalization
        total_volume = buy_volumes.sum() + sell_volumes.sum()
        Q = total_volume / (2.0 * self._num_levels)
        
        if Q > 1e-12:
            return ofi / Q
        else:
            return np.zeros(self._num_levels, dtype=self._dtype)
    
    def _compute_ofi_event_driven_optimized(self):
        """
        Optimized OFI computation for event-driven mode.
        """
        K = self._lookback_interval
        
        # Calculate sampled indices: K-1, 2K-1, 3K-1, ...
        self._sampled_indices = np.arange(K - 1, self._num_samples, K)
        
        if len(self._sampled_indices) == 0:
            return
        
        # Prepare window boundaries for all sampled indices
        window_starts = np.maximum(0, self._sampled_indices - K + 1)
        window_ends = self._sampled_indices + 1
        
        if self._use_numba:
            # Create output array for sampled indices only
            sampled_ofi = np.zeros((len(self._sampled_indices), self._num_levels), dtype=self._dtype)
            
            _compute_window_ofi_vectorized(
                self._buy_prices,
                self._buy_volumes,
                self._sell_prices,
                self._sell_volumes,
                window_starts,
                window_ends,
                sampled_ofi
            )
            
            # Copy results to main OFI array
            self._ofi[self._sampled_indices] = sampled_ofi
        else:
            # Numpy version
            for idx, i in enumerate(self._sampled_indices):
                start_idx = window_starts[idx]
                end_idx = window_ends[idx]
                
                if end_idx - start_idx < 2:
                    self._ofi[i, :] = 0.0
                    continue
                
                self._ofi[i, :] = self._compute_window_ofi_numpy(start_idx, end_idx)
    
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
        
        # Recompute
        self._compute_ofi()
        
        # Log summary
        unit = "ms" if self._mode == 'time_driven' else "snapshots"
        print(f"OFI recomputed: mode={self._mode}, L={self._num_samples}, "
              f"n={self._num_levels}, lookback_interval={self._lookback_interval}{unit}")