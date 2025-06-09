"""
Dataset loading and processing module for aerodynamic simulation data.

This module handles:
- Loading preprocessed simulation data
- Creating input samples using different methods
- Splitting data into train/validation/test sets
- Generating time-series samples for machine learning tasks

# @Author: Zewei Chen
# @DateTime: Jun.2025
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Union


def load_data(data_path: str) -> np.ndarray:
    """
    Load and concatenate all 10-channel data files across Mach numbers.

    Args:
        data_path: Path to directory containing .npy files

    Returns:
        np.ndarray: Combined data with shape (N_ma, T, C, H, W)
                   where N_ma is number of Mach numbers (30)
    """
    data_list = []
    for ma in range(30, 60):
        # Load 10-channel data for current Mach number
        file_path = os.path.join(data_path, f"Ma{ma:02d}_t400_10C.npy")
        data_10c = np.load(file_path).astype(np.float32)

        # Add Mach number as additional channel
        T, C, H, W = data_10c.shape
        ma_channel = np.ones((T, 1, H, W), dtype=np.float32) * ((ma - 30) / 29)
        data_11c = np.concatenate((ma_channel, data_10c), axis=1)
        data_list.append(data_11c)

    return np.stack(data_list, axis=0)


def create_input_samples(
        data: np.ndarray,
        method_type: int,
        height_slice: slice = slice(None, 160)
) -> np.ndarray:
    """
    Create input samples using specified channel selection method.

    Args:
        data: Input data with shape (N_ma, T, C, H, W)
        method_type: Integer specifying channel selection method (1-5)
        height_slice: Slice object for height dimension cropping

    Returns:
        np.ndarray: Selected data samples

    Raises:
        ValueError: If invalid method_type is provided
    """
    method_channels = {
        1: [0, 1, 2, 4, 5, 8, 9, 10],  # Method 1 channels
        2: [0, 6, 7, 8, 9, 10],  # Method 2 channels
        3: [0, 4, 5, 6, 7, 8, 9, 10],  # Method 3 channels
        4: slice(None),  # All channels (Method 4)
        5: [0, 8, 9, 10]  # Method 5 channels
    }

    if method_type not in method_channels:
        raise ValueError(f"Invalid method_type: {method_type}. Must be 1-5.")

    # Apply channel selection and height cropping
    channel_selector = method_channels[method_type]
    return data[:, :, channel_selector, height_slice, :]


class DatasetSplitter:
    """
    Class for splitting data into train/validation/test sets.

    Attributes:
        data: Full dataset with shape (N_ma, T, C, H, W)
        method_type: Channel selection method (1-5)
        random_seed: Random seed for reproducibility
    """

    def __init__(self, data: np.ndarray, method_type: int, random_seed: int = 123456):
        """
        Initialize DatasetSplitter.

        Args:
            data: Input dataset
            method_type: Channel selection method
            random_seed: Seed for random operations
        """
        self.data = data
        self.method_type = method_type
        self.random_seed = random_seed

    def split_data(
            self,
            time_steps: int,
            split_type: str = "train"
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Split data into train/val or test sets.

        Args:
            time_steps: Number of time steps to keep (400, 200, or 100)
            split_type: Either "train" (returns train+val) or "test" (returns test)

        Returns:
            For "train": (train_data, val_data)
            For "test": test_data

        Raises:
            ValueError: If invalid time_steps or split_type provided
        """
        if time_steps not in {400, 200, 100}:
            raise ValueError("time_steps must be 400, 200, or 100")
        if split_type not in {"train", "test"}:
            raise ValueError('split_type must be "train" or "test"')

        time_stride = {400: 1, 200: 2, 100: 4}[time_steps]

        # Create reproducible random split
        np.random.seed(self.random_seed)
        n_ma = self.data.shape[0]
        indices = np.random.permutation(n_ma)

        # Split indices
        train_idx = indices[:int(n_ma * 0.7)]
        val_idx = indices[int(n_ma * 0.7):int(n_ma * 0.8)]
        test_idx = indices[int(n_ma * 0.8):]

        if split_type == "train":
            train_data = create_input_samples(
                self.data[train_idx, ::time_stride],
                self.method_type
            )
            val_data = create_input_samples(
                self.data[val_idx, ::time_stride],
                self.method_type
            )
            return train_data, val_data
        else:
            return create_input_samples(
                self.data[test_idx, ::time_stride],
                self.method_type
            )


class TimeSeriesDataset:
    """
    Dataset class for generating time-series samples.

    Attributes:
        data: Input data with shape (N_ma, T, C, H, W)
        time_step: Length of input time series
    """

    def __init__(self, data: np.ndarray, time_step: int = 32):
        """
        Initialize TimeSeriesDataset.

        Args:
            data: Input data
            time_step: Length of input time series
        """
        self.data = data
        self.time_step = time_step
        self.indices = self._generate_indices()

    def _generate_indices(self) -> List[Tuple[int, int]]:
        """
        Generate all possible (ma_idx, start_time) index pairs.

        Returns:
            List of (ma_idx, start_time) tuples
        """
        indices = []
        n_ma, n_time, _, _, _ = self.data.shape

        for ma_idx in range(n_ma):
            for t in range(n_time - self.time_step):
                indices.append((ma_idx, t))

        return indices

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get input-label pair for given index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (input_sequence, target)
            input_sequence: Shape (time_step*C, H, W)
            target: Shape (3, H, W)
        """
        ma_idx, start_t = self.indices[idx]

        # Get input sequence (last 3 channels are targets)
        input_seq = self.data[ma_idx, start_t:start_t + self.time_step, :]

        # Flatten time and channel dimensions
        input_seq = input_seq.reshape(-1, *self.data.shape[3:])

        # Get target (next time step's output variables)
        target = self.data[ma_idx, start_t + self.time_step, -3:]

        return input_seq, target


def create_dataset(
        data_path: str,
        time_steps: int,
        time_window: int,
        method_type: int,
        data_type: str = "train"
) -> Union[Tuple[TimeSeriesDataset, TimeSeriesDataset], np.ndarray]:
    """
    Main function for dataset creation.

    Args:
        data_path: Path to data directory
        time_steps: Total time steps (400, 200, or 100)
        time_window: Input sequence length
        method_type: Channel selection method (1-5)
        data_type: "train" or "test"

    Returns:
        For "train": (train_dataset, val_dataset)
        For "test": test_data

    Raises:
        ValueError: If invalid parameters provided
    """
    print("Preparing dataset...")

    # Input validation
    if time_steps not in {400, 200, 100}:
        raise ValueError("time_steps must be 400, 200, or 100")
    if time_window <= 0:
        raise ValueError("time_window must be positive")
    if method_type not in {1, 2, 3, 4, 5}:
        raise ValueError("method_type must be between 1-5")
    if data_type not in {"train", "test"}:
        raise ValueError('data_type must be "train" or "test"')

    # Load and process data
    data = load_data(data_path)
    splitter = DatasetSplitter(data, method_type)

    if data_type == "train":
        train_data, val_data = splitter.split_data(time_steps, "train")
        print(f"Train data shape: {train_data.shape}")
        print(f"Validation data shape: {val_data.shape}")

        return (
            TimeSeriesDataset(train_data, time_window),
            TimeSeriesDataset(val_data, time_window)
        )
    else:
        test_data = splitter.split_data(time_steps, "test")
        return test_data
