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

def data_load(data_path):
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
        data_10c = np.load(os.path.join(data_path, f"Ma{ma:02d}_t400_10C.npy")).astype(np.float32)
        # Add Mach number as additional channel (normalized between 0-1)
        T, C, H, W = data_10c.shape[0], data_10c.shape[1], data_10c.shape[2], data_10c.shape[3]
        ma_data = np.ones((T, 1, H, W))
        ma_data = ((ma - 30)/(59 - 30)) * ma_data
        ma_data = ma_data.astype(np.float32)
        # Concatenate Mach number channel with original 10 channels
        data_11c = np.concatenate((ma_data, data_10c), axis=1)
        data_list.append(data_11c)
    # Stack all Mach number data along first dimension
    data = np.stack(data_list, axis=0)
    return data

def input_method(data, method_type):
    """
    Create input samples using different channel selection methods.

    Args:
        data: Input data with shape (N_ma, T, C, H, W)
        method_type: Integer specifying which channels to select

    Returns:
        np.ndarray: Selected channels concatenated along channel dimension
    """
    data_sample = None
    if method_type == 1:
        # Method 1: Select channels 0-2, 4-5, and last 3 channels
        data_sample1 = data[:, :, 0:3, :160, :]  # Crop height dimension (remove points far from wall)
        data_sample2 = data[:, :, 4:6, :160, :]
        data_sample3 = data[:, :, -3:, :160, :]
        data_sample = np.concatenate((data_sample1, data_sample2, data_sample3), axis=2)
    elif method_type == 2:
        # Method 2: Select channels 0, 6-7, and last 3 channels
        data_sample1 = data[:, :, 0:1, :160, :]
        data_sample2 = data[:, :, 6:8, :160, :]
        data_sample3 = data[:, :, -3:, :160, :]
        data_sample = np.concatenate((data_sample1, data_sample2, data_sample3), axis=2)
    elif method_type == 3:
        # Method 3: Select channels 0, 4-7, and last 3 channels
        data_sample1 = data[:, :, 0:1, :160, :]
        data_sample2 = data[:, :, 4:8, :160, :]
        data_sample3 = data[:, :, -3:, :160, :]
        data_sample = np.concatenate((data_sample1, data_sample2, data_sample3), axis=2)
    elif method_type == 4:
        # Method 4: Use all channels
        data_sample = data[:, :, :, :160, :]
    elif method_type == 5:
        # Method 5: Select channel 0 and last 3 channels
        data_sample1 = data[:, :, 0:1, :160, :]
        data_sample2 = data[:, :, -3:, :160, :]
        data_sample = np.concatenate((data_sample1, data_sample2), axis=2)
    return data_sample


class DatasetSource:
    """
    Class for loading and splitting the dataset into train/validation/test sets.
    """
    def __init__(self, data, method_type):
        """
        Initialize DatasetSource with raw data and method type.

        Args:
            data: Raw simulation data
            method_type: Channel selection method type (1-5)
        """
        self.data = data
        self.method_type = method_type

    def divide_data(self, time_all, data_type):
        """
        Split data into training/validation or test sets.

        Args:
            time_all: Total time steps (400, 200, or 100)
            data_type: "train" for train+val, else for test

        Returns:
            Tuple of datasets or single test dataset
        """
        data_length = self.data.shape[0]
        np.random.seed(123456)
        indices = np.random.permutation(data_length)

        # Split indices into 70% train, 10% validation, 20% test
        train_idx, valid_idx, test_idx = indices[0:int(data_length * 0.7)], indices[int(data_length * 0.7):int(data_length * 0.8)], indices[int(data_length * 0.8):]

        if data_type == "train":
            train_dataset, valid_dataset = None, None
            train_data = self.data.take(train_idx, axis=0)  # Take first 70% as training set

            # Apply time step subsampling if needed
            if time_all == 400:
                train_dataset = input_method(train_data, self.method_type)
            elif time_all == 200:
                train_dataset = input_method(train_data[:, ::2, :, :, :], self.method_type)
            elif time_all == 100:
                train_dataset = input_method(train_data[:, ::4, :, :, :], self.method_type)

            del train_data  # Free memory

            valid_data = self.data.take(valid_idx, axis=0)
            if time_all == 400:
                valid_dataset = input_method(valid_data, self.method_type)
            elif time_all == 200:
                valid_dataset = input_method(valid_data[:, ::2, :, :, :], self.method_type)
            elif time_all == 100:
                valid_dataset = input_method(valid_data[:, ::4, :, :, :], self.method_type)

            return train_dataset, valid_dataset

        else:
            # Handle test/inference data
            infer_dataset = None
            infer_data = self.data.take(test_idx, axis=0)
            if time_all == 400:
                infer_dataset = input_method(infer_data, self.method_type)  # Last 20% as test set
            elif time_all == 200:
                infer_dataset = input_method(infer_data[:, ::2, :, :, :], self.method_type)
            elif time_all == 100:
                infer_dataset = input_method(infer_data[:, ::4, :, :, :], self.method_type)

            return infer_dataset


class DatasetMake:
    """
    Class for creating time-series samples from the dataset.
    """
    def __init__(self, data, time_step=32):
        """
        Initialize DatasetMake with data and time step length.

        Args:
            data: Input data
            time_step: Length of time sequence for each sample
        """
        self.data = data
        self.time_step = time_step
        self.indices = self._generate_indices()  # Generate sample indices

    def _generate_indices(self):
        """
        Generate sample indices ensuring each sample stays within same Mach number.

        Returns:
            List of tuples (ma_idx, t) representing valid samples
        """
        indices = []
        ma, time, _, _, _ = self.data.shape
        for ma_idx in range(ma):  # For each Mach number
            for t in range(time - self.time_step):  # Ensure no out-of-bounds
                indices.append((ma_idx, t))  # Save Mach index and time start point
        return indices

    def __len__(self):
        """Return total number of samples."""
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Get input and label data for sample at given index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (input_data, label_data)
            - input_data: Time sequence of shape [time_step*C, H, W]
            - label_data: Next time step's last 3 channels
        """
        ma_idx, t = self.indices[idx]  # Get Mach number and time start point
        # Input is time sequence of length time_step
        input_data = self.data[ma_idx, t:t + self.time_step, :]  # Shape [32, C, H, W]
        # Reshape input by flattening time and channel dimensions
        input_data2 = input_data.reshape(-1, *self.data.shape[3:])

        # Label is next time step's last 3 channels (pressure components)
        label_data = self.data[ma_idx, t + self.time_step, -3:]  # Shape [3, H, W]

        return input_data2, label_data


def create_dataset(data_path, time_all, time_step, method_type, data_type="train"):
    """
    Main function to prepare dataset for training or inference.

    Args:
        data_path: Path to data directory
        time_all: Total time steps (400, 200, or 100)
        time_step: Length of time sequence
        method_type: Channel selection method (1-5)
        data_type: "train" or other for test

    Returns:
        Prepared dataset(s) depending on data_type
    """
    print("Preparing DataSet......")
    data = data_load(data_path)

    if data_type == "train":
        train_data, valid_data = DatasetSource(data, method_type).divide_data(time_all=time_all, data_type=data_type)
        train_dataset = DatasetMake(train_data, time_step)  # Shape: (70%n , 2) where 2 represents (inputs, labels)
        valid_dataset = DatasetMake(valid_data, time_step)
        print(f"TrainDataSet shape: {train_data.shape}")
        print(f"ValidDataSet shape: {valid_data.shape}")
        return train_dataset, valid_dataset

    else:
        infer_data = DatasetSource(data, method_type).divide_data(time_all=time_all, data_type=data_type)
        return infer_data
