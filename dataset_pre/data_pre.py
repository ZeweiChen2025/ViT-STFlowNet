"""
Data preprocessing pipeline for aerodynamic simulation data.

This script processes raw simulation data files (.dat) into normalized numpy arrays
with multiple channels (8C and 10C formats) suitable for machine learning tasks.

The pipeline includes:
1. Reading and parsing raw .dat files
2. Computing Signed Distance Fields (SDF)
3. Applying Gaussian filters
4. Normalizing data globally across all Mach numbers
5. Saving processed data in numpy format

@Author: Zewei Chen
@DateTime: Jun.2025
"""

import numpy as np
import matplotlib as mpl
import os
from typing import Tuple, Optional


def sdf(point: np.ndarray, surface: np.ndarray) -> float:
    """
    Compute Signed Distance Field (SDF) value for a point relative to a surface.

    Args:
        point: 2D point coordinates (x,y)
        surface: Array of points defining the surface (N,2)

    Returns:
        float: Minimum distance from point to surface
    """
    return np.min(np.linalg.norm(point - surface, axis=1))


def gaussian_filter(sdf_values: np.ndarray, lambda_const: float) -> np.ndarray:
    """
    Apply Gaussian filter to SDF values.

    Args:
        sdf_values: Input SDF values
        lambda_const: Decay constant for the Gaussian filter

    Returns:
        np.ndarray: Filtered SDF values
    """
    return np.exp(-lambda_const * sdf_values)


def min_max_normalization_numpy(
        data: np.ndarray,
        ma: float,
        max_val: np.ndarray,
        min_val: np.ndarray
) -> np.ndarray:
    """
    Normalize input data using min-max scaling and dimensionless output channels.

    Args:
        data: Input data with shape (T, C, H, W)
        ma: Mach number
        max_val: Maximum values for normalization (shape (1,5,1,1))
        min_val: Minimum values for normalization (shape (1,5,1,1))

    Returns:
        np.ndarray: Normalized data with same shape as input
    """
    # Normalize input channels (0-2) using min-max scaling
    data[:, 0:2, :, :] = (data[:, 0:2, :, :] - min_val) / (max_val - min_val)

    # Dimensionless normalization for output channels (2-4)
    reference_u = 0.01 * ma * 328.3929  # Reference velocity
    reference_rou = 0.904773  # Reference density
    reference_p = 69694.6  # Reference pressure

    # Pressure coefficient normalization
    data[:, 2:3, :, :] = (data[:, 2:3, :, :] - reference_p) / (
            0.5 * reference_rou * reference_u ** 2
    )

    # Velocity component normalization
    data[:, 3:4, :, :] /= reference_u
    data[:, 4:5, :, :] /= reference_u

    return data


class DataPreprocessor:
    """Class for preprocessing aerodynamic simulation data."""

    def __init__(self, data_file: str, ma: int):
        """
        Initialize data preprocessor.

        Args:
            data_file: Base directory containing data files
            ma: Mach number (30-59, representing 0.30-0.59)
        """
        self.data_file = data_file
        self.ma = ma

    def process_dat_files(self) -> np.ndarray:
        """
        Process raw .dat files into numpy array format.

        Returns:
            np.ndarray: Processed data with shape (T, C, H, W)
                        T: Time steps (400)
                        C: Channels (5)
                        H: Height (64)
                        W: Width (108)
        """
        data_list = []
        npy_file = os.path.join(self.data_file, "tecplot_dat", f"{self.ma}")

        for time_step in range(101, 501):
            file_path = os.path.join(npy_file, f"{time_step}.dat")

            with open(file_path, 'r') as file:
                # Skip header lines (first 19) and read data lines
                lines = file.readlines()[19:10259]

            # Parse data into numpy array
            data = np.array(
                [list(map(float, line.split())) for line in lines]
            ).astype(np.float32)

            # Swap velocity components (u and v)
            data[[1, 2]] = data[[2, 1]]

            # Process data blocks
            # Block 1: First 10 rows (special handling)
            data01 = data[:10 * 64, :5].reshape(10, 64, 5).transpose(2, 0, 1)
            data_12 = data01[:, 0:2, :].reshape(5, 64, 2).transpose(0, 2, 1)
            data1 = np.concatenate((data_12, data01[:, 2:, :]), axis=1)
            data1 = data1[:, :, ::-1]  # Reverse column order

            # Block 2: Next 71 rows
            data2 = data[(10 * 64):(81 * 64), :5].reshape(71, 64, 5).transpose(2, 0, 1)
            data2 = data2[:, :, ::-1]

            # Block 3: Next 72 rows
            data3 = data[(81 * 64):(153 * 64), :5].reshape(72, 64, 5).transpose(2, 0, 1)
            data3 = data3[:, :, ::-1]

            # Block 4: Remaining 7 rows
            data4 = data[(153 * 64):, :5].reshape(7, 64, 5).transpose(2, 0, 1)
            data4 = data4[:, :, ::-1]

            # Combine all blocks into final grid
            combined_data = np.concatenate(
                (data2, data4, data3[:, ::-1, :],
                 data1[:, ::-1, :], data2[:, 0:1, :]),
                axis=1
            )
            data_list.append(combined_data)

        # Stack all time steps into 4D array
        return np.stack(data_list, axis=0)

    def save_data(self, data: np.ndarray) -> None:
        """
        Process 5-channel data into 8-channel format and save to file.

        Adds:
        - SDF (Signed Distance Field)
        - Gaussian filtered x and y coordinates

        Args:
            data: Input 5-channel data with shape (T,5,H,W)
        """
        T = data.shape[0]
        sdf_list = []

        print(f"********** Processing Ma:{0.01 * self.ma} **********")

        for t in range(T):
            # Extract surface points (first column)
            surface_points = data[t, 0:2, :, 0].T

            # Compute SDF for all grid points
            grid_points = data[t, 0:2].reshape(2, 161 * 64).T
            sdf_values = np.apply_along_axis(
                lambda p: sdf(p, surface_points),
                1,
                grid_points
            )
            sdf_grid = sdf_values.reshape(161, 64).T
            sdf_list.append(sdf_grid)

        # Process SDF data
        sdf_data = np.stack(sdf_list, axis=0)
        sdf_data = np.expand_dims(sdf_data, axis=1)
        sdf_data = np.transpose(sdf_data, (0, 1, 3, 2))

        # Create 6-channel data (original 5 + SDF)
        data_6c = np.concatenate(
            (data[:, 0:2, :, :], sdf_data, data[:, 2:, :, :]),
            axis=1
        )

        # Apply Gaussian filter to create 8-channel data
        lambda_const = 1.0
        filtered = gaussian_filter(data_6c[:, 2, :, :], lambda_const)
        fM_x = np.expand_dims(filtered * data_6c[:, 0, :, :], axis=1)
        fM_y = np.expand_dims(filtered * data_6c[:, 1, :, :], axis=1)

        ###    In order to ensure that the size of the npy data meets GitHub's upload requirements,
        # only 5 channels of data are saved here for subsequent testing tasks.   ###
        ## Mx, My, p, u, v
        data_5c = np.concatenate(
            (fM_x, fM_y, data_6c[:, 3:, :, :]),
            axis=1
        )
        np.save(os.path.join(self.data_file, "True_dat", f"Ma{self.ma:02d}_t400_true.npy"), data_5c)

        data_xy = data_6c[:, :2, :, :]
        np.save(os.path.join(self.data_file, "True_dat", f"xy_point.npy"), data_xy)

        print(f"Ma:{0.01 * self.ma} 5C data saved successfully.")



def create_normalized_data(data_file: str) -> None:
    """
    Create 10-channel normalized data from 8-channel data.

    Adds:
    - Repeated x and y coordinates from first row

    Args:
        data_file: Base directory containing data files
    """
    # Find global min/max values across all Mach numbers
    min_val = None
    max_val = None

    for ma in range(30, 60):
        data_path = os.path.join(data_file, "True_dat", f"Ma{ma:02d}_t400_true.npy")
        data_5c = np.load(data_path)

        if ma == 30:
            min_val = np.min(data_5c[:, 0:2, :, :], axis=(0, 2, 3), keepdims=True)
            max_val = np.max(data_5c[:, 0:2, :, :], axis=(0, 2, 3), keepdims=True)
        else:
            current_min = np.min(data_5c[:, 0:2, :, :], axis=(0, 2, 3), keepdims=True)
            current_max = np.max(data_5c[:, 0:2, :, :], axis=(0, 2, 3), keepdims=True)

            min_val = np.minimum(min_val, current_min)
            max_val = np.maximum(max_val, current_max)

    print("Global max values:", max_val)
    print("Global min values:", min_val)

    # Save normalization parameters
    np.save(os.path.join(data_file, "Norm_dat", "Max_val.npy"), max_val)
    np.save(os.path.join(data_file, "Norm_dat", "Min_val.npy"), min_val)

    # Process each Mach number
    for ma in range(30, 60):
        data_path = os.path.join(data_file, "True_dat", f"Ma{ma:02d}_t400_true.npy")
        data_5c = np.load(data_path)

        # Normalize data
        normalized = min_max_normalization_numpy(data_5c, ma, max_val, min_val)

        # Save 10-channel data
        output_path = os.path.join(
            data_file,
            "Norm_dat",
            f"Ma{ma:02d}_t400_norm.npy"
        )
        np.save(output_path, normalized)


if __name__ == "__main__":
    DATA_ROOT = "../Data"

    # Process raw data for all Mach numbers

    # for ma in range(30, 60):
    for ma in range(31, 32):
        processor = DataPreprocessor(DATA_ROOT, ma)
        raw_data = processor.process_dat_files()
        processor.save_data(raw_data)

    # Create normalized 10-channel data
    create_normalized_data(DATA_ROOT)
