"""
Visualization tool for comparing CFD and predicted flow fields
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator

# Physical reference values for normalization
REF_VELOCITY = 328.3929  # Reference velocity factor
REF_DENSITY = 0.904773   # Reference density [kg/m³]
REF_PRESSURE = 69694.6   # Reference pressure [Pa]

# Set command-line arguments
parser = argparse.ArgumentParser(description="ViT-STFlowNet")

parser.add_argument('--test_ma',
                    type=float,
                    default=0.31,
                    help="Test mach: 0.31, 0.4, 0.54, 0.63")
args = parser.parse_args()

def min_max_numpy(data, mach):
    """Normalize flow field data using reference values

    Args:
        data: 4D numpy array (time, variables, x, y)
        mach: Mach number for reference velocity calculation

    Returns:
        Normalized data array
    """
    ref_u = mach * REF_VELOCITY
    # Pressure coefficient normalization
    data[:, 0] = (data[:, 0] - REF_PRESSURE) / (0.5 * REF_DENSITY * ref_u ** 2)
    # Velocity normalization
    data[:, 1:3] /= ref_u
    return data


def plot_flow_field(method_type, loss_num, time_step, ma, field_type='p'):
    """
    Plot flow field comparison (CFD vs Prediction vs Error) for specified field type

    Args:
        method_type: Type of method used for prediction
        loss_num: Loss function number
        time_step: Time step to analyze
        ma: Mach number
        field_type: Type of field to plot ('p' for pressure, 'u' for x-velocity, 'v' for y-velocity)

    Returns:
        None (saves plot to file)
    """
    # Set global font settings
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 14

    # Load and prepare data
    num_ma = int(ma * 100)
    real_data0 = np.load(f"./Data/True_dat/Ma{num_ma}_t400_true.npy")[:, :, :, :]
    real_data = real_data0[time_step:, -3:, :, :]

    data_file = f"./compare/Method{method_type}_Loss{loss_num}_{time_step}"
    pred_data = np.load(os.path.join(data_file, f"pred_data{num_ma}.npy"))
    pred_data = np.concatenate((pred_data, pred_data[:, :, 0:1, :]), axis=-2)

    # Normalize data
    real_data = min_max_numpy(real_data, ma)
    pred_data = min_max_numpy(pred_data, ma)

    xy_data = np.load(f"./Data/True_dat/xy_point.npy")
    # Configure plot layout
    fig = plt.figure(figsize=(20, 15))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05)

    # Plot dimensions
    subplot_width = 0.2
    subplot_height = 0.25
    cbar_width = 0.02
    cbar_height = 0.15

    ##########  Modifications need to be made based on the scope of the cloud map.  ###########
    # Set field-specific parameters
    if field_type == 'p':
        # Pressure field settings
        levels = np.arange(-1.2, 1.1, 0.1)
        error_levels = np.concatenate([
            np.arange(0, 0.001, 0.0002),
            np.arange(0.001, 0.004, 0.0005),
            np.arange(0.004, 0.01, 0.001),
            np.arange(0.01, 0.022, 0.002),
            [0.022]
        ])
        cmap = 'jet'
        field_idx = 0  # Pressure is first channel
    elif field_type == 'u':
        # X-velocity field settings
        levels = np.arange(-0.2, 1.6, 0.1)
        error_levels = np.concatenate([
            np.arange(0, 0.001, 0.0002),
            np.arange(0.001, 0.004, 0.0005),
            np.arange(0.004, 0.01, 0.001),
            np.arange(0.01, 0.014, 0.002),
            [0.014]
        ])
        cmap = 'jet'
        field_idx = 1  # U is second channel
    elif field_type == 'v':
        # Y-velocity field settings
        levels = np.arange(-0.4, 1.1, 0.1)
        error_levels = np.concatenate([
            np.arange(0, 0.001, 0.0002),
            np.arange(0.001, 0.004, 0.0005),
            np.arange(0.004, 0.01, 0.001),
            [0.01]
        ])
        cmap = 'jet'
        field_idx = 2  # V is third channel


    # Time steps to plot
    time_steps = [35, 135, 235, 335]

    for idx, t in enumerate(time_steps):
        # Get coordinates and field data
        data_real0 = [xy_data[t, i, :, :] for i in range(2)]  # X,Y coordinates
        columns_data_real = real_data[t, field_idx, :, :]
        columns_data_pred = pred_data[t, field_idx, :, :]

        # Calculate subplot positions
        left = 0.05 + idx * (subplot_width + 0.02)
        bottom_p = 0.7  # CFD plot position
        bottom_u = 0.4  # Prediction plot position
        bottom_v = 0.1  # Error plot position

        # Plot CFD results
        ax1 = fig.add_axes([left, bottom_p, subplot_width, subplot_height])
        contourf_real = ax1.contourf(data_real0[0], data_real0[1], columns_data_real, levels=levels, cmap=cmap)

        # Add colorbar for last subplot
        if idx == 3:
            cax1 = fig.add_axes([left + subplot_width + 0.01, bottom_p + (subplot_height - cbar_height) / 2,
                                 cbar_width, cbar_height])
            cbar1 = plt.colorbar(contourf_real, cax=cax1)
            cbar1.set_ticks(np.arange(levels[0], levels[-1] + 0.1, 0.2))
            cbar1.ax.text(0.5, 1.05, 'CFD', fontsize=16, fontstyle='italic', weight='bold',
                          ha='center', va='bottom', transform=cbar1.ax.transAxes)

        # Configure axes
        if idx == 0:
            ax1.set_ylabel('Y', fontsize=16, fontstyle='italic', weight='bold')
        ax1.set_title(f'T={t + 65}', fontsize=22, fontstyle='italic', weight='bold')
        ax1.set_xlim([-0.1, 0.3])
        ax1.set_ylim([-0.1, 0.3])
        ax1.contour(data_real0[0], data_real0[1], columns_data_real, colors='darkblue',
                    levels=levels, linestyles='dashed', linewidths=0.5)
        ax1.set_aspect('equal')
        ax1.text(0.02, 0.93, f'({chr(97 + idx)})', transform=ax1.transAxes,
                 fontsize=22, weight='bold', color='black',
                 bbox=dict(facecolor='white', edgecolor='none', pad=2))

        # Configure ticks
        _configure_ticks(ax1)

        # Plot prediction results
        ax2 = fig.add_axes([left, bottom_u, subplot_width, subplot_height])
        contourf_pred = ax2.contourf(data_real0[0], data_real0[1], columns_data_pred, levels=levels, cmap=cmap)

        if idx == 3:
            cax2 = fig.add_axes([left + subplot_width + 0.01, bottom_u + (subplot_height - cbar_height) / 2,
                                 cbar_width, cbar_height])
            cbar2 = plt.colorbar(contourf_pred, cax=cax2)
            cbar2.set_ticks(np.arange(levels[0], levels[-1] + 0.1, 0.2))
            cbar2.ax.text(0.5, 1.05, 'Pred', fontsize=16, fontstyle='italic', weight='bold',
                          ha='center', va='bottom', transform=cbar2.ax.transAxes)

        if idx == 0:
            ax2.set_ylabel('Y', fontsize=16, fontstyle='italic', weight='bold')
        ax2.set_xlim([-0.1, 0.3])
        ax2.set_ylim([-0.1, 0.3])
        ax2.contour(data_real0[0], data_real0[1], columns_data_pred, colors='darkblue',
                    levels=levels, linestyles='dashed', linewidths=0.5)
        ax2.set_aspect('equal')
        ax2.text(0.02, 0.93, f'({chr(97 + 4 + idx)})', transform=ax2.transAxes,
                 fontsize=22, weight='bold', color='black',
                 bbox=dict(facecolor='white', edgecolor='none', pad=2))

        _configure_ticks(ax2)

        # Plot error results
        ax3 = fig.add_axes([left, bottom_v, subplot_width, subplot_height])
        abs_diff = np.abs(columns_data_real - columns_data_pred)
        contourf_diff = ax3.contourf(data_real0[0], data_real0[1], abs_diff, levels=error_levels, cmap=cmap)

        if idx == 3:
            cax3 = fig.add_axes([left + subplot_width + 0.01, bottom_v + (subplot_height - cbar_height) / 2,
                                 cbar_width, cbar_height])
            cbar3 = plt.colorbar(contourf_diff, cax=cax3)
            cbar3.ax.text(0.5, 1.05, 'Error', fontsize=16, fontstyle='italic', weight='bold',
                          ha='center', va='bottom', transform=cbar3.ax.transAxes)

        ax3.set_xlabel('X', fontsize=16, fontstyle='italic', weight='bold')
        if idx == 0:
            ax3.set_ylabel('Y', fontsize=16, fontstyle='italic', weight='bold')
        ax3.set_xlim([-0.1, 0.3])
        ax3.set_ylim([-0.1, 0.3])
        ax3.set_aspect('equal')
        ax3.text(0.02, 0.93, f'({chr(97 + 8 + idx)})', transform=ax3.transAxes,
                 fontsize=22, weight='bold', color='black',
                 bbox=dict(facecolor='white', edgecolor='none', pad=2))

        _configure_ticks(ax3)

    # Save plot
    save_path = os.path.join(data_file, "summary")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    ma_path = os.path.join(save_path, f"{ma}")
    if not os.path.exists(ma_path):
        os.mkdir(ma_path)
    plt.savefig(os.path.join(ma_path, f"{field_type}_all.png"), dpi=300, bbox_inches='tight', pad_inches=0)


def _configure_ticks(ax):
    """Configure tick settings for subplots"""
    # Set minor tick locators
    ax.xaxis.set_minor_locator(MultipleLocator(0.02))  # x-axis minor ticks every 0.02
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))  # y-axis minor ticks every 0.01

    # Configure major ticks
    ax.tick_params(axis='both', which='major', direction='out',
                   length=4, width=0.8, bottom=True, top=False, left=True, right=False)

    # Configure minor ticks
    ax.tick_params(axis='both', which='minor', direction='out',
                   length=2, width=0.4, bottom=True, top=False, left=True, right=False)


if __name__ == "__main__":

    ma = args.test_ma  # test mach: 0.31, 0.4, 0.54, 0.63
    plot_flow_field(2, 2, 64, ma, field_type='p')
    plot_flow_field(2, 2, 64, ma, field_type='u')
    plot_flow_field(2, 2, 64, ma, field_type='v')





