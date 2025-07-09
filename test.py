import os
import time
import argparse
import numpy as np

import torch
from sklearn.metrics import r2_score
from src import ViT, GradientRMSE, RMSE


# Set command-line arguments
parser = argparse.ArgumentParser(description="ViT-STFlowNet")

parser.add_argument('--test_ma',
                    type=float,
                    default=0.31,
                    help="Test mach: 0.31, 0.4, 0.54, 0.63")
parser.add_argument("--method_type",
                    default=2,
                    type=int,
                    help="type of method")
parser.add_argument("--loss_num",
                    default=2,
                    type=int,
                    help="number of loss type")
parser.add_argument("--time_step",
                    default=64,
                    type=int,
                    help="input of time steps")
parser.add_argument("--encoder_depth",
                    default=3,
                    type=int,
                    help="depth of encoder and decoder")

args = parser.parse_args()

# Set random seed for reproducibility
random_seed = 1
np.random.seed(random_seed)

start_time = time.time()

def prediction(ma):
    # Set file paths for saving results and loading model
    save_file = f"./compare/Method{args.method_type}_Loss{args.loss_num}_{args.time_step}"
    save_model = os.path.join(save_file, "model")
    basic_file = './Data'

    # Load normalized data (10-channel) and concatenate with Mach number channel
    ###    The original code uses 10 channels of data, but here, to reduce the size of the data, 5 channels of data are used.    ###
    # data_10c = np.load(os.path.join(basic_file, "10C_dat", f"Ma{ma:02d}_t400_10C.npy")).astype(np.float32)
    data_10c = np.load(os.path.join(basic_file, "Norm_dat", f"Ma{ma:02d}_t400_norm.npy")).astype(np.float32)
    T, C, H, W = data_10c.shape[0], data_10c.shape[1], data_10c.shape[2], data_10c.shape[3]
    ma_data = np.ones((T, 1, H, W))
    ma_data = ((ma - 30) / (59 - 30)) * ma_data
    ma_data = ma_data.astype(np.float32)
    data_11c = np.concatenate((ma_data, data_10c), axis=1)

    # Select input channels based on method type
    in_channels = None
    if args.method_type == 1:
        in_channels = 8
    elif args.method_type == 2:
        in_channels = 6
        data_sample1 = data_11c[:, 0:1, :160, :]
        data_sample2 = data_11c[:, 6:8, :160, :]
        data_sample3 = data_11c[:, -3:, :160, :]
        infer_dataset = np.concatenate((data_sample1, data_sample2, data_sample3), axis=1)

        ###  In order to accommodate 5 channels of data, the following modifications were made:
        infer_dataset = data_11c[:, :, :160, :]

    elif args.method_type == 3:
        in_channels = 8
        data_sample1 = data_11c[:, 0:1, :160, :]
        data_sample2 = data_11c[:, 4:8, :160, :]
        data_sample3 = data_11c[:, -3:, :160, :]
        infer_dataset = np.concatenate((data_sample1, data_sample2, data_sample3), axis=1)
    elif args.method_type == 4:
        in_channels = 11
    elif args.method_type == 5:
        in_channels = 4

    # Initialize ViT model with specified input/output dimensions and architecture
    net = ViT(image_size=(160, 64), in_channels=in_channels * args.time_step, out_channels=3, patch_size=8, encoder_depths=args.encoder_depth,
              encoder_embed_dim=768, encoder_num_heads=12, decoder_depths=args.encoder_depth, decoder_embed_dim=512,
              decoder_num_heads=8, mlp_ratio=4, dropout_rate=0.1, compute_dtype=torch.float32)

    # Select training device (GPU if available, otherwise CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"--------------------------Using device: {device}--------------------------")
    net.to(device)

    # Load pre-trained model weights
    m_state_dict = torch.load(os.path.join(save_model, f"method{args.method_type}_{args.loss_num}loss_minloss.pth"), map_location=device)
    net.load_state_dict(m_state_dict)

    # Define loss function (RMSE)
    loss_function = RMSE(reduction='mean').to(device)
    print(f"Data preparation time before training: {time.time() - start_time:.2f}s")

    # Initialize evaluation statistics containers
    test_losses = []
    re_list = []
    mre_list = []
    r2_list = []
    data_list = []  # Store predictions for each timestep

    # Convert dataset to torch tensors
    true_set = torch.tensor(infer_dataset, dtype=torch.float32)
    pred_set = torch.tensor(infer_dataset, dtype=torch.float32)
    T = true_set.shape[0]

    net.eval()
    with torch.no_grad():
        inputs_t = true_set[:args.time_step, ...]
        for idx in range(T - args.time_step):
            inputs = inputs_t.reshape(-1, *true_set.shape[2:])
            inputs_batch = inputs.unsqueeze(0).to(device)
            pred_batch = net(inputs_batch)
            pred = pred_batch.squeeze(0)

            # Update prediction set with model output
            pred_set[idx + args.time_step, -3:, :, :] = pred

            # Slide input window
            inputs_old = inputs_t[1:, ...]
            inputs_new = pred_set[idx + args.time_step, ...].unsqueeze(0)
            inputs_t = torch.concatenate((inputs_old, inputs_new), dim=0)

            label = true_set[idx + args.time_step, -3:, :, :].to(device)
            loss = loss_function(pred, label)
            loss_aver = loss.item()

            # Record RMSE loss
            test_losses.append(loss_aver)

            # Convert tensors to numpy arrays for metric calculation
            label_numpy = label.cpu().detach().numpy()
            pred_numpy = pred.cpu().detach().numpy()

            # Compute relative error and mean relative error
            re = (np.max(np.abs(label_numpy - pred_numpy)) / (np.max(label_numpy) - np.min(label_numpy)))
            mre = np.mean(np.abs(label_numpy - pred_numpy)) / (np.max(label_numpy) - np.min(label_numpy))
            re_list.append(re)
            mre_list.append(mre)

            # Compute R² score
            label_flatten = label_numpy.flatten()
            pred_flatten = pred_numpy.flatten()
            r2 = r2_score(label_flatten, pred_flatten)
            r2_list.append(r2)

            # Store predictions
            data_list.append(np.squeeze(pred_numpy))

        # Combine metrics into one array for saving
        rmse_array = np.array(test_losses).reshape((-1, 1))
        re_array = np.array(re_list).reshape((-1, 1))
        mre_array = np.array(mre_list).reshape((-1, 1))
        r2_array = np.array(r2_list).reshape((-1, 1))
        loss_array = np.concatenate((rmse_array, re_array, mre_array, r2_array), axis=1)

        final_data = np.stack(data_list, axis=0)  # (T, C, H, W)
        print(f"final_pred data shape:{final_data.shape}")
        final_data_r1 = revert_min_max_normalization(final_data, 0.01 * ma)

    # Save metrics and predictions
    np.save(os.path.join(save_file, f"pred_loss{ma}.npy"), loss_array)
    np.save(os.path.join(save_file, f"pred_data{ma}.npy"), final_data_r1)
    print("Avg RMSE:", np.mean(loss_array[:, 0]))
    print("Avg MAX:", np.mean(loss_array[:, 1]))
    print("MAX MAX:", np.max(loss_array[:, 1]))


def revert_min_max_normalization(normalized_data, ma):
    """
    Revert normalized data back to original physical values.

    Args:
        normalized_data (np.ndarray): Normalized data.
        ma (float): Mach number scaling factor.

    Returns:
        np.ndarray: Data after reverting normalization.
    """

    reference_u = ma * 328.3929  # freestream velocity
    reference_rou = 0.904773  # density
    reference_p = 69694.6  # pressure

    # Restore pressure and velocity components
    normalized_data[:, 0:1, :, :] = normalized_data[:, 0:1, :, :] * 0.5 * reference_rou * reference_u * reference_u + reference_p
    normalized_data[:, 1:2, :, :] = normalized_data[:, 1:2, :, :] * reference_u
    normalized_data[:, 2:3, :, :] = normalized_data[:, 2:3, :, :] * reference_u

    return normalized_data


if __name__ == "__main__":

    ma = args.test_ma   # test mach:0.31, 0.4, 0.54, 0.63
    int_ma = int(100 * ma)
    prediction(int_ma)
