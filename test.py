import os
import time
import argparse
import numpy as np

import torch
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from src import create_dataset
from src import ViT, GradientRMSE, RMSE

parser = argparse.ArgumentParser(description="Hypersonic turbulent over the windward side of a lifting body")

parser.add_argument('--data_dir',
                    type=str,
                    default="E:/Desktop/BWD/10C_dat",      # 通道：x y x0 y0(翼型表面) SDF MX MY   P U V
                    help="dataset store direction")
parser.add_argument('--batch_size',
                    type=int,
                    default=1,
                    help="mini batch_size")
parser.add_argument("--method_type",
                    default=1,
                    type=int,
                    help="type of method")
parser.add_argument("--loss_num",
                    default=2,
                    type=int,
                    help="number of loss type")
parser.add_argument("--step_lr",
                    default=0.3,
                    type=float,
                    help="learning rate step")
parser.add_argument("--time_all",
                    default=400,
                    type=int,
                    help="all time")
parser.add_argument("--time_step",
                    default=64,
                    type=int,
                    help="input of time steps")
parser.add_argument("--encoder_depth",
                    default=3,
                    type=int,
                    help="depth of encoder and decoder")
args = parser.parse_args()

random_seed = 1

np.random.seed(random_seed)

start_time = time.time()


def prediction():

    save_file = f"./compare/Method{args.method_type}_Loss{args.loss_num}_{args.time_step}"
    save_model = os.path.join(save_file, "model")

    test_set = create_dataset(args.data_dir, args.time_all, args.time_step, args.method_type, data_type="pred")

    in_channels = None
    if args.method_type == 1:
        in_channels = 8
    elif args.method_type == 2:
        in_channels = 6
    elif args.method_type == 3:
        in_channels = 8
    elif args.method_type == 4:
        in_channels = 11

    net = ViT(image_size=(160, 64), in_channels=in_channels * args.time_step, out_channels=3, patch_size=8, encoder_depths=args.encoder_depth,
              encoder_embed_dim=768, encoder_num_heads=12, decoder_depths=args.encoder_depth, decoder_embed_dim=512,
              decoder_num_heads=8, mlp_ratio=4, dropout_rate=0.1, compute_dtype=torch.float32)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"--------------------------device：{device}--------------------------")

    net.to(device)

    m_state_dict = torch.load(os.path.join(save_model, f"method{args.method_type}_{args.loss_num}loss_minloss.pth"), map_location=device)
    net.load_state_dict(m_state_dict)

    loss_function = RMSE(reduction='mean').to(device)
    print(f"Initial setup completed in {time.time()-start_time:.2f}s")
    all_loss_array = []
    all_pred_data = []
    for i in range(6):
        # 预测损失
        test_losses = []
        re_list = []
        mre_list = []
        r2_list = []
        data_list = []
        true_set = torch.tensor(test_set[i, ...], dtype=torch.float32)
        pred_set = torch.tensor(test_set[i, ...], dtype=torch.float32)
        T = true_set.shape[0]
        net.eval()
        with torch.no_grad():
            inputs_t = true_set[:args.time_step, ...]
            for idx in range(T - args.time_step):
                inputs = inputs_t.reshape(-1, *true_set.shape[2:])
                # plt.imshow(inputs[2, ...])
                # plt.show()
                inputs_batch = inputs.unsqueeze(0).to(device)
                pred_batch = net(inputs_batch)
                pred = pred_batch.squeeze(0)
                pred_set[idx + args.time_step, -3:, :, :] = pred
                inputs_old = inputs_t[1:, ...]
                inputs_new = pred_set[idx + args.time_step, ...].unsqueeze(0)
                inputs_t = torch.concatenate((inputs_old, inputs_new), dim=0)
                label = true_set[idx + args.time_step, -3:, :, :].to(device)

                loss = loss_function(pred, label)
                loss_aver = loss.item()

                test_losses.append(loss_aver)

                label_numpy = label.cpu().detach().numpy()
                pred_numpy = pred.cpu().detach().numpy()

                re = (np.max(np.abs(label_numpy - pred_numpy)) / (np.max(label_numpy) - np.min(label_numpy)))
                mre = np.mean(np.abs(label_numpy - pred_numpy)) / (np.max(label_numpy) - np.min(label_numpy))
                re_list.append(re)
                mre_list.append(mre)

                label_flatten = label_numpy.flatten()
                pred_flatten = pred_numpy.flatten()
                r2 = r2_score(label_flatten, pred_flatten)
                r2_list.append(r2)

                data_list.append(np.squeeze(pred_numpy))

            rmse_array = np.stack(test_losses).reshape((-1, 1))
            re_array = np.stack(re_list).reshape((-1, 1))
            mre_array = np.stack(mre_list).reshape((-1, 1))
            r2_array = np.stack(r2_list).reshape((-1, 1))
            loss_array = np.concatenate((rmse_array, re_array, mre_array, r2_array), axis=1)

            final_data = np.stack(data_list, axis=0)  # (T, C, H, W)
            print(f"final_pred data shape:{final_data.shape}")

            np.random.seed(123456)
            indices = np.random.permutation(30)
            train_idx, valid_idx, test_idx = indices[0:int(30 * 0.7)], indices[int(30 * 0.7):int(
                30 * 0.8)], indices[int(30 * 0.8):]
            ma = test_idx[i] * 0.01 + 0.3
            final_data_r1 = revert_min_max_normalization(final_data, ma)

        all_loss_array.append(loss_array)
        all_pred_data.append(final_data_r1)

    all_loss = np.stack(all_loss_array, axis=0)
    all_pred = np.stack(all_pred_data, axis=0)
    np.save(os.path.join(save_file, f"all_pred_loss.npy"), all_loss)
    np.save(os.path.join(save_file, f"all_pred_data.npy"), all_pred)
    print("Avg RMSE:", np.mean(all_loss[:, :, 0]))
    print("Avg MAX:", np.mean(all_loss[:, :, 1]))
    print("MAX MAX:", np.max(all_loss[:, :, 1]))


def revert_min_max_normalization(normalized_data, ma):

    reference_u = ma * 328.3929
    reference_rou = 0.904773
    reference_p = 69694.6

    # 恢复三个通道的原始值
    normalized_data[:, 0:1, :, :] = normalized_data[:, 0:1, :, :] * 0.5 * reference_rou * reference_u * reference_u + reference_p
    normalized_data[:, 1:2, :, :] = normalized_data[:, 1:2, :, :] * reference_u
    normalized_data[:, 2:3, :, :] = normalized_data[:, 2:3, :, :] * reference_u

    return normalized_data

if __name__ == "__main__":
    prediction()

