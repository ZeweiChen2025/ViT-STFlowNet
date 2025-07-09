"""
training process

The training process code is for reference only and cannot be run directly.
Due to the large size of the training data, it cannot be uploaded to the GitHub website.
If necessary, please contact us via email: zwchen2024@163.com

@Author: Zewei Chen
@DateTime: Jun.2025
"""
import os
import time
import argparse
import numpy as np

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from src import create_dataset
from src import ViT, GradientRMSE, RMSE

# Argument parser to configure training hyperparameters and runtime options
parser = argparse.ArgumentParser(description="ViT-STFlowNet")

parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")
parser.add_argument('--data_dir',
                    type=str,
                    default="./Data/10C_dat",
                    help="Directory containing dataset")
parser.add_argument('--batch_size',
                    type=int,
                    default=64,
                    help="Mini-batch size")
parser.add_argument("--lr",
                    default=0.0008,
                    type=float,
                    help="Initial learning rate")
parser.add_argument("--step_lr",
                    default=0.3,
                    type=float,
                    help="Learning rate decay factor")
parser.add_argument("--epochs",
                    default=1000,
                    type=int,
                    help="Total number of epochs")
parser.add_argument("--method_type",
                    default=2,
                    type=int,
                    help="Method type (used to select input channels)")
parser.add_argument("--loss_num",
                    default=2,
                    type=int,
                    help="Loss function type")
parser.add_argument("--time_all",
                    default=400,
                    type=int,
                    help="Total time steps in dataset")
parser.add_argument("--time_step",
                    default=32,
                    type=int,
                    help="Sliding window size for input time steps")
parser.add_argument("--encoder_depth",
                    default=3,
                    type=int,
                    help="Depth of encoder and decoder")
args = parser.parse_args()

# Set seed for reproducibility
random_seed = 1
np.random.seed(random_seed)


def save_model_parameters(dir_name, net, optimizer, epoch):
    """Save model and optimizer state to a file for resuming training."""
    filename = os.path.join(dir_name, f"vit_continue_epoch{epoch}.pth")
    state = {
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, filename)


def load_model_parameters(model, optimizer, epoch):
    """Load model and optimizer state from a checkpoint file."""
    filename = f'{args.savemodel_dir}/vit_continue_epoch{epoch}.pth'
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer


def train():
    """Main training loop for the Vision Transformer model."""
    save_file = f"./compare/Method{args.method_type}_Loss{args.loss_num}"
    if not os.path.exists(save_file):
        os.mkdir(save_file)
    save_model = os.path.join(save_file, "model")
    if not os.path.exists(save_model):
        os.mkdir(save_model)

    # Load training and validation datasets
    train_set, valid_set = create_dataset(args.data_dir, args.time_all, args.time_step, args.method_type, data_type="train")
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_set, batch_size=args.batch_size, shuffle=True)

    # Select input channel size based on method type
    in_channels = None
    if args.method_type == 1:
        in_channels = 8
    elif args.method_type == 2:
        in_channels = 6
    elif args.method_type == 3:
        in_channels = 8
    elif args.method_type == 4:
        in_channels = 11

    # Initialize Vision Transformer model
    net = ViT(image_size=(160, 64), in_channels=in_channels * args.time_step, out_channels=3, patch_size=8,
              encoder_depths=args.encoder_depth, encoder_embed_dim=768, encoder_num_heads=12,
              decoder_depths=args.encoder_depth, decoder_embed_dim=512, decoder_num_heads=8,
              mlp_ratio=4, dropout_rate=0.1, compute_dtype=torch.float32)

    # Select GPU or CPU as training device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device_id}")

    print(f"--------------------------Device: {device}--------------------------")

    net.to(device)

    # Select loss function based on loss_num
    if args.loss_num == 1:
        loss_function = RMSE(reduction='mean').to(device)
    elif args.loss_num == 2:
        loss_function = GradientRMSE(dynamic_flag=True).to(device)
    elif args.loss_num == 3:
        loss_function = GradientRMSE(dynamic_flag=False).to(device)

    # Initialize optimizer and learning rate scheduler
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0001)
    pla_lr_scheduler = lr_scheduler.StepLR(optimizer, 0.2 * args.epochs, args.step_lr)

    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    print(f"Initial setup completed in {time.time()-start_time:.2f}s")

    min_valid_loss = 100  # Initialize high validation loss for comparison

    # Main epoch loop
    for epoch in range(1, args.epochs + 1):
        local_time_beg = time.time()

        for i, (inputVar, targetVar) in enumerate(train_loader):
            net.train()
            inputs = inputVar.to(device)
            label = targetVar.to(device)

            optimizer.zero_grad()
            pred = net(inputs)

            loss = loss_function(pred, label)
            loss_aver = loss.item()
            train_losses.append(loss_aver)

            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10)  # Gradient clipping
            optimizer.step()

        train_loss = np.average(train_losses) / args.batch_size
        avg_train_losses.append(train_loss)

        print(f"epoch: {epoch}, epoch average train loss: {train_loss :.6f}, "
              f"epoch time: {(time.time() - local_time_beg):.2f}s")

        # Validation every epoch
        if epoch % 1 == 0:
            print(f"=================Start Evaluation=====================")
            eval_time_beg = time.time()

            with torch.no_grad():
                net.eval()

                for i, (inputVar, targetVar) in enumerate(valid_loader):
                    inputs = inputVar.to(device)
                    label = targetVar.to(device)
                    pred = net(inputs)

                    loss = loss_function(pred, label)
                    loss_aver = loss.item()
                    valid_losses.append(loss_aver)

            torch.cuda.empty_cache()

            valid_loss = np.average(valid_losses) / args.batch_size
            avg_valid_losses.append(valid_loss)

            print(f"epoch: {epoch} train loss: {train_loss} valid loss: {valid_loss:.6f}")
            print(f"epoch: {epoch}, epoch average valid loss: {valid_loss :.6f}, "
                  f"epoch time: {(time.time() - eval_time_beg):.2f}s")
            print(f"==================End Evaluation======================")

            pla_lr_scheduler.step()  # Apply LR schedule

        # Reset loss trackers for next epoch
        train_losses = []
        valid_losses = []

        # Save training loss to file
        with open(os.path.join(save_file, "train_loss.txt"), 'wt') as f:
            j = 1
            for i in avg_train_losses:
                print(j, i, file=f)
                j += 1

        # Save validation loss to file
        with open(os.path.join(save_file, "valid_loss.txt"), 'wt') as f:
            j = 1
            for i in avg_valid_losses:
                print(j, i, file=f)
                j += 1

        # Save model if current validation loss is the lowest seen so far
        if min_valid_loss >= valid_loss:
            min_valid_loss = valid_loss
            torch.save(net.state_dict(), os.path.join(save_model, f"method{args.method_type}_{args.loss_num}loss_minloss.pth"))

        # Save final model at last epoch
        if epoch == args.epochs:
            save_model_parameters(save_file, net, optimizer, epoch)


if __name__ == "__main__":
    print("pid:", os.getpid())

    start_time = time.time()
    train()
    print(f"Start-to-End total time: {(time.time() - start_time):.2f}s")
