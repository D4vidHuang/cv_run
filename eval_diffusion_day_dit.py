import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion, DiffusiveRestoration


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Restoring Raindrop Clarity with Patch-Based Denoising Diffusion Models')
    parser.add_argument("--config", type=str, default='daytime_64.yml',
                        help="Path to the config file")
    parser.add_argument('--resume', default='/home1/yeying/release/RDiffusion/Param/RainDrop/DIT/RainDrop_DiT_ddpm.pth.tar', type=str,
                        help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument("--grid_r", type=int, default=16,
                        help="Grid cell width r that defines the overlap between patches")
    parser.add_argument("--sampling_timesteps", type=int, default=25,
                        help="Number of implicit sampling steps")
    parser.add_argument("--test_set", type=str, default='Raindrop_DiT',
                        help="restoration test set results: ['Raindrop_DiT', 'RDiffusion', 'IDT', 'restormer', 'Uformer', 'ICRA256', 'onego', 'atgan']")
    parser.add_argument("--image_folder", default='results/', type=str,
                        help="Location to save restored images")
    parser.add_argument('--seed', default=61, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')
    #parser.add_argument('--sid', type=str, default='00199')
    args = parser.parse_args()
    print(args)

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    if torch.cuda.is_available():
        print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    print("=> using dataset '{}'".format(config.data.dataset))
    DATASET = datasets.__dict__[config.data.dataset](config)
    _, val_loader = DATASET.get_loaders(parse_patches=False, validation=args.test_set)

    # create model
    print("=> creating denoising-diffusion model with wrapper...")
    diffusion = DenoisingDiffusion(args, config)
    model = DiffusiveRestoration(diffusion, args, config)
    for i in range(1, 212):
        sid_str = f"{i:05d}"  # 格式化为五位数字字符串
        try:
            print(f"Processing sid {sid_str} ...")
            model.restore(val_loader, validation=args.test_set, r=args.grid_r, sid=sid_str)
        except Exception as e:
            print(f"Skipping sid {sid_str} due to error: {e}")


if __name__ == '__main__':
    main()
