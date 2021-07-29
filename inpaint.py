import os
import time
import datetime
from types import SimpleNamespace
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from config import *
from deepfillv2 import test_dataset
from deepfillv2 import utils

def WGAN_tester():

    # Save the model if pre_train == True
    def load_model_generator():
        pretrained_dict = torch.load(
            DEEPFILL_MODEL_PATH, map_location=torch.device(GPU_DEVICE)
        )
        generator.load_state_dict(pretrained_dict)

    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # configurations
    results_path = os.path.dirname(OUTIMAGE)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Build networks
    opt = SimpleNamespace(
        pad_type=PAD_TYPE,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        latent_channels=LATENT_CHANNELS,
        activation=ACTIVATION,
        norm=NORM,
        init_type=INIT_TYPE,
        init_gain=INIT_GAIN,
        use_cuda=CUDA,
        gpu_device=GPU_DEVICE,
    )
    generator = utils.create_generator(opt).eval()
    print("-- INPAINT: Loading Pretrained Model --")
    load_model_generator()

    # To device
    generator = generator.to(GPU_DEVICE)

    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    trainset = test_dataset.InpaintDataset()

    # Define the dataloader
    dataloader = DataLoader(
        trainset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # ----------------------------------------
    #            Testing
    # ----------------------------------------
    # Testing loop
    for batch_idx, (img, mask) in enumerate(dataloader):
        img = img.to(GPU_DEVICE)
        mask = mask.to(GPU_DEVICE)

        # Generator output
        with torch.no_grad():
            first_out, second_out = generator(img, mask)

        # forward propagation
        first_out_wholeimg = (
            img * (1 - mask) + first_out * mask
        )  # in range [0, 1]
        second_out_wholeimg = (
            img * (1 - mask) + second_out * mask
        )  # in range [0, 1]

        masked_img = img * (1 - mask) + mask
        mask = torch.cat((mask, mask, mask), 1)
        img_list = [second_out_wholeimg]
        name_list = ["second_out"]
        utils.save_sample_png(
            sample_folder=results_path,
            sample_name=os.path.basename(OUTIMAGE),
            img_list=img_list,
            name_list=name_list,
            pixel_max_cnt=255,
        )
        print("-- Inpainting is finished --")


if __name__ == "__main__":
    WGAN_tester()
