import torch

# GENERIC
GPU_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INIMAGE = "./input/input_img.png"
MASKIMAGE = "./input/mask.png"
OUTIMAGE = "./output/inpainted_img.png"
RESIZE_TO = (512, 512)
CUDA = True if torch.cuda.is_available() else False

# DEEPFILLv2
DEEPFILL_MODEL_PATH = "./model/deepfillv2_WGAN.pth"
GPU_ID = -1
INIT_TYPE = "xavier"
INIT_GAIN = 0.02
PAD_TYPE = "zero"
IN_CHANNELS = 4
OUT_CHANNELS = 3
LATENT_CHANNELS = 48
ACTIVATION = "elu"
NORM = "in"
NUM_WORKERS = 0
