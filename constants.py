import platform
import torch
import glob
import os
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

SEED = 1127
def seed_everything(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()

SOURCE_WINDOWS = 'C:/'
SOURCE_LINUX = '/mnt/c/'
SECONDARY_DIR = '/New folder/Github/CAM2/data/night2day/train/'
NIG_DIR = 'night'
DAY_DIR = 'day'

BATCH_SIZE = 10
IMG_INPUT_LEN = 100
NUM_CHANNEL = 3
LATENT_DIM = 500
PROCESSOR_OUT = 200
EPOCHS = 5
LR = 1e-4

if platform.system() == 'Linux':
    DIR = SOURCE_LINUX + SECONDARY_DIR
else:
    DIR = SOURCE_WINDOWS + SECONDARY_DIR