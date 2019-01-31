DIR = 'night2day/Train/'
NIG_DIR = 'night'
DAY_DIR = 'day'

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

DIR = 'data/night2day/Train/'
NIG_DIR = 'night'
DAY_DIR = 'day'

BATCH_SIZE = 10
IMG_INPUT_LEN = 120
NUM_CHANNEL = 3
LATENT_DIM = 600
PROCESSOR_OUT = 200
EPOCHS = 5
LR = 1e-4
BOTTLENEK = 512

G_PATH = 'CAM2/night2day/model/generator'
D_PATH = 'CAM2/night2day/model/discriminator'
