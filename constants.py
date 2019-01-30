import platform
import torch
import glob
import os
import numpy as np

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

if platform.system() == 'Linux':
    DIR = SOURCE_LINUX + SECONDARY_DIR
else:
    DIR = SOURCE_WINDOWS + SECONDARY_DIR