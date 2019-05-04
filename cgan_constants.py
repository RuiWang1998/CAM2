import numpy as np
import torch

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
IMG_INPUT_LEN = 150
NUM_CHANNEL = 3
LATENT_DIM = 600
PROCESSOR_OUT = 200
EPOCHS = 18000
LR_G = 1e-3
LR_D = 1e-3
BOTTLENEK = 400

G_PATH = 'data/night2day/model/generator'
D_PATH = 'data/night2day/model/discriminator'
