import torch

from function import visualize
from models import Generator

generator = Generator()
generator.load_state_dict(torch.load('C:/New folder/Github/CAM2/CAM2/model/generator'))