'''
This file should be loading the pictures. Note that we cannot shuffle the pictures because we need pairs
'''
import torch
import torchvision
import torchvision.transforms as transforms

from cgan_constants import BATCH_SIZE, DAY_DIR, DIR, IMG_INPUT_LEN, NIG_DIR

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transf = transforms.Compose([
    transforms.Resize(IMG_INPUT_LEN),
    transforms.CenterCrop(IMG_INPUT_LEN),
    transforms.ToTensor(), ])

day_dataset = torchvision.datasets.ImageFolder(root=(DIR + DAY_DIR + '/'), transform=transf)
nig_dataset = torchvision.datasets.ImageFolder(root=(DIR + NIG_DIR + '/'), transform=transf)

day_loader = torch.utils.data.DataLoader(dataset=day_dataset,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True)
nig_loader = torch.utils.data.DataLoader(dataset=nig_dataset,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True)
