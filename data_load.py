'''
This file should be loading the pictures. Note that we cannot shuffle the pictures because we need pairs
'''
import torch
import torchvision

from constants import DIR, DAY_DIR, NIG_DIR, BATCH_SIZE

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

day_dataset = torchvision.datasets.ImageFolder(root=(DIR + DAY_DIR + '/'), 
                                                transform=torchvision.transforms.ToTensor())

nig_dataset = torchvision.datasets.ImageFolder(root=(DIR + NIG_DIR + '/'), 
                                                transform=torchvision.transforms.ToTensor())           

day_loader = torch.utils.data.DataLoader(dataset=day_dataset,
                                            batch_size=BATCH_SIZE, 
                                            shuffle=True)

nig_loader = torch.utils.data.DataLoader(dataset=nig_dataset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True)