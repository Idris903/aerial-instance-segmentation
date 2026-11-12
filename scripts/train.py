import torch
import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),"../src"))

from dataset import Tile_Dataset
from torch.utils.data import DataLoader
from model import Detector


train_dataset = Tile_Dataset(os.path.expanduser("~/data/tiles/train"))
train_dataloader = DataLoader(train_dataset, 
                              batch_size = 8, 
                              shuffle= True,
                              num_workers= 2,
                              persistent_workers= True)


device = torch.device("cuda")
model = Detector(3, 15).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr= 1e-4)

scaler = torch.amp.GradScaler(device.type)









