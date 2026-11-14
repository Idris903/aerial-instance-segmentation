import torch
import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),"../src"))

from dataset import Tile_Dataset
from torch.utils.data import DataLoader
from model import Detector
from loss import focal_loss
from loss import reg_l1_loss
from tqdm import tqdm



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

checkpoint_path = os.path.expanduser("~/checkpoint")

os.makedirs(checkpoint_path, exist_ok= True)


epochs = 2
size_weight = 0.1
offset_weight = 1


for epoch in range(epochs):
    
    progress_bar = tqdm(train_dataloader,
                    desc= f"Epoch : {epoch+1} / {epochs}")
    
    model.train()
    
    for i, (img, targets) in enumerate(progress_bar):
        
        img= img.to(device)
        
        for k, v in targets.items():
            
            targets[k] = v.to(device)
           
        with torch.amp.autocast(device_type=device.type):
            
            outputs = model(img)
            
            heat_pred= outputs["heatmap"]
            heat_tar= targets["heatmap"]
            
            heat_loss = focal_loss(heat_pred, heat_tar)
            
            size_pred = outputs["size"]
            size_tar = targets["size"]
            
            size_loss= reg_l1_loss(size_pred, size_tar, targets["mask"])
            
            offset_pred = outputs["offset"]
            offset_tar = targets["offset"]
                        
            offset_loss= reg_l1_loss(offset_pred, offset_tar, targets["mask"])
            
            total_loss = offset_weight * offset_loss + heat_loss + size_weight * size_loss
            
        optimizer.zero_grad()
        
        scaler.scale(total_loss).backward()
        
        scaler.step(optimizer)
        
        scaler.update()
        
        progress_bar.set_postfix(heat = heat_loss.item(), 
                                 size = size_loss.item(),
                                 offset = offset_loss.item(),
                                 total = total_loss.item())
        
    checkpoint_file = os.path.join(checkpoint_path, f"epoch_{epoch+1}.pt")
    torch.save({"state_dict" : model.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "total_loss" : total_loss.item()}, checkpoint_file)
    
 
        
        
    
    
        
        
        
        
        
        
        
        
        
        
            
            
            
            
            
            
            
           
           
            
            
        
        
        
        
        
        










