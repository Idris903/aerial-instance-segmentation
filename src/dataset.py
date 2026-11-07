import os 
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


from tiling import group_annotations
from targets import build_targets
from torchvision.transforms.functional import pil_to_tensor

class Tile_Dataset(Dataset):
    def __init__(self, folder):
        super().__init__()
        
        self.folder = folder
        
        with open(os.path.join(folder, "annotations.json")) as f:
            coco = json.load(f)
            
        self.tiles = coco["images"]
        
        self.annotations_by_tile = group_annotations(coco)
    
    def __len__(self):
        return len(self.tiles)
        
    def __getitem__(self, index):
        
        path = os.path.join(self.folder, self.tiles[index]["file_name"])
        img= Image.open(path).convert("RGB")
        return pil_to_tensor(img) 
    
    