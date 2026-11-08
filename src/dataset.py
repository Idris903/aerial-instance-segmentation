import os 
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


from tiling import group_annotations
from targets import build_targets
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms.functional import convert_image_dtype

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
        img = convert_image_dtype(pil_to_tensor(img), dtype= torch.float32)
    
        tile_id  = self.tiles[index]["id"]
        
        target_anns = self.annotations_by_tile.get(tile_id, [])
            
        np_targets = build_targets(target_anns)
        
        tensor_targets = []
        for target in np_targets:
            tensor_targets.append(torch.from_numpy(target))
            
        return img, {"heatmap" : tensor_targets[0], "size" : tensor_targets[1], "offset" : tensor_targets[2], "mask" : tensor_targets[3]}
            
            
            
            
        
        
    
    
    

     
    
    