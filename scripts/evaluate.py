import torch
import sys 
import os
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src"))

from decoder import whole_decode
from dataset import Tile_Dataset
from model import Detector
from tqdm import tqdm


device = torch.device("cuda")

hold_out_dataset = Tile_Dataset(os.path.expanduser("~/data/tiles/val"))
hold_out_dataloader = DataLoader(hold_out_dataset,
                                 shuffle= False,
                                 batch_size= 8,
                                 num_workers= 2,
                                 persistent_workers= True)

model = Detector(3, 15).to(device= device)

checkpoint_file = os.path.join(os.path.expanduser("~/checkpoint"), f"epoch_20.pt")
model_checkpoint = torch.load(checkpoint_file, weights_only= False)
model.load_state_dict(model_checkpoint["state_dict"])

hold_path = os.path.expanduser("~/data/tiles/holdout/annotations.json")
val_path = os.path.expanduser("~/data/tiles/val/annotations.json")

hold_categories = {}
val_categories = {}

with open(hold_path) as f:
    hold = json.load(f)
    
with open(val_path) as f:
    val = json.load(f)
    
for category in hold["categories"]:
    
    hold_categories[category["id"]] = category["name"]
    
for category in val["categories"]:
    
    val_categories[category["name"]] = category["id"]

model.eval()


progress_bar_holdout = tqdm(hold_out_dataloader)


coco_boxes = []
with torch.inference_mode():
    
    for  img, target in progress_bar_holdout:
        
        with torch.amp.autocast(device_type= device.type):
           
            img = img.to(device)
            
            outputs = model(img)
            
            img_boxes = whole_decode(heat= outputs["heatmap"], size= outputs["size"], offset= outputs["offset"])
            
            tile_ids  = target["tile_id"]
            
            for i, tile_boxes in enumerate(img_boxes):
                
                for j in range(tile_boxes["bbox"].shape[0]):
                    
                    coco_boxes.append({"bbox" : tile_boxes["bbox"][j].tolist(), 
                                        "category_id" : val_categories[hold_categories[tile_boxes["class"][j].item()]],
                                        "score" : tile_boxes["scores"][j].item(),
                                        "image_id" :  tile_ids[i].item()})
                    


ground_truth = COCO(val_path)

pred = ground_truth.loadRes(coco_boxes)

scorer = COCOeval(ground_truth, pred, "bbox")
   

scorer.evaluate()
scorer.accumulate()
scorer.summarize()
                    
                
                
                
            
            

