import os
import json 
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tiling import tile_split, group_annotations, get_img_id



data_path = os.path.expanduser("~/data/isaid")

train_images = os.path.join(data_path, "train", "images")
train_json = os.path.join(data_path, "train", "Annotations", "iSAID_train.json")

val_images_folder = os.path.join(data_path, "val", "images")
val_json = os.path.join(data_path, "val", "Annotations", "iSAID_val.json")

out_path = os.path.expanduser("~/data/tiles")

with open(train_json) as f:
    coco = json.load(f)
    
train_annotations = group_annotations(coco)


image_ids = get_img_id(coco)

random.Random(42).shuffle(image_ids)

holdout_length =int(len(image_ids)/10)

holdout_ids = image_ids[:holdout_length]

train_ids = image_ids[holdout_length:]

print(len(holdout_ids))
print(len(train_ids))

train_out = tile_split(train_ids, coco, train_annotations, train_images, os.path.join(out_path, "train"))

holdout_out = tile_split(holdout_ids, coco, train_annotations, train_images, os.path.join(out_path, "holdout"), skip_empty = False)

with open(val_json) as f:
    val_coco = json.load(f)
    
val_annotations = group_annotations(val_coco)
val_ids = get_img_id(val_coco)

val_out = tile_split(val_ids, val_coco, val_annotations, val_images_folder, os.path.join(out_path, "val"), skip_empty = False)

print("train:", len(train_out["images"]), len(train_out["annotations"]))
print("holdout:", len(holdout_out["images"]), len(holdout_out["annotations"]))
print("val:", len(val_out["images"]), len(val_out["annotations"]))    
    




