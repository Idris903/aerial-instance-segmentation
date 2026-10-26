import os
import json
from PIL import Image

def tile_start(length, tile  = 512, stride = 384):
    
    if length <= tile:
        return [0]
    
    
    starts = []
    
    start = 0
    
    while start + tile  <= length:
        
        starts.append(start)
        
        start = start + stride
        
    if starts[-1] + tile < length:
                    
        starts.append(length - tile)
                    
           
    
    return starts
    
        
def tile_corner(width, height):

    corners = []
   
    x_starts = tile_start(width)
    y_starts = tile_start(height)
    
    for y in y_starts:
        for x in x_starts:
            corners.append((x, y))
        
    
    
    
    return corners
   

     
def shift_box(bbox, corner):
    
    
    x, y, w, h = bbox
    
    tile_x, tile_y = corner
    
    shift_x = x - tile_x
    
    shift_y = y - tile_y
    
    return [shift_x, shift_y, w, h] 
        
        
def clip_box(bbox, tile = 512 ):
    """Clips a box's edges that are outside of a tile. 
    This takes place after the box has been converted to tile coordinates.
    Returns the clipped box as [x, y, w, h], or None if no part of the box is in the tile """
    
    x, y, w, h = bbox
    
    right = x + w
    bottom = y + h
    
    clip_x = max(x, 0)
    clip_y = max(y, 0)
    clip_right = min(right, tile)
    clip_bottom = min(bottom,tile)
    
   
    if clip_right <= clip_x or clip_bottom <=clip_y:
        return None
    
    return ([clip_x, clip_y, clip_right - clip_x, clip_bottom - clip_y])
    

def keep_box(original, clipped, min_fraction=0.5):
    
    _, _, orig_w, orig_h = original
    
    if clipped is None:
        return False
    
    else:
        _, _, clip_w, clip_h = clipped
    
    clip_area = clip_w * clip_h
    orig_area = orig_w * orig_h
    
    if clip_area < orig_area * min_fraction:
        
        return False
    else:
        return True
    
   
def group_annotations(coco):
    
    annotations_by_image = {}
    
    for annotation in coco["annotations"]:
        image_id = annotation["image_id"]
        
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
            
        annotations_by_image[image_id].append(annotation)
    
    return annotations_by_image

         
def tile_annotations(image_annotations, corner, tile = 512):
    final_bbox = []
    
    
    for annotation in image_annotations:
        
        bbox = annotation["bbox"]
        
        shifted_bbox = shift_box(bbox, corner)
        
        clip_bbox = clip_box(shifted_bbox, tile) 
        
        if keep_box(shifted_bbox, clip_bbox):
            final_bbox.append({"bbox": clip_bbox, "category_id": annotation["category_id"]})
            
    return final_bbox
        
         
def crop_tile(image, corner, tile = 512):
   
    x, y = corner
    
    bottom = y + tile 
    right = x + tile
    
    return image.crop((x, y, right, bottom))
    
    
def tile_split(image_ids, coco, annotations_by_image, image_folder, out_folder, skip_empty = True):
    
    output = {"images": [], "annotations": [], "categories" : coco["categories"]}

    id_counter = 0
    annotation_counter = 0

    os.makedirs(out_folder, exist_ok= True)
    
    look_up = {}
    
    for img in coco["images"]:
        
        look_up[img["id"]] = img
        
    for image_id in image_ids:
        
        info = look_up[image_id] 
    
        test_path = os.path.join(image_folder, info["file_name"])
        test_img = Image.open(test_path).convert("RGB")

        w, h = test_img.size

        test_img_corners = tile_corner(w, h)

        test_img_anno = annotations_by_image.get(info["id"], [])

        base_name = os.path.splitext(info["file_name"])[0]

        for corner in test_img_corners:
            
            
            boxes = tile_annotations(test_img_anno, corner)
            
            if skip_empty and len(boxes) == 0:
                continue

            tile_image = crop_tile(test_img, corner)
            
            x, y = corner
            
            tile_name = f"{base_name}_{x}_{y}.png" 
            
            tile_path = os.path.join(out_folder, tile_name)
            
            tile_image.save(tile_path)
            
            output["images"].append({"id" : id_counter, 
                                "file_name": tile_name, 
                                "width" : 512,
                                "height" : 512})
    
    
            for box in boxes:
                output["annotations"].append({"id": annotation_counter,
                                                "image_id": id_counter,
                                                "category_id": box["category_id"],
                                                "bbox": box["bbox"],
                                                "area" : box["bbox"][2] * box["bbox"][3],
                                                "iscrowd": 0})
                annotation_counter += 1     
        
            id_counter += 1  
        
    json_path = os.path.join(out_folder, "annotations.json")
    with open(json_path, "w") as f:
        json.dump(output, f)
    return output


def get_img_id(coco):
    
    image_ids = []
    
    for image in coco["images"]:
    
        image_ids.append(image["id"])
    
    return image_ids 


  

            
        

         
    
    
    
    
     
    
        
        
  
   
    
    
    
    



    







        
    
     
        
         