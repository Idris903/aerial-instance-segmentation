import numpy as np

def box_to_center(bbox, stride = 4 ):
    
    x, y, w, h = bbox
    
    center_x = (x + (w/2)) / stride
    center_y = (y + (h/2)) / stride
    center_cell = (int(center_x), int(center_y))
    

    offset= (center_x - center_cell[0], center_y - center_cell[1])
    
    cell_size_x = w /stride
    cell_size_y = h / stride
    
    size = (cell_size_x, cell_size_y)
    
    
    
    return center_cell, offset, size
    
def draw_gaussian(heatmap_channel, cell, sigma, grid = 128):

    cx, cy = cell
    radius = int(3 * sigma) + 1

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            row = dy + cy
            column = dx + cx 
            
            if row >= grid or row < 0   or column >= grid  or column < 0:
                continue
            
            value = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
            
            heatmap_channel[row, column] = max(heatmap_channel[row, column], value)
            


def build_targets(annotaitons, num_classes = 15, grid = 128, stride = 4):
    
    heatmap = np.zeros((num_classes,grid,grid), dtype= np.float32)
    offset_map = np.zeros((2, grid,grid), dtype= np.float32)
    size_map = np.zeros((2, grid, grid), dtype= np.float32)
    mask = np.zeros((grid,grid), dtype= np.float32)
    
    for annotation in annotaitons:
        bbox = annotation["bbox"] 
        cat_id = annotation["category_id"]
        
        cell, offset_size, size = box_to_center(bbox, stride)
        cx, cy = cell
        
        if cx >= grid or cy >= grid or cx <0 or cy < 0:
            continue
        
        sigma = max(min(size) / 6, 0.5)
        draw_gaussian(heatmap[cat_id - 1], cell, sigma)
        
        mask[cy, cx]= 1
        size_map[0, cy, cx] = size[0]
        size_map[1, cy, cx] = size[1]
        offset_map[0, cy,cx] = offset_size[0]
        offset_map[1, cy, cx] = offset_size[1]
        
    return heatmap, size_map, offset_map, mask


        
    
    
    
    
        
    
            
            
    