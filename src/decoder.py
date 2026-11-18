import torch


def decoder(heatmap):
    
    pred = torch.sigmoid(heatmap)
    
    maxed_pred = torch.nn.functional.max_pool2d(pred, kernel_size= 3,
                                                padding= 1,
                                                stride=1
                                                )
    
    center = pred == maxed_pred
    
    return center.float() * pred

def top_k_peaks(peak, K=100):
    
    flat_peak = torch.flatten(peak, start_dim=1, end_dim= -1)
    
    return torch.topk(flat_peak, K, dim = 1, largest = True, sorted = True )

def make_dims(top_k_score, top_k_postion, grid_size = 128*128, row_size = 128):

    class_num = top_k_postion // grid_size
    row = (top_k_postion % grid_size) // row_size
    column = (top_k_postion % grid_size) % row_size  
    
    return class_num, row, column




def make_box(size, offset, col, row, score, class_num, stride = 4):
    
    tile_value = []
    for img in range(size.shape[0]):
            
        width = size[img, 0, row[img], col[img]]
        height = size[img, 1, row[img],col[img]]
        
        off_x = offset[img,0,row[img],col[img]]
        off_y = offset[img,1,row[img],col[img]]
        
        cell_x = col[img] + off_x
        cell_y = row[img] + off_y
        
        center = [cell_x , cell_y]
        
        peak_size = [width, height]
        
        scaled_size = [i * stride for i in peak_size]
        
        scaled_center = [i * stride for i in center]
        
        top_left_x = scaled_center[0] - scaled_size[0] / 2
        top_left_y = scaled_center[1] - scaled_size[1] /2
            
        bbox = [top_left_x , top_left_y, scaled_size[0], scaled_size[1]]
        
        bboxes = torch.stack(bbox, dim= 1)
        
        tile_value.append({"bbox" : bboxes,
                           "class": class_num[img] + 1,
                           "scores": score[img]})
        
    return tile_value


def whole_decode(heat, size, offset, k = 100):
    
    peak_map = decoder(heat)
    
    topk_scores, top_postions = top_k_peaks(peak_map, k)
    
    class_num, row, col = make_dims(topk_scores, top_postions)
    
    return make_box(size= size, offset= offset, col= col, row= row, score= topk_scores, class_num= class_num)
    
    
    
    
    
    
                           
       
        
        
        
        
        
        
            
      

   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


    
    
    
    
    
    
    
    




    
    
    