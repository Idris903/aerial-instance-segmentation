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
    
    for img in range(size.shape[0]):
            
        s_width = size[img, 0, row[img], col[img]]
        s_height = size[img, 1, row[img],col[img]]
        
        off_x = offset[img,0,row[img],col[img]]
        off_y = offset[img,1,row[img],col[img]]
        
        cell_x = col[img] + off_x
        cell_y = row[img] + off_y
        
        center = [cell_x , cell_y]
        
        peak_size = [s_width, s_height]
        
        scaled_size = [i * stride for i in peak_size]
        
        scaled_center = [i * stride for i in center]
        
        top_left_x = scaled_center[0] - scaled_size[0] / 2
        top_left_y = scaled_center[1] - scaled_size[1] /2
        
    bbox = [top_left_x , top_left_y, scaled_size[0], scaled_size[1]]
        
    return {"bbox" : bbox[img],
            "class" : class_num[img] + 1,
            "score" : score[img]}

   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


    
    
    
    
    
    
    
    




    
    
    