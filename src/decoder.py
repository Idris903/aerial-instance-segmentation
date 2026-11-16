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
    
    


    
    
    
    
    
    
    
    




    
    
    