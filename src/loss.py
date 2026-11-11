import torch

def reg_l1_loss(pred, target, mask):
    
    pos_loss = torch.abs(pred-target)
    
    unsqueeze_mask = mask.unsqueeze(1)
    
    true_loss = pos_loss * unsqueeze_mask
    
    total_loss = true_loss.sum()
    
    return total_loss / (unsqueeze_mask.sum() + 1e-4)
    
     
    
    
    