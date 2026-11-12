import torch

def reg_l1_loss(pred, target, mask):
    
    pos_loss = torch.abs(pred-target)
    
    unsqueeze_mask = mask.unsqueeze(1)
    
    true_loss = pos_loss * unsqueeze_mask
    
    total_loss = true_loss.sum()
    
    return total_loss / (unsqueeze_mask.sum() + 1e-4)

def focal_loss(logits, target, alpha = 2, beta = 4):
    
    pred_prob = torch.sigmoid(logits)
    clamped_pred = torch.clamp(pred_prob, min = 0.001, max = 0.999)
    
    positive_mask = (1 == target).float()
    negative_mask = (1 != target).float()
    
    positive_loss = positive_mask * (-torch.pow((1 - clamped_pred), alpha) * torch.log(clamped_pred))
    
    negative_loss = negative_mask * torch.pow(1 - target, beta) * torch.pow(clamped_pred, alpha) * -(torch.log(1-clamped_pred))
    
   
    
    total_loss = positive_loss.sum() + negative_loss.sum() 
    
    return total_loss / torch.clamp(positive_mask.sum(), min = 1)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
     
    
    
    