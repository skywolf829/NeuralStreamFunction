import torch
import torch.nn.functional as F

def l1(x, y):
    return F.l1_loss(x, y)

def mse(x, y):
    return F.mse_loss(x, y)

def l1_occupancy(gt, y):
    # Expects x to be [..., 3] or [..., 4] for (u, v, o) or (u, v, w, o)
    # Where o is occupancy
    is_nan_mask = torch.isnan(gt)[...,0].detach()
    
    o_loss = l1((~is_nan_mask).to(torch.float32).detach(), y[..., -1])
    vf_loss = l1(gt[~is_nan_mask, :].detach(), y[~is_nan_mask, 0:-1])
    return o_loss + vf_loss

def angle_same_loss(x, y):
    angles = (1 - F.cosine_similarity(x, y))
    mask = (y.norm(dim=1) != 0).type(torch.float32).detach()
    weighted_angles = angles * mask
    return weighted_angles.mean()

def angle_parallel_loss(x, y):
    angles = (1 - F.cosine_similarity(x, y)**2)
    mask = (y.norm(dim=1) != 0).type(torch.float32).detach()
    weighted_angles = angles * mask
    return weighted_angles.mean()

def angle_orthogonal_loss(x, y):
    angles = (F.cosine_similarity(x, y)**2)
    mask = (y.norm(dim=1) != 0).type(torch.float32).detach()
    weighted_angles = angles * mask
    return weighted_angles.mean()

def magangle_orthogonal_loss(x, y):
    mags = F.mse_loss(torch.norm(x,dim=1), torch.norm(y,dim=1))
    angles = (F.cosine_similarity(x, y)**2)
    mask = (y.norm(dim=1) != 0).type(torch.float32).detach()
    weighted_angles = angles * mask
    return 0.9*mags + 0.1*weighted_angles.mean()

def magangle_parallel_loss(x, y):
    x_norm = torch.norm(x,dim=1)
    y_norm = torch.norm(y,dim=1)
    mags = F.mse_loss(x_norm, y_norm)
    angles = (1 - F.cosine_similarity(x, y)**2)
    mask = (y.norm(dim=1) != 0).type(torch.float32).detach()
    weighted_angles = angles * mask
    return 0.9*mags + 0.1*weighted_angles.mean()

def magangle_same_loss(x, y):
    x_norm = torch.norm(x,dim=1)
    y_norm = torch.norm(y,dim=1)
    mags = F.mse_loss(x_norm, y_norm)
    angles = (1 - F.cosine_similarity(x, y))
    mask = (y.norm(dim=1) != 0).type(torch.float32).detach()
    weighted_angles = angles * mask
    return 0.9*mags + 0.1*weighted_angles.mean()
