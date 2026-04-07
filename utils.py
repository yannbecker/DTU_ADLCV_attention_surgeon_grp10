import torch 

def load_model(device="cpu"):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device) # Dinov2 ViT-B/14 (16 does not exist)
    return model
