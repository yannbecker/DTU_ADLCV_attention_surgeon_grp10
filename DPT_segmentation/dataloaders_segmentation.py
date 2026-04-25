import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from vit import forward_vit



class ADE20KDataset(Dataset):
    def __init__(self, img_dir, mask_dir, return_name = True, size=(224, 224)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size

        self.img_names = sorted(os.listdir(img_dir))
        self.mask_names = sorted(os.listdir(mask_dir))
        self.return_name = True

    def set_return_name(self, b : bool):
        self.return_name = b

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir, self.img_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])
        
        image = Image.open(img_path).convert("RGB")
    
        
        mask = Image.open(mask_path).convert("L") 

        image = TF.resize(image, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)

        image = TF.to_tensor(image)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        
        mask = torch.from_numpy(np.array(mask)).long()
        mask = mask - 1 # classes from 0 to 149. 
        
        if self.return_name:
            return image, mask, self.img_names[idx] 
        else :
            return image, mask
    
def ADE20K_through_ViT(model, dataloader, device, data_dir, process_type):
    " Get feature images from the original ADE20K dataset through ViT and preprocess masks to the right format "
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir,"feature_images"), exist_ok=True)
    os.makedirs(os.path.join(data_dir,"preprocessed_masks"), exist_ok=True)
    os.makedirs(os.path.join(data_dir,"feature_images",process_type), exist_ok=True)
    os.makedirs(os.path.join(data_dir,"preprocessed_masks",process_type), exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        for i, (images, mask, img_names) in enumerate(tqdm(dataloader)):
            # images: (B, 3, 224, 224)
            images = images.to(device)
            l1, l2, l3, l4 = forward_vit(model.pretrained,images)
            
            # iteration on batch
            for j in range(images.size(0)):
                feature_dict = {
                    "layer_1": l1[j].detach().cpu(),
                    "layer_2": l2[j].detach().cpu(),
                    "layer_3": l3[j].detach().cpu(),
                    "layer_4": l4[j].detach().cpu()
                }
                # We use the original name to name the file .pth
                save_path_img = os.path.join(data_dir, "feature_images",process_type,f"{img_names[j][:-4]}.pth")
                torch.save(feature_dict, save_path_img)
            
                save_path_mask = os.path.join(data_dir, "preprocessed_masks", process_type ,f"{img_names[j][:-4]}.pth")
                
                torch.save(mask[j], save_path_mask)
            


class ADE20KFeatureDataset(Dataset):
    def __init__(self, feature_images_dir, preprocessed_masks_dir, size = (224,224), return_name = False):
        self.feature_images_dir = feature_images_dir
        self.preprocessed_masks_dir = preprocessed_masks_dir
        self.filenames = [f.replace(".pth", "") for f in os.listdir(feature_images_dir)]
        self.size = size
        self.set_return_name = return_name

    def set_return_name(self, b : bool):
        self.return_name = b

    def __getitem__(self, idx):
        
        name = self.filenames[idx]
        # Load extracted features from vit
        features = torch.load(os.path.join(self.feature_images_dir, f"{name}.pth"))
        
        # Load mask
        mask = torch.load(os.path.join(self.preprocessed_masks_dir, f"{name}.pth"))

        return (features["layer_1"], features["layer_2"], features["layer_3"], features["layer_4"]), mask

    def __len__(self):
        return len(self.filenames)

