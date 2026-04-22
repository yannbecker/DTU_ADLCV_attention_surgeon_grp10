import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np


def colormap_to_class_indices(mask, colormap):
    """
    Convert a color-coded segmentation mask to class indices.
    
    Args:
        mask (PIL Image or np.array): The input segmentation mask with RGB colors.
        colormap (np.array): An array of shape (num_classes, 3) where each row is an RGB color corresponding to a class index.
    
    Returns:
        np.array: A 2D array of shape (H, W) with class indices.
    """
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    
    # Initialize an array for class indices
    class_indices = np.zeros(mask.shape[0:2], dtype=np.int64)
    
    # For each color in the colormap, create a boolean mask and assign the corresponding class index
    for idx, color in enumerate(colormap):
        matches = np.all(mask == color, axis=-1)
        class_indices[matches] = idx
    
    return class_indices

class ADE20KMinimalDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=(224, 224)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size
        # On trie pour que img1.jpg corresponde bien à img1.png
        self.img_names = sorted(os.listdir(img_dir))
        self.mask_names = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # 1. Chargement
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])
        
        image = Image.open(img_path).convert("RGB")
        # ADE20K utilise des masques .png où chaque pixel est l'ID de la classe
        #mask = Image.open(mask_path)
        
        mask = Image.open(mask_path) # Mode L

        # 2. Transformations communes (Redimensionnement)
        # Note : Pour un ViT, 'size' doit souvent être (224, 224) ou (384, 384)
        image = TF.resize(image, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)

        # 3. Conversion en tenseurs
        image = TF.to_tensor(image) # Normalise entre [0, 1]
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        
        # Le masque ne doit PAS être normalisé, on veut les IDs bruts
        # On convertit en long pour la CrossEntropyLoss
        #mask = colormap_to_class_indices(mask, ade20k_palette)
        mask = torch.from_numpy(np.array(mask)).long()
        mask = mask - 1 # classes from 0 to 149

        return image, mask

ade20k_palette = np.array([
    [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
    [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
    [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
    [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
    [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
    [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
    [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
    [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
    [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
    [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
    [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
    [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
    [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
    [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
    [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
    [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
    [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
    [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
    [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
    [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
    [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
    [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
    [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
    [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
    [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
    [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
    [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
    [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
    [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
    [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
    [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
    [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
    [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
    [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
    [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
    [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
    [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
    [102, 255, 0], [92, 0, 255]
])

# --- Utilisation ---

if __name__ == "__main__":
    mode = "test"
    if mode == "test":
        current_dir = "Final_Project/DTU_ADLCV_attention_surgeon_grp10/DPT_segmentation"
        print(f"Current working directory: {current_dir}")
        img_dir = os.path.join(current_dir, 'data/images')
        mask_dir = os.path.join(current_dir, 'data/annotations')
        dataset = ADE20KMinimalDataset(img_dir=img_dir, mask_dir=mask_dir)
        dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
        image, mask = dataset.__getitem__(98)
        print(mask.shape)  # Devrait être (H, W) avec des valeurs d'indices de classe
        print(mask)  # Devrait montrer les indices de classe présents dans le masque
        print(f"Unique class indices in the mask: {torch.unique(mask)}")  # Devrait montrer les indices de classe uniques présents dans le masque


    if mode == "affichage":
        current_dir = "Final_Project/DTU_ADLCV_attention_surgeon_grp10/DPT_segmentation"
        img_dir = os.path.join(current_dir, 'data/images')
        mask_dir = os.path.join(current_dir, 'data/annotations')
        dataset = ADE20KMinimalDataset(img_dir=img_dir, mask_dir=mask_dir)
        
    
        # Display size of images and masks in the first batch
        images, masks = next(iter(dataloader))
        
        print(f"Batch of images shape: {images.shape}")  # Should be [B, 3, H, W]
        print(f"Batch of masks shape: {masks.shape}")    # Should be [B, H, W]

        # plotting a batch of images and masks

        def plot_batch(images, masks):
            batch_size = images.size(0)
            fig, axes = plt.subplots(batch_size, 2, figsize=(10, 5 * batch_size))
            
            for i in range(batch_size):
                img = images[i].permute(1, 2, 0).numpy()  # Convert to HWC
                mask = masks[i].numpy()  # Mask is already in HWC format
                
                axes[i, 0].imshow(img)
                axes[i, 0].set_title("Image")
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(mask)
                axes[i, 1].set_title("Mask")
                axes[i, 1].axis('off')
            
            plt.tight_layout()
            plt.show()

        plot_batch(*next(iter(dataloader)))