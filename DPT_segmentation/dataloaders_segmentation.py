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
        # On trie pour que img1.jpg corresponde bien à img1.png
        self.img_names = sorted(os.listdir(img_dir))
        self.mask_names = sorted(os.listdir(mask_dir))
        self.return_name = True

    def set_return_name(self, b : bool):
        self.return_name = b

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # 1. Chargement
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])
        
        image = Image.open(img_path).convert("RGB")
        # ADE20K utilise des masques .png où chaque pixel est l'ID de la classe
        #mask = Image.open(mask_path)
        
        mask = Image.open(mask_path).convert("L") # Mode L

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
            

    # create new data : data/train/feature_images, data/validation/feature_images
    # Load train_dataset, val_dataset ADE20K
    # Put train_dataset through vit -> forward_vit(images) for images in dataset
    # Save Layer1 - Layer4 for every image
    # Same for validation





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


if __name__ == "__main__":


    # A MODIFIER : ajouter argparse
    device = "mps"
    mode = "affichage"


    if mode == "test_mask":
        current_dir = "Final_Project/DTU_ADLCV_attention_surgeon_grp10/DPT_segmentation"
        print(f"Current working directory: {current_dir}")
        img_dir = os.path.join(current_dir, 'data/feature_images')
        mask_dir = os.path.join(current_dir, 'data/annotations')
        dataset = ADE20KFeatureDataset(feature_dir=img_dir, mask_dir=mask_dir)
        #dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
        image, mask = dataset.__getitem__(98)
        print(f"Image features: {[f.shape for f in image]}")  # Show form of the features from vit



    if mode == "affichage":
        current_dir = "Final_Project/DTU_ADLCV_attention_surgeon_grp10/DPT_segmentation"
        img_dir = os.path.join(current_dir, 'data/images')
        mask_dir = os.path.join(current_dir, 'data/annotations')
        dataset = ADE20KDataset(img_dir=img_dir, mask_dir=mask_dir)
        dataloader = DataLoader(dataset, batch_size=12, shuffle=False)
        
    
        # Display size of images and masks in the first batch
        images, masks, _ = next(iter(dataloader))
        
        #print(f"Batch of images shape: {images.shape}")  # Should be [B, 3, H, W]
        #print(f"Batch of masks shape: {masks.shape}")    # Should be [B, H, W]
        #print(masks[0])  # Print the first mask tensor to check values (should be class indices)
        m0 = masks[0].unsqueeze(0)
        m1 = masks[1].unsqueeze(0)
        m2 =masks[2].unsqueeze(0)
        masque = (m1 == -1)
        m1[masque] = 1000
        from torchmetrics.classification import MulticlassJaccardIndex
        mIoU_metric = MulticlassJaccardIndex(num_classes=150, ignore_index=-1)
        for i in range(10):
            mIoU_metric.update(m1,m0)
            mIoU_metric.update(m1,m0)
            mIoU_metric.update(m1,m0)
            mIoU_metric.update(m1,m0)
            mIoU_metric.update(m1,m0)
            mIoU_metric.update(m1,m2)
        print(mIoU_metric.compute().item())
        
        
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

        #plot_batch(*next(iter(dataloader)))