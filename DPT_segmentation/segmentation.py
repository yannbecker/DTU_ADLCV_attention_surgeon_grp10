# From classification.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path

# From DPT-paper repository
from models import DPTSegmentationModel
from torchmetrics.classification import MulticlassJaccardIndex

# from dataloaders_segmentation.py
from dataloaders_segmentation import ADE20KDataset, ADE20K_through_ViT, ADE20KFeatureDataset


class DinoSegmenter(DPTSegmentationModel):
    """ DinoSegmenter: DINOv2 backbone + DPT segmentation head """
    def __init__(self, device, **kwargs): 

        super().__init__(**kwargs) 

        # Freeze the backbone for linear probing [cite: 40, 44]
        for param in self.pretrained.model.parameters():
            param.requires_grad = False

        # Internal state for the pruning mask (12 layers x 12 heads)
        self.mask = torch.ones(12, 12).to(device)
        self.hooks = []
        self._register_pruning_hooks()

    def _register_pruning_hooks(self):
        """
        Registers hooks to zero out head outputs during forward pass.
        ViT-B/14 has 768 dim / 12 heads = 64 dim per head.
        """

        def hook_fn(module, input, output, layer_idx):
            # The output of DINOv2 attention is (Batch, Tokens, 768)
            # We reshape the mask for this layer to (1, 1, 12, 1) -> (Batch, Tokens, Heads, HeadDim)
            # However, it's simpler to just zero out the 64-dim slices.
            mask_layer = self.mask[layer_idx]  # Shape: (12,)

            # Create a 768-dim binary mask for this specific layer
            # Each '1' or '0' in mask_layer is expanded to 64 dimensions
            full_mask = mask_layer.repeat_interleave(64).to(output.device)

            # Apply the mask to the output tensor
            return output * full_mask

        # Clear existing hooks if any
        for h in self.hooks:
            h.remove()
        self.hooks = []

        # Attach hooks to each of the 12 blocks [cite: 28]
        for i in range(12):
            # We hook the 'attn' layer's output
            layer = self.pretrained.model.blocks[i].attn
            # Use a closure to pass the layer index
            handle = layer.register_forward_hook(
                lambda mod, inp, out, idx=i: hook_fn(mod, inp, out, idx)
            )
            self.hooks.append(handle)

    def set_mask(self, mask_1d):
        """
        Updates the internal mask.
        mask_1d: tensor of shape (144,) provided by the RL Agent.
        """
        self.mask = mask_1d.view(12, 12)


def train_one_epoch(model, loader, criterion, optimizer, device, alpha = 0.2, with_loss_aux = True, features = True):
    model.train()
    running_loss = 0.0
    
    for images, masks in tqdm(loader, desc="Training", leave=False): 
        if features :
            images, masks = [l.to(device) for l in images], masks.to(device)
        else :
            images, masks = images.to(device), masks.to(device)
        path2, outputs = model(images, features = args.features) # model = DPTSegmenter, 
        # feature = True -> train directly from the output of the vit, feature = False -> train from RGB images that go through the vit
        
        aux_outputs = model.auxlayer(path2) 
        loss = criterion(outputs, masks) + (alpha*criterion(aux_outputs, masks) if with_loss_aux else 0) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)


def validate(
        model, 
        loader, 
        criterion, 
        device, 
        miou_metric,  
        alpha=0.1, 
        with_loss_aux=True,
        features = True # feature = True -> train directly from the output of the vit, feature = False -> train from RGB images that go through the vit
             ):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validating", leave=False):
            if features :
                images, masks = [l.to(device) for l in images], masks.to(device)
            else :
                images, masks = images.to(device), masks.to(device)
            path2, outputs = model(images, features = args.features)
            outputs_aux = model.auxlayer(path2)
            loss = criterion(outputs, masks) + (alpha * criterion(outputs_aux, masks) if with_loss_aux else 0) # A MODIFIER / VERIFIER
            # predict the class for each pixel and compute the loss
            outputs = torch.argmax(outputs, dim=1) 
            
            running_loss += loss.item()
            # Update of the mIoU metric
            # outputs: [B, H, W], masks: [B, H, W]
            miou_metric.update(outputs, masks)

    Final_mIoU = miou_metric.compute().item() 
    miou_metric.reset() # Reset the metric for the next epoch
    return running_loss / len(loader), Final_mIoU


def get_loaders(data_dir, batch_size, num_workers, use_features = True):

    if use_features: # Use new dataset with preprocessed masks and feature images
        feature_dir_train = os.path.join(data_dir, 'feature_images', "training")
        feature_dir_validation = os.path.join(data_dir, 'feature_images', "validation")
        mask_dir_train = os.path.join(data_dir, 'preprocessed_masks', "training")
        mask_dir_validation = os.path.join(data_dir, 'preprocessed_masks', "validation")
        dataset_train = ADE20KFeatureDataset(feature_images_dir=feature_dir_train, preprocessed_masks_dir=mask_dir_train)
        dataset_val = ADE20KFeatureDataset(feature_images_dir=feature_dir_validation, preprocessed_masks_dir=mask_dir_validation)

    else : # Use old dataset with raw masks and RGB images
        img_dir_train = os.path.join(data_dir, 'images', "training")
        img_dir_val = os.path.join(data_dir, 'images', "validation")
        mask_dir_train = os.path.join(data_dir, 'annotations', "training")
        mask_dir_val = os.path.join(data_dir, 'annotations', "validation")
        dataset_train = ADE20KDataset(img_dir=img_dir_train, mask_dir=mask_dir_train, size = (224,224))
        dataset_val = ADE20KDataset(img_dir=img_dir_val, mask_dir=mask_dir_val, size = (224,224))
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def main(args): # Training loop function

    # -------------------- DEVICE SETUP
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:        
        device = "cpu"
    
    print(f"Using device: {device}")

    # ------------------- DATASET SETUP

    train_loader, test_loader = get_loaders(
        args.data_dir, args.batch_size, args.num_workers, use_features=args.features
    )

    # ------------------- MODEL SETUP
    model = DinoSegmenter(device, num_classes=150).to(device) 
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss(ignore_index=-1) 
    # SGD + 0.9 momentum -> AdamW, lr = 1e-4
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    train_losses, val_losses, mIoUs = [], [], []
    mIoU_metric = MulticlassJaccardIndex(num_classes=150, ignore_index=-1).to(device) # A MODIFIER / VERIFIER -> Attention à l'indice des classes, 0 = background dans ADE20K
    # Early stopping variables
    patience = args.patience  
    best_miou = -1
    epochs_without_improvement = 0
    start_epoch = 0

    # ------------------- POTENTIAL RESUME FROM CHECKPOINT

    if args.resume_path and os.path.isfile(args.resume_path):
        print(f"Loading checkpoint: {args.resume_path}")
        checkpoint = torch.load(args.resume_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        
        if 'train_losses' in checkpoint:
            train_losses = checkpoint['train_losses']
            val_losses = checkpoint['val_losses']
            mIoUs = checkpoint['val_mIoU']
            best_miou = max(mIoUs) if mIoUs else -1
            
        print(f"Resuming from epoch {start_epoch}")

    # ------------------- TRAINING LOOP
    for epoch in range(start_epoch, args.epochs + start_epoch):

        print(f"Starting epoch {epoch+1}/{args.epochs + start_epoch}...")
        t_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, features = args.features)
        v_loss, v_mIoU = validate(model, test_loader, criterion, device, mIoU_metric, features=args.features) 

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        mIoUs.append(v_mIoU)

        print(
            f"Epoch {epoch+1}/{args.epochs + start_epoch} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Val mIoU: {100*v_mIoU:.4f}%" 
            
        )

        # Save Checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0 :
            checkpoint_path = os.path.join(
                args.checkpoint_dir, f"dino_segmenter_id{args.id}_ep{epoch+1}_{args.epochs + start_epoch}_bs{args.batch_size}_lr{args.lr}.pth" 
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mIoU": mIoUs, 
                    "train_losses": train_losses,
                    "val_losses" : val_losses
                },
                checkpoint_path,
            )

        # ------------------- EARLY STOPPING STRATEGY
        if v_mIoU > best_miou:
            best_miou = v_mIoU
            epochs_without_improvement = 0
            best_checkpoint_path = os.path.join(args.checkpoint_dir, f"best_model_{args.id}.pth")
            torch.save(model.state_dict(), best_checkpoint_path)
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            # End of the run
            print(f"Early stop : mIoU didn't increase for {patience} epochs !")
            # Save figure
            os.makedirs("figure", exist_ok=True)
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, args.epochs + 1), train_losses, label="Train Loss")
            plt.plot(range(1, args.epochs + 1), val_losses, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(os.path.join("figure", f"train_val_curve_{args.id}.jpg"))
            print("Training curve saved to figure/train_val_curve.jpg")
            break

    # ------------------- VISUALIZATION
    os.makedirs("figure", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, args.epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join("figure", f"train_val_curve_{args.id}.jpg"))
    print("Training curve saved to figure/train_val_curve.jpg")


if __name__ == "__main__":

    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent.parent.parent.parent

    parser = argparse.ArgumentParser(
        description="AttentionSurgeon: DINOv2 Linear Probing for Segmentation [cite: 31]" # A MODIFIER -> Segmentation
    )

    # Mode of functionment
    parser.add_argument("mode",type=str,choices = ["generate_feature_dataset", "train", "test_mask", "test", "inference"], help="Mode of operation") 
    
    # Id of training
    parser.add_argument("--id",type=str,help="Id of the training")
    
    # Path Arguments
    parser.add_argument(
        "--data_dir", type=str, default="/dtu/blackhole/04/223076/ADE20KFeatureDataset/data", help="Path to dataset"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/zhome/f0/d/223076/ADLCV/checkpoints",
        help="Path to save weights",
    )
    parser.add_argument(
    "--resume_path", type=str, default=None, help="Path to checkpoint to resume training from"
    )

    # Hyperparameters
    parser.add_argument(
        "--features", type=bool, default=True, help="Do we train directly from the output of the vit ? : True = Yes"
    )

    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=48, help="Batch size for training"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for linear head"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="How many epochs without improvement for early stopping"
    )

    # Hardware
    parser.add_argument(
        "--device", type=str, default="cuda", help="Force device (e.g., cuda, mps, cpu)"
    )
    parser.add_argument("--num_workers", type=int, default=10, help="DataLoader workers") 
    
    # Other
    parser.add_argument(
        "--process_type", type = str, default = None, choices = ['training', 'validation'], help = "training or validation")

    parser.add_argument(
        "--output_dir", type = str, default = None, help = "Output directory to store feature images and preprocessed masks"
    )
    parser.add_argument(
        "--checkpoint_interval", type = int, default = 10, help = "Interval to which save checkpoint weights"
    )



    args = parser.parse_args()

    if args.mode == "test":
        
        img_dir = args.data_dir + '/images'
        print("image directory : ", img_dir)
        mask_dir = args.data_dir + '/annotations'
        dataset = ADE20KDataset(img_dir=img_dir, mask_dir=mask_dir, return_name = True)
        dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
        for img, mask, _ in dataloader:
            print(mask.shape)

    if args.mode == "train":
        main(args)

    elif args.mode == "generate_feature_dataset" :
        # Generate the feature dataset by taking the output of ADE20K images through the ViT
        print("args.data_dir : ",args.data_dir)
        model = DinoSegmenter(args.device, num_classes=150).to(args.device) 
        img_dir = os.path.join(project_root, args.data_dir, 'images', args.process_type) 
        print("image directory : ", img_dir)
        mask_dir = os.path.join(project_root, args.data_dir, 'annotations', args.process_type) 
        dataset = ADE20KDataset(img_dir=img_dir, mask_dir=mask_dir, return_name = True)
        dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
        output_dir = os.path.join(project_root,args.output_dir)
        ADE20K_through_ViT(model = model, dataloader=dataloader, data_dir=output_dir, device = args.device, process_type=args.process_type)

    elif args.mode == "inference":
        # Inference on using the segmentation model
        current_directory = "Final_Project/DTU_ADLCV_attention_surgeon_grp10/DPT_segmentation"
        img_dir = os.path.join(current_directory,"data","images")
        mask_dir = os.path.join(current_directory,"data","annotations")
        feature_img_dir = os.path.join(current_directory,"data","feature_images")
        preprocessed_mask_dir = os.path.join(current_directory,"data","preprocessed_masks")
        model_path = os.path.join(current_directory,"models_segmentation","best_model_03.pth" )#"dino_segmenter_id03_ep30_100_bs32_lr0.0001.pth")

        dataset0 = ADE20KDataset(img_dir=img_dir, mask_dir=mask_dir, return_name = False)
        dataset1 = ADE20KFeatureDataset(feature_images_dir=feature_img_dir, preprocessed_masks_dir=preprocessed_mask_dir,return_name=False)
        dataloader0 = iter(DataLoader(dataset0, shuffle=False, batch_size=args.batch_size))
        dataloader1 = iter(DataLoader(dataset1, shuffle=False, batch_size=args.batch_size))

        for k in range(3): # change the range to get different images and the batch_size to vizualize more or less at the same time
            img, mask, _ = next(dataloader0)
            ft_img, prepro_mask = next(dataloader1)

        model = DinoSegmenter(args.device, num_classes=150).to(args.device)
        # model_weights = torch.load(model_path, map_location = torch.device('cpu'))["model_state_dict"] 
        model_weights = torch.load(model_path, map_location = torch.device('cpu'))
        model.load_state_dict(model_weights)

        _, outputs = model(ft_img, features = True)
        outputs = torch.argmax(outputs, dim=1)
        print("shape of output : ", outputs.shape)
        print("shape of mask : ", mask.shape)

        def plot_batch(images, masks, outputs, prepro_masks):
            # S'assurer qu'on traite bien le nombre d'images présentes
            actual_batch_size = images.size(0)
            
            # Correction de figsize : (largeur, hauteur) 
            # Pour 2 lignes, (15, 6) est généralement plus équilibré
            fig, axes = plt.subplots(2, actual_batch_size, figsize=(actual_batch_size * 3, 6), squeeze=False)
            
            for i in range(actual_batch_size):
                # 1. Extraction et conversion CPU/Numpy
                # Ground Truth
                gt = prepro_masks[i].detach().cpu().squeeze().numpy()
                
                # Inference : On prend l'Argmax sur la dimension des classes (souvent dim=0 ou dim=1)
                # Si output est (C, H, W), on fait argmax(0)
                res = outputs[i].detach().cpu()
                if res.ndim == 3: # Si (Classes, H, W)
                    res = res.argmax(dim=0)
                res = res.numpy()

                # 2. Affichage Ground Truth
                axes[0, i].imshow(gt)
                axes[0, i].set_title(f"GT {i+1}", fontsize=10)
                axes[0, i].axis('off')

                # 3. Affichage Inference
                # On utilise le même cmap que la GT pour pouvoir comparer
                axes[1, i].imshow(res)
                axes[1, i].set_title(f"Pred {i+1}", fontsize=10)
                axes[1, i].axis('off')
            
            plt.tight_layout()
            plt.show()

        plot_batch(img, mask, prepro_masks=prepro_mask,outputs= outputs)

        



        

# commands to run the script:

# ---- TRAINING ----
# python3 segmentation.py train --id 00 ...

# ---- GENERATE FEATURE DATASET ----
# python3 segmentation.py generate_feature_dataset --batch_size 128 --device cuda --num_workers 10 --process_type validation --output_dir <output_dir_path>

# ---- PERFORM INFERENCE ----
# python3 segmentation.py inference --batch_size 4