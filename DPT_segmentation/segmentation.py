# From classification.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from utils import load_model  # Adjusted import to your structure
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
import argparse

# From DPT-paper repository
from models import DPTSegmentationModel
from torchmetrics.classification import MulticlassJaccardIndex


class DinoSegmenter(DPTSegmentationModel):
    
    def __init__(self, device):

        # Freeze the backbone for linear probing [cite: 40, 44]
        for param in self.pretrained.model.parameters():
            param.requires_grad = False

        # Equivalent de self.classifier = nn.Linear(768, num_classes)
        self.segmenter = None # A MODIFIER

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


def train_one_epoch(model, loader, criterion, optimizer, device, gamma = 0.1, with_loss_aux = False):
    model.train()
    running_loss = 0.0
    
    for images, masks in tqdm(loader, desc="Training", leave=False): # A MODIFIER -> ADE20K
        images, masks = images.to(device), masks.to(device)

        path2, outputs = model(images) # model = DPTSegmenter
        aux_outputs = model.auxlayer(path2) 
        loss = criterion(outputs, masks) + (gamma*criterion(aux_outputs, masks) if with_loss_aux else 0) # A MODIFIER / VERIFIER
        

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
        miou_metric,  # A MODIFIER -> Initialiser un metric MulticlassJaccardIndex de torchmetrics
        gamma=0.1, 
        with_loss_aux=False
             ):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validating", leave=False):
            images, masks = images.to(device), masks.to(device)
            path2, outputs = model(images)
            loss = criterion(outputs, masks) + (gamma * criterion(model.auxlayer(path2), masks) if with_loss_aux else 0) # A MODIFIER / VERIFIER
            running_loss += loss.item()

            # Update of the mIoU metric
            # outputs: [B, C, H, W], masks: [B, H, W]
            miou_metric.update(outputs, masks)

    Final_mIoU = miou_metric.compute().item() # A MODIFIER / VERIFIER
    return running_loss / len(loader), Final_mIoU


def get_loaders(data_dir, batch_size, num_workers):
    base_transform = transforms.Compose( # A MODIFIER -> Adapter à ADE2OK
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    train_set = torchvision.datasets.CIFAR10( # A MODIFIER -> Adapter à ADE2OK
        root=data_dir, train=True, download=True, transform=base_transform
    )
    test_set = torchvision.datasets.CIFAR10( # A MODIFIER -> Adapter à ADE2OK
        root=data_dir, train=False, download=True, transform=base_transform
    )

    train_loader = DataLoader( # A MODIFIER -> Adapter à ADE2OK
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader( # A MODIFIER -> Adapter à ADE2OK
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader





def main(args):
    device = (
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # ------------------- DATASET SETUP
    base_transform = transforms.Compose( # A MODIFIER -> Adapter à ADE2OK
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    train_loader, test_loader = get_loaders(
        args.data_dir, args.batch_size, args.num_workers
    )

    # ------------------- MODEL SETUP
    model = DinoClassifier(device, num_classes=10).to(device) # A MODIFIER -> DinoSegmenter
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss() # A MODIFIER -> Vérifier que c'est bien la CE loss
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr) # A MODIFIER -> Est ce qu'on utilise Adam

    train_losses, val_losses = [], []

    # ------------------- TRAINING LOOP
    for epoch in range(args.epochs):
        t_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = validate(model, test_loader, criterion, device) # A MODIFIER -> vmIoU

        train_losses.append(t_loss)
        val_losses.append(v_loss)

        print(
            f"Epoch {epoch+1}/{args.epochs} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.2f}%" # A MODIFIER -> Val mIoU
        )

        # Save Checkpoint
        checkpoint_path = os.path.join(
            args.checkpoint_dir, f"dino_classifier_latest.pth" # A MODIFIER -> DinoSegmenter
        )
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_acc": v_acc, # A MODIFIER -> mIoU
            },
            checkpoint_path,
        )

    # ------------------- VISUALIZATION
    os.makedirs("figure", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, args.epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join("figure", "train_val_curve.jpg"))
    print("Training curve saved to figure/train_val_curve.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AttentionSurgeon: DINOv2 Linear Probing for Classification [cite: 31]" # A MODIFIER -> Segmentation
    )

    # Path Arguments
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Path to dataset"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Path to save weights",
    )

    # Hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for linear head"
    )

    # Hardware
    parser.add_argument(
        "--device", type=str, default=None, help="Force device (e.g., cuda, mps, cpu)"
    )
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers") # A MODIFIER -> Workers ?

    args = parser.parse_args()
    main(args)
