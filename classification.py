import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from Final_Project.DTU_ADLCV_attention_surgeon_grp10.DPT_segmentation.utils import load_model  # Adjusted import to your structure
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
import argparse


class DinoClassifier(nn.Module):
    """
    Linear probing model: Frozen DINOv2 backbone + Trainable Linear Head.
    """

    def __init__(self, device, num_classes=10):
        super(DinoClassifier, self).__init__()
        self.transformer = load_model(device)

        # Freeze the backbone for linear probing [cite: 40, 44]
        for param in self.transformer.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(768, num_classes)

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
            layer = self.transformer.blocks[i].attn
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

    def forward(self, x):
        # DINOv2 returns the CLS token by default in this configuration
        features = self.transformer(x)
        logits = self.classifier(features)
        return logits


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    return running_loss / len(loader), accuracy


def get_loaders(data_dir, batch_size, num_workers):
    base_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=base_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=base_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


def main(args):
    device = (
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # ------------------- DATASET SETUP
    base_transform = transforms.Compose(
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
    model = DinoClassifier(device, num_classes=10).to(device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)

    train_losses, val_losses = [], []

    # ------------------- TRAINING LOOP
    for epoch in range(args.epochs):
        t_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = validate(model, test_loader, criterion, device)

        train_losses.append(t_loss)
        val_losses.append(v_loss)

        print(
            f"Epoch {epoch+1}/{args.epochs} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.2f}%"
        )

        # Save Checkpoint
        checkpoint_path = os.path.join(
            args.checkpoint_dir, f"dino_classifier_latest.pth"
        )
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_acc": v_acc,
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
        description="AttentionSurgeon: DINOv2 Linear Probing for Classification [cite: 31]"
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
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")

    args = parser.parse_args()
    main(args)
