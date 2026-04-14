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
import re


class DinoClassifier(nn.Module):
    """
    Linear probing model: Frozen DINOv2 backbone + Trainable Linear Head.
    """

    def __init__(self, device, num_classes=10):
        super(DinoClassifier, self).__init__()
        self.transformer = load_model(device)
        self.num_heads = 12

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

    # Inside class DinoClassifier
    def set_heads_requires_grad(self, requires_grad=True):
        """
        Ensures that the outputs of the attention layers track gradients
        even if the weights themselves are frozen.
        """
        for layer_idx in range(12):
            # We target the attn layer output
            self.transformer.blocks[layer_idx].attn.requires_grad_(requires_grad)

    def get_taylor_importance(self, images, labels, criterion):
        self.eval() 
        taylor_scores = torch.zeros(12, 12).to(images.device)
        activations = {}
        grads = {}

        def save_activation(name):
            def hook(model, input, output):
                # CRITICAL FIX: The output of a frozen layer has requires_grad=False.
                # We force it to True so we can register a hook and compute Taylor importance.
                output.requires_grad_(True) 
                activations[name] = output.detach()
                output.register_hook(lambda g: grads.update({name: g}))
            return hook

        # 1. Attach temporary hooks
        temp_hooks = []
        for i in range(12):
            # Using the existing transformer blocks
            h = self.transformer.blocks[i].attn.register_forward_hook(save_activation(f"layer_{i}"))
            temp_hooks.append(h)

        # 2. Forward pass (Inside a gradient-enabled context)
        with torch.set_grad_enabled(True):
            logits = self.forward(images)
            loss = criterion(logits, labels)

            # 3. Backward pass
            self.zero_grad()
            loss.backward()

        # 4. Compute Taylor Importance: |Activation * Gradient|
        for i in range(12):
            if f"layer_{i}" in grads:
                act = activations[f"layer_{i}"]  # (B, N, 768)
                grad = grads[f"layer_{i}"]       # (B, N, 768)
                
                # Reshape to 12 heads
                act = act.view(act.size(0), act.size(1), 12, 64)
                grad = grad.view(grad.size(0), grad.size(1), 12, 64)
                
                # Sum over HeadDim (64) and average over Batch/Tokens
                score = torch.abs((act * grad).sum(dim=-1)).mean(dim=(0, 1))
                taylor_scores[i] = score

        # Clean up hooks
        for h in temp_hooks:
            h.remove()

        return taylor_scores
    
    def get_intra_layer_ranks(self, taylor_importance_matrix):
        """
        Computes the relative rank of each head within its own layer.
        Input: (12, 12) matrix of Taylor Importance scores.
        Output: (12, 12) matrix of normalized ranks [0, 1].
        """
        # taylor_importance_matrix shape: (Layers=12, Heads=12)
        
        # 1. Get ranks: argsort twice gives the rank (0 to 11)
        # Higher importance = Higher rank index
        ranks = torch.argsort(torch.argsort(taylor_importance_matrix, dim=1), dim=1).float()
        
        # 2. Normalize to [0, 1]
        # (Rank / (Num_Heads - 1)) -> 0.0 is the least important, 1.0 is the most important
        normalized_ranks = ranks / (self.num_heads - 1) 
        
        return normalized_ranks

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


def get_loaders(dataset_name, data_dir, batch_size, num_workers):
    base_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),  # Standard ImageNet cropping
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    if dataset_name == "cifar10":
        train_set = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=base_transform
        )
        test_set = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=base_transform
        )
        num_classes = 10

    elif dataset_name.startswith("imagenet"):
        # Dynamically extract the number of classes requested
        if dataset_name == "imagenet":
            num_classes = 1000
        else:
            match = re.search(r"imagenet(\d+)", dataset_name)
            if match:
                num_classes = int(match.group(1))
            else:
                raise ValueError(
                    f"Invalid dataset name format: {dataset_name}. Use 'imagenet' or 'imagenetX'."
                )

        # We ONLY look at the 'train' folder since the 'val' folder on the HPC is flat/unstructured
        hpc_train = os.path.join(data_dir, "ILSVRC/Data/CLS-LOC/train")
        std_train = os.path.join(data_dir, "train")

        if os.path.exists(hpc_train):
            train_dir = hpc_train
        elif os.path.exists(std_train):
            train_dir = std_train
        else:
            raise FileNotFoundError(
                f"Could not find structured train folder in {data_dir}."
            )

        print(f"Loading structured ImageNet from {train_dir}...")
        full_dataset = torchvision.datasets.ImageFolder(
            train_dir, transform=base_transform
        )

        # Subsetting logic: Keep only the first `num_classes`
        if num_classes < 1000:
            print(f"Subsetting ImageNet to the first {num_classes} classes...")
            valid_indices = set(range(num_classes))

            full_dataset.samples = [
                s for s in full_dataset.samples if s[1] in valid_indices
            ]
            full_dataset.targets = [
                t for t in full_dataset.targets if t in valid_indices
            ]
            full_dataset.classes = full_dataset.classes[:num_classes]

        # Create a robust 90/10 split from the structured data
        print("Creating a 90/10 Train/Validation split...")
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size

        train_set, test_set = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader, num_classes


def main(args):
    device = (
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # ------------------- DATASET SETUP
    train_loader, test_loader, num_classes = get_loaders(
        args.dataset, args.data_dir, args.batch_size, args.num_workers
    )
    print(f"Dataset initialized with {num_classes} classes.")

    # ------------------- MODEL SETUP
    model = DinoClassifier(device, num_classes=num_classes).to(device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
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
            args.checkpoint_dir, f"dino_{args.dataset}_latest.pth"
        )
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_acc": v_acc,
                "num_classes": num_classes,
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
    plt.savefig(os.path.join("figure", f"train_val_curve_{args.dataset}.jpg"))
    print(f"Training curve saved to figure/train_val_curve_{args.dataset}.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AttentionSurgeon: DINOv2 Linear Probing for Classification [cite: 31]"
    )

    # Path Arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="cifar10, imagenet, imagenet100, imagenet50, etc.",
    )
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
