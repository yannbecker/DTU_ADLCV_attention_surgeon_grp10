"""
This code plot the activation of each head for a specific task/dataset. This takes place before the RL pruning.
Pipeline ideation:
Select task/dataset -> for each image/batch: -> Forward the input and compute the accuracy
                                             -> Get the Attention map of this input for each of the 144 heads (in a tensor attn_heads of shape (12*12))
                    -> Average and compute the 4 metrics
                    -> plot the heatmap
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
from classification import DinoClassifier, get_loaders

# ----------------- METRIC FUNCTIONS -----------------


def compute_entropy(attn_map):
    """
    Computes average attention entropy: -sum(p * log(p)).
    High entropy = diffuse attention; Low entropy = focused attention.
    attn_map shape: (batch, heads, query_tokens, key_tokens)
    """
    # Add epsilon to avoid log(0)
    entropy = -torch.sum(attn_map * torch.log(attn_map + 1e-8), dim=-1)
    return entropy.mean(dim=(0, 2))  # Mean over batch and query tokens


def compute_distance(attn_map, grid_size=16):
    """
    Computes average attention distance (local vs. global).
    grid_size: 16 for DINOv2 ViT-B/14 (approx 16x16 patches for 224x224)
    """
    # Create coordinate grid for patches
    x = torch.arange(grid_size)
    y = torch.arange(grid_size)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    coords = (
        torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        .float()
        .to(attn_map.device)
    )

    # Calculate pairwise Euclidean distances between all patches
    # dists shape: (num_patches, num_patches)
    dists = torch.cdist(coords, coords, p=2)

    # Exclude CLS token if present (DINOv2 has 1 CLS token + patches)
    # attn_map shape usually: (B, H, N, N) where N = patches + 1
    attn_patches = attn_map[:, :, 1:, 1:]

    avg_dist = torch.sum(attn_patches * dists, dim=-1)
    return avg_dist.mean(dim=(0, 2))


# ----------------- CENSUS RUNNER -----------------


class HeadCensus:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.metrics = {
            "entropy": torch.zeros(12, 12).to(device),
            "distance": torch.zeros(12, 12).to(device),
            "importance": torch.zeros(12, 12).to(device),
            "magnitude": torch.zeros(12, 12).to(device),
        }
        self.attn_weights = {}
        self._register_hooks()

    def _register_hooks(self):
        def get_attn_hook(layer_idx):
            def hook(module, input, output):
                # DINOv2 returns a tuple or tensor depending on version
                # Usually: (batch, heads, seq_len, seq_len)
                self.attn_weights[layer_idx] = output

            return hook

        # DINOv2 ViT-B/14 architecture: model.transformer.blocks -> model.blocks
        for i in range(12):
            # Accessing the attention layer specifically
            layer = self.model.transformer.blocks[i].attn
            layer.register_forward_hook(get_attn_hook(i))

    def run_census(self, dataloader, num_batches=10):
        self.model.eval()
        # We need gradients for "importance via gradient-based attribution"
        criterion = nn.CrossEntropyLoss()

        for i, (images, labels) in enumerate(tqdm(dataloader, desc="Computing Census")):
            if i >= num_batches:
                break
            images, labels = images.to(self.device), labels.to(self.device)

            # Enable gradient for importance metric
            images.requires_grad = True
            outputs = self.model(images)
            loss = criterion(outputs, labels)

            self.model.zero_grad()
            loss.backward()

            with torch.no_grad():
                for layer_idx in range(12):
                    attn = self.attn_weights[layer_idx]
                    # Compute the 4 project metrics
                    self.metrics["entropy"][layer_idx] += compute_entropy(attn)
                    self.metrics["distance"][layer_idx] += compute_distance(attn)
                    # Importance: simplified as gradient magnitude
                    if hasattr(self.model.transformer.blocks[layer_idx].attn, "qkv"):
                        grad = self.model.transformer.blocks[
                            layer_idx
                        ].attn.qkv.weight.grad
                        self.metrics["importance"][layer_idx] += grad.norm(p=2)
                    self.metrics["magnitude"][layer_idx] += attn.norm(
                        p=2, dim=(0, 2, 3)
                    )

        # Normalize across batches
        for key in self.metrics:
            self.metrics[key] /= num_batches

    def plot_results(self, task_name):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Attention Head Census - Task: {task_name}", fontsize=16)

        metric_names = list(self.metrics.keys())
        for i, ax in enumerate(axes.flat):
            data = self.metrics[metric_names[i]].cpu().numpy()
            im = ax.imshow(data, cmap="viridis")
            ax.set_title(f"Average {metric_names[i].capitalize()}")
            ax.set_xlabel("Head Index")
            ax.set_ylabel("Layer Index")
            fig.colorbar(im, ax=ax)

        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig(f"results/census_{task_name}.png")
        print(f"Heatmaps saved to results/census_{task_name}.png")


# ----------------- CLI SETUP -----------------


def main():
    parser = argparse.ArgumentParser(description="DINOv2 Attention Head Census")
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=["classification", "detection", "segmentation"],
        help="Downstream task to evaluate [cite: 40]",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of batches to process for the census",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing the datasets",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/dino_classifier_epoch_30.pth",
        help="Path to the trained linear probe weights",
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Task-Specific Model and Dataloader
    if args.task == "classification":
        # Initialize the DINOv2-based classifier
        model = DinoClassifier(device=device, num_classes=10).to(device)

        # Load the trained linear probe weights
        if os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded weights from {args.checkpoint}")
        else:
            print(
                f"Warning: Checkpoint {args.checkpoint} not found. Running with random head."
            )

        # Retrieve loaders using the helper from classification.py
        dataloader, _ = get_loaders(args.data_dir, args.batch_size, num_workers=2)

    else:
        # Placeholder for Task 2 and 3 implementation
        print(
            f"Task {args.task} not yet implemented. Please create detection.py/segmentation.py"
        )
        return

    # Execute Census and save visualizations
    census = HeadCensus(model, device)
    census.run_census(dataloader, num_batches=args.samples)
    census.plot_results(args.task)


if __name__ == "__main__":
    main()
