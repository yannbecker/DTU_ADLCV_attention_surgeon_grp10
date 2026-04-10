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
from utils import load_model
from classification import DinoClassifier  # Assuming this structure

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
        self.hooks = []
        self.attn_weights = {}
        self.attn_grads = {}
        self._register_hooks()

    def _register_hooks(self):
        def get_attn_hook(layer_idx):
            def hook(module, input, output):
                # DINOv2 attention returns (attn_map) if select_token is used
                # This depends on your specific forward implementation
                self.attn_weights[layer_idx] = output

            return hook

        def get_grad_hook(layer_idx):
            def hook(grad):
                self.attn_grads[layer_idx] = grad

            return hook

        # Accessing DINOv2 blocks
        for i in range(12):
            # This hook path depends on the specific Timm/Facebook implementation of DINOv2
            # For standard ViT: model.transformer.blocks[i].attn
            layer = self.model.transformer.blocks[i].attn
            self.hooks.append(layer.register_forward_hook(get_attn_hook(i)))

    def run_census(self, dataloader, num_batches=10):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        for i, (images, labels) in enumerate(
            tqdm(dataloader, desc="Computing Metrics")
        ):
            if i >= num_batches:
                break
            images, labels = images.to(self.device), labels.to(self.device)

            # For gradient-based importance, we need gradients
            images.requires_grad = True
            outputs = self.model(images)
            loss = criterion(outputs, labels)

            # Backward pass for importance (Task 1c)
            self.model.zero_grad()
            loss.backward()

            with torch.no_grad():
                for layer_idx in range(12):
                    attn = self.attn_weights[layer_idx]

                    # 1. Entropy
                    self.metrics["entropy"][layer_idx] += compute_entropy(attn)

                    # 2. Distance
                    self.metrics["distance"][layer_idx] += compute_distance(attn)

                    # 3. Magnitude (L2 norm of head activations)
                    # Simplified: using mean of attention weights as proxy or hook output features
                    self.metrics["magnitude"][layer_idx] += attn.norm(
                        p=2, dim=(0, 2, 3)
                    )

        # Normalize
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
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--samples", type=int, default=10, help="Number of batches to process"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Task-Specific Model
    if args.task == "classification":
        from classification import train_loader  # Assuming data is exported

        model = DinoClassifier(num_classes=10).to(device)
        # Load weights if trained
        checkpoint = torch.load(
            "checkpoints/dino_classifier_epoch_30.pth", map_location=device
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        dataloader = train_loader
    else:
        print(
            f"Task {args.task} not yet implemented. Please create detection.py/segmentation.py"
        )
        return

    census = HeadCensus(model, device)
    census.run_census(dataloader, num_batches=args.samples)
    census.plot_results(args.task)


if __name__ == "__main__":
    main()
