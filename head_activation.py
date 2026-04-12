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
    attn_map shape: (B, H, N, N)
    """
    # Sum over the last dimension (key tokens)
    entropy = -torch.sum(
        attn_map * torch.log(attn_map + 1e-8), dim=-1
    )  # Shape: (B, H, N)
    return entropy.mean(dim=(0, 2))  # Mean over Batch and Query tokens -> Shape: (H,)


def compute_distance(attn_map, grid_size=16):
    """
    Computes average attention distance (Task 1b).
    DINOv2 ViT-B/14 on 224x224 images has 16x16 patches.
    """
    device = attn_map.device
    # Create coordinate grid
    x = torch.arange(grid_size, device=device)
    y = torch.arange(grid_size, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    coords = torch.stack(
        [grid_x.flatten(), grid_y.flatten()], dim=1
    ).float()  # (256, 2)

    # Pairwise Euclidean distances
    dists = torch.cdist(coords, coords, p=2)  # (256, 256)

    # Skip the 1 CLS token AND the 4 Register tokens - Slice from index 5 to the end to isolate the 256 patch tokens
    attn_patches = attn_map[:, :, 5:, 5:]  # Now (B, H, 256, 256)

    # Weighted average distance per head
    avg_dist = torch.sum(attn_patches * dists, dim=-1)  # (B, H, 256)
    return avg_dist.mean(dim=(0, 2))  # (H,)


# ----------------- CENSUS RUNNER -----------------


class HeadCensus:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.num_layers = 12
        self.num_heads = 12
        self.head_dim = 768 // self.num_heads

        self.metrics = {
            "entropy": torch.zeros(12, 12).to(device),
            "distance": torch.zeros(12, 12).to(device),
            "importance": torch.zeros(12, 12).to(device),
            "magnitude": torch.zeros(12, 12).to(device),
        }

    def get_attention_map(self, x, layer_idx):
        """
        Manually computes attention map for DINOv2 blocks (from your rl_utils.py logic).
        """
        block = self.model.transformer.blocks[layer_idx]
        B, N, C = x.shape

        # 1. Get QKV
        qkv = (
            block.attn.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2. Compute Attention Matrix
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        return attn, v  # Return V for magnitude calculation

    def run_census(self, dataloader, num_batches=10):
        self.model.eval()

        for param in self.model.transformer.parameters():
            param.requires_grad = True  # Enable gradients for importance metric

        criterion = nn.CrossEntropyLoss()

        for b_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Census")):
            if b_idx >= num_batches:
                break
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass through the backbone (manually to capture layers)
            # We track gradients for 'importance' (Task 1c)
            images.requires_grad = True

            # Step-by-step through transformer blocks
            x = self.model.transformer.prepare_tokens_with_masks(images)

            for i in range(self.num_layers):
                attn_map, v_tokens = self.get_attention_map(x, i)

                with torch.no_grad():
                    # (a) Entropy
                    self.metrics["entropy"][i] += compute_entropy(attn_map)
                    # (b) Distance
                    self.metrics["distance"][i] += compute_distance(attn_map)
                    # (d) Magnitude (L2 norm of value tokens)
                    self.metrics["magnitude"][i] += v_tokens.norm(p=2, dim=(0, 2, 3))

                # Continue full forward pass for gradients
                x = self.model.transformer.blocks[i](x)

            # (c) Importance via Gradient Attribution
            # We backward from the final classification loss
            logits = self.model.classifier(self.model.transformer.norm(x[:, 0]))
            loss = criterion(logits, labels)
            self.model.zero_grad()
            loss.backward()

            with torch.no_grad():
                for i in range(self.num_layers):
                    # Importance based on gradient of QKV weights
                    grad = self.model.transformer.blocks[i].attn.qkv.weight.grad
                    if grad is not None:
                        self.metrics["importance"][i] += grad.norm(p=2)

        # Normalize
        for key in self.metrics:
            self.metrics[key] /= num_batches

        for param in self.model.transformer.parameters():
            param.requires_grad = False  # Disable gradients after census

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
