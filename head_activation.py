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
    # skip the cls and register tokens
    attn_map = attn_map[:, :, 5:, 5:]
    # renormalize
    attn_map = attn_map / (attn_map.sum(dim=-1, keepdim=True) + 1e-8)

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

    # renormalization
    attn_patches = attn_patches / (attn_patches.sum(dim=-1, keepdim=True) + 1e-8)

    # Weighted average distance per head
    avg_dist = torch.sum(attn_patches * dists, dim=-1)  # (B, H, 256)
    return avg_dist.mean(dim=(0, 2))  # (H,)

def compute_attention_rollout(all_layer_attn):
    """
    Computes Attention Rollout as per Abnar & Zuidema (2020).
    
    Args:
        all_layer_attn: List or Tensor of shape (L, B, H, N, N) 
                        containing attention maps for each layer.
    Returns:
        rollout_maps: Tensor of shape (L, B, N, N) showing accumulated flow.
    """
    num_layers, B, H, N, _ = all_layer_attn.shape
    device = all_layer_attn.device
    
    # Initialize the rollout with an Identity matrix (the flow at "layer -1")
    curr_rollout = torch.eye(N, device=device).expand(B, N, N)
    rollout_results = []

    for i in range(num_layers):
        # 1. Average across heads (Standard Rollout procedure)
        # Shape: (B, N, N)
        layer_attn = all_layer_attn[i].mean(dim=1)
        
        # 2. Account for Residual Connections
        # A_bar = 0.5 * I + 0.5 * A
        # This models the fact that the token retains its own identity
        eye = torch.eye(N, device=device).expand(B, N, N)
        layer_attn_res = 0.5 * layer_attn + 0.5 * eye
        
        # 3. Re-normalize to ensure rows sum to 1
        layer_attn_res = layer_attn_res / layer_attn_res.sum(dim=-1, keepdim=True)
        
        # 4. Multiply current layer flow by accumulated flow
        # Rollout(i) = A_i * Rollout(i-1)
        curr_rollout = torch.matmul(layer_attn_res, curr_rollout)
        
        rollout_results.append(curr_rollout)

    return torch.stack(rollout_results) # (L, B, N, N)

# ----------------- CENSUS RUNNER -----------------


class HeadCensus:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.num_layers = 12
        self.num_heads = 12
        self.head_dim = 768 // self.num_heads

        # Metric order: 0: Entropy, 1: Distance, 2: Rollout, 3: Depth, 4: Residual Contrib
        self.all_metrics = torch.zeros(5, self.num_layers, self.num_heads).to(device)
        
        # Pre-fill Depth (Metric index 3)
        for i in range(self.num_layers):
            self.all_metrics[3, i, :] = (i + 1) / 12.0

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

    def compute_residual_contribution(self, x, layer_idx):
        """
        Computes the L2 norm contribution of each head to the residual connection.
        x: input tensor to the transformer block (B, N, C)
        """
        block = self.model.transformer.blocks[layer_idx]
        B, N, C = x.shape
        
        # 1. Get QKV and split into heads
        # DINOv2 uses a combined linear layer for QKV
        qkv = block.attn.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2. Standard Attention calculation
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        
        # 3. Context vector (weighted sum of values)
        # Shape: (B, num_heads, N, head_dim)
        head_outputs = attn @ v 
        
        # 4. Project each head back to the embedding dimension (C)
        # The block.attn.proj is the W_O matrix. We must apply the relevant 
        # slice of W_O to each head's output.
        proj_weight = block.attn.proj.weight # Shape: (C, C)
        # Reshape weights to (num_heads, head_dim, C) to process heads individually
        proj_weight_per_head = proj_weight.view(C, self.num_heads, self.head_dim).permute(1, 2, 0)
        
        # head_outputs: (B, H, N, D) @ proj_weight_per_head: (H, D, C)
        # Result: (B, H, N, C)
        out_per_head = torch.einsum('bhnd,hdc->bhnc', head_outputs, proj_weight_per_head)
        
        # 5. Compute Norm (Magnitude of the contribution)
        # We take the mean norm across Batch and Token dimensions
        contribution = torch.norm(out_per_head, p=2, dim=-1) # (B, H, N)
        return contribution.mean(dim=(0, 2)) # (H,)

    
    def run_census(self, dataloader, num_batches=10):
        self.model.eval()
        for param in self.model.transformer.parameters():
            param.requires_grad = True 

        for b_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Census")):
            if b_idx >= num_batches: break
            images, labels = images.to(self.device), labels.to(self.device)
            
            # 1. Forward pass & collect maps for Rollout
            layer_attns = [] # To store (B, H, N, N) per layer
            x = self.model.transformer.prepare_tokens_with_masks(images)
            
            for i in range(self.num_layers):
                attn_map, _ = self.get_attention_map(x, i)
                layer_attns.append(attn_map)
                
                with torch.no_grad():
                    # (0) Entropy
                    self.all_metrics[0, i] += compute_entropy(attn_map)
                    # (1) Distance
                    self.all_metrics[1, i] += compute_distance(attn_map)
                    # (4) Residual Contribution
                    self.all_metrics[4, i] += self.compute_residual_contribution(x, i)

                x = self.model.transformer.blocks[i](x)

            # 2. Compute Rollout (Metric 2)
            # Stack to (L, B, H, N, N)
            stacked_attns = torch.stack(layer_attns)
            rollout_maps = compute_attention_rollout(stacked_attns) # (L, B, N, N)
            
            with torch.no_grad():
                for i in range(self.num_layers):
                    # We average the rollout flow map for that layer 
                    # Note: Rollout is N x N, we take the mean "influence" per layer
                    self.all_metrics[2, i] += rollout_maps[i].mean()

        # Normalize across batches (excluding static Depth at index 3)
        for idx in [0, 1, 2, 4]:
            self.all_metrics[idx] /= num_batches

    def plot_results(self, task_name):
        metric_names = ["Entropy", "Distance", "Rollout", "Depth", "Res_Contrib"]
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        fig.suptitle(f"Head Census - {task_name}", fontsize=16)

        for i in range(5):
            data = self.all_metrics[i].cpu().numpy()
            im = axes[i].imshow(data, cmap="magma")
            axes[i].set_title(metric_names[i])
            axes[i].set_xlabel("Head")
            axes[i].set_ylabel("Layer")
            fig.colorbar(im, ax=axes[i])

        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig(f"results/census_full_{task_name}.png")


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
