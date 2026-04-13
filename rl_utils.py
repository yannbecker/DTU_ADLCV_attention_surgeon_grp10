import torch
from torch.utils.data import Subset, DataLoader
import numpy as np

"""
REPENSER A CA, comment recuperer et pruning 

"""
num_heads = 12
head_dim = 768 // num_heads


def split_qkv_heads(model, i):
    """
    Dinov2 store 12 qkv matrices in one tensor. This function is a helper to go from (num_heads * head_dim * 3_qkv , out_features) -> (num_heads, head_dim, 3_qkv , out_features).
    Operate for the i-th block.
    """
    qkvs = model.blocks[i].attn.qkv.weight.reshape(num_heads, 3, head_dim, 768)
    return qkvs


def get_head_attention(model, x, i):
    """
    Get the attention_map for input x and block i
    """
    B, N, C = x.shape
    qkv = (
        model.blocks[i]
        .attn.qkv(x)
        .reshape(B, N, 3, num_heads, C // num_heads)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv[0], qkv[1], qkv[2]

    # Calcul de la matrice d'attention : (Softmax(QK^T / sqrt(d)))
    attn = (q @ k.transpose(-2, -1)) * (head_dim**-0.5)
    attn = attn.softmax(dim=-1)

    return attn


def calculate_theoretical_flops(mask, img_size=224, patch_size=14, embed_dim=768):
    """
    Calculates the theoretical FLOPs for the Attention layers based on the active heads.
    Standard ViT-B/14 on 224x224:
    - 256 patches + 1 CLS token = 257 tokens (N)
    - 12 layers, 12 heads per layer (144 total)
    """
    # 1. Constants
    N = (img_size // patch_size) ** 2 + 1  # Sequence length (257)
    D = embed_dim  # Embedding dimension (768)
    h_dim = D // 12  # Dimension per head (64)

    # Reshape mask to (Layers, Heads)
    mask = mask.view(12, 12)

    total_attn_flops = 0

    # 2. Attention FLOPs calculation per layer
    # We only count the FLOPs for the heads that are ACTIVE (mask == 1)
    for i in range(12):
        active_heads = torch.sum(mask[i]).item()
        D_prime = active_heads * h_dim  # Reduced effective dimension

        # A. QKV Projections: 3 * (N * D * D_prime)
        qkv_projections = 3 * N * D * D_prime

        # B. Attention Matrix (QK^T): active_heads * (N * N * h_dim)
        attn_scores = active_heads * N * N * h_dim

        # C. Attention Weighted Sum (Softmax(QK^T)V): active_heads * (N * N * h_dim)
        attn_weighted_sum = active_heads * N * N * h_dim

        # D. Output Projection (O): N * D_prime * D
        output_projection = N * D_prime * D

        total_attn_flops += (
            qkv_projections + attn_scores + attn_weighted_sum + output_projection
        )

    # 3. Fixed FLOPs (MLP layers and Patch Embedding don't change in this project)
    # MLP has two linear layers: D -> 4D and 4D -> D
    # FLOPs = 12 layers * (N * D * 4D + N * 4D * D)
    mlp_flops = 12 * (N * D * (4 * D) + N * (4 * D) * D)

    return total_attn_flops + mlp_flops


def get_flops_ratio(mask):
    """
    Returns the ratio of current FLOPs to base FLOPs for the reward function.
    """
    full_mask = torch.ones(144)
    base_flops = calculate_theoretical_flops(full_mask)
    current_flops = calculate_theoretical_flops(mask)
    return current_flops / base_flops

def get_proxy_loader(base_loader, num_samples=500, seed=42):
    """
    Creates a fixed, small proxy DataLoader from the base validation loader.
    This guarantees the RL agent is evaluated on the exact same subset every step.
    """
    dataset = base_loader.dataset
    total_samples = len(dataset)
    
    # Ensure we don't request more samples than the dataset holds
    num_samples = min(num_samples, total_samples)
    
    # Use a fixed seed to ensure the proxy set is consistent across runs/episodes
    np.random.seed(seed)
    indices = np.random.choice(total_samples, num_samples, replace=False)
    
    proxy_dataset = Subset(dataset, indices)
    
    proxy_loader = DataLoader(
        proxy_dataset, 
        batch_size=base_loader.batch_size, 
        shuffle=False, 
        num_workers=base_loader.num_workers,
        pin_memory=True # Speeds up transfer to GPU on HPC
    )
    
    return proxy_loader


class FastProxyValidator:
    """
    Handles rapid evaluation of the model for the RL environment.
    Combines Proxy Data with Baseline Caching.
    """
    def __init__(self, model, proxy_loader, criterion, device):
        self.model = model
        self.proxy_loader = proxy_loader
        self.criterion = criterion
        self.device = device
        
        # Activation Caching
        # We pre-compute and cache the prepared tokens (patch embeddings + CLS + Registers)
        # This allows us to skip the image-to-patch extraction step during every RL step.
        self.cached_inputs = []
        self.cached_labels = []
        self._build_cache()

    def _build_cache(self):
        """
        Runs through the proxy loader once and stores the initial transformer inputs.
        Requires ~400MB of VRAM for 500 images, easily fits on the HPC.
        """
        self.model.eval()
        print(f"Building Proxy Cache for {len(self.proxy_loader.dataset)} samples...")
        
        with torch.no_grad():
            for images, labels in self.proxy_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Extract tokens exactly as DINOv2 does before the transformer blocks
                # This skips the CNN/Patch Embedding overhead in future steps
                tokens = self.model.transformer.prepare_tokens_with_masks(images)
                
                self.cached_inputs.append(tokens)
                self.cached_labels.append(labels)
                
        print("Proxy Cache built successfully.")

    def evaluate(self, mask):
        """
        Executes a rapid forward pass using cached tokens and the current RL mask.
        """
        self.model.eval()
        self.model.set_mask(mask)
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for tokens, labels in zip(self.cached_inputs, self.cached_labels):
                
                # 1. Partial Forward Pass (Transformer blocks only)
                # We bypass the standard model(x) to inject our cached tokens
                x = tokens
                for blk in self.model.transformer.blocks:
                    x = blk(x)
                    
                x = self.model.transformer.norm(x)
                
                # DINOv2 CLS token is at index 0
                cls_token = x[:, 0]
                
                # 2. Classifier Head
                logits = self.model.classifier(cls_token)
                loss = self.criterion(logits, labels)
                
                # 3. Metrics
                running_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100.0 * correct / total
        avg_loss = running_loss / len(self.cached_labels)
        
        return avg_loss, accuracy