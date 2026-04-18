"""
This code defines the RL agent for pruning.
We need to define:
- State Space (????) I thought of giving the task
- Action Space (Mask 12*12 ??)
- Reward (????)
- Define the MLP of the RL agent (2 mlp if PPO actor/critic), ONlY actor if GRPO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt

from classification import DinoClassifier, validate, get_loaders
from rl_utils import get_flops_ratio, get_proxy_loader, FastProxyValidator
from head_activation import HeadCensus

# ----------------- ENVIRONMENT -----------------


class PruningEnv:
    def __init__(self, model, dataloader, device, max_pruning=72):
        self.model = model
        self.device = device
        self.max_pruning = max_pruning  # Up to 72 heads

        self.num_heads = 144
        self.base_flops = self.num_heads  # Simplified proxy for FLOPs

        self.proxy_loader = get_proxy_loader(dataloader, num_samples=500)
        self.validator = FastProxyValidator(
            model, self.proxy_loader, nn.CrossEntropyLoss(), device
        )

        self.micro_census_loader = get_proxy_loader(dataloader, num_samples=16, seed=99)
        self.census = HeadCensus(model, device)

        self.reset()

    def reset(self):
        self.mask = torch.ones(self.num_heads).to(self.device)
        self.steps = 0
        # Get initial accuracy for the state
        _, self.current_acc = self.validator.evaluate(self.mask)
        return self._get_state()

    def _get_valid_action_mask(self):
        """
        PREVENTS LAYER COLLAPSE:
        If a layer has 1 (or 0) heads left, mask it out so the agent cannot prune it.
        """
        valid_mask = self.mask.clone()
        for layer_idx in range(12):
            start_idx = layer_idx * 12
            end_idx = start_idx + 12

            # Count active heads in this specific layer
            active_in_layer = self.mask[start_idx:end_idx].sum().item()

            if active_in_layer <= 1:
                # Forbid pruning any remaining heads in this layer
                valid_mask[start_idx:end_idx] = 0

        return valid_mask

    def _get_state(self):
        # Re-run the census on just 16 images - zero out old metrics and fill in new ones
        self.census.head_metrics.zero_()
        for i in range(12):
            self.census.head_metrics[3, i, :] = (i + 1) / 12.0
        self.census.run_census(self.micro_census_loader, num_batches=1)

        # Ensure pruned heads are explicitly zeroed out in the metrics
        spatial_mask = self.mask.view(12, 12).unsqueeze(0)
        current_metrics = (
            self.census.head_metrics[:7].clone().to(self.device) * spatial_mask
        )

        # 2. Get Scalars
        flops_ratio = get_flops_ratio(self.mask)
        scalars = torch.tensor([self.current_acc / 100.0, flops_ratio]).to(self.device)

        # 3. Get the Safeguard Mask
        safe_action_mask = self._get_valid_action_mask()

        return {
            "metric_grid": current_metrics,
            "scalars": scalars,
            "mask": safe_action_mask,  # Give the policy the restricted mask!
        }

    def step(self, action_idx):
        # Apply pruning (Action: index of the next head to prune)
        self.mask[action_idx] = 0
        self.steps += 1

        # Sync the mask with the model
        self.model.set_mask(self.mask)

        # Evaluate new accuracy after pruning
        _, new_acc = self.validator.evaluate(self.mask)

        # # Calculate Reward: accuracy * (1 - (current FLOPs / base FLOPs))
        flops_ratio = get_flops_ratio(self.mask)
        # reward = (new_acc / 100.0) * (1.0 - flops_ratio)

        # Simple Delta-Accuracy Reward (FLOPS drop is constant per action, don't work if idle actions are possible)
        reward = new_acc - self.current_acc
        # Optional: Add a tiny survival bonus so the agent slightly prefers state stability
        reward += 0.05

        print(
            f"   -> Step {self.steps}: Pruned Head {action_idx} | New Acc: {new_acc:.2f}% | FLOPs Ratio: {flops_ratio:.4f} | Reward: {reward:.4f}"
        )

        self.current_acc = new_acc
        done = self.steps >= self.max_pruning

        return self._get_state(), reward, done


# ----------------- PPO AGENT -----------------


class ActorCritic(nn.Module):
    def __init__(self, state_dim=146, action_dim=144):
        super(ActorCritic, self).__init__()
        # Small MLP policy network
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()
        )
        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)

    def forward(self, state, mask):
        x = self.shared(state)
        logits = self.actor(x)

        # Mask already pruned heads (set logits to very low value)
        logits = logits.masked_fill(mask == 0, -1e9)
        probs = F.softmax(logits, dim=-1)

        value = self.critic(x)
        return Categorical(probs), value


class MiniViT(nn.Module):
    """
    A lightweight Vision Transformer specifically designed for a 12x12 metric grid.
    Uses 1x1 patches so each of the 144 attention heads becomes its own token.
    """

    def __init__(self, n_metrics=7, embed_dim=64, num_layers=3, num_heads=4):
        super(MiniViT, self).__init__()

        self.num_patches = 144  # 12x12 grid
        self.embed_dim = embed_dim

        # 1. Patch Embedding (1x1 patch means we just project the n_metrics to embed_dim)
        # Input shape: (Batch, n_metrics, 12, 12)
        # Flattened shape: (Batch, n_metrics, 144) -> permuted to (Batch, 144, n_metrics)
        self.patch_proj = nn.Linear(n_metrics, embed_dim)

        # 2. CLS Token & Positional Encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # +1 for the CLS token
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )

        # 3. Transformer Encoder Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.shape[0]

        # Flatten spatial dimensions: (B, n_metrics, 12, 12) -> (B, n_metrics, 144)
        x = x.view(B, x.shape[1], -1)
        # Permute to (B, 144, n_metrics) for the linear layer
        x = x.permute(0, 2, 1)

        # Project tokens to embed_dim
        x = self.patch_proj(x)

        # Prepend CLS token: (B, 145, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add Positional Encoding
        x = x + self.pos_embed

        # Pass through Transformer
        x = self.transformer(x)
        x = self.norm(x)

        # Extract the CLS token output (index 0)
        return x[:, 0]


class AdvancedActorCritic(nn.Module):
    """
    The new PPO architecture integrating the MiniViT and scalar metrics.
    """

    def __init__(self, n_metrics=7, vit_embed_dim=64, hidden_dim=256, action_dim=144):
        super(AdvancedActorCritic, self).__init__()

        # 1. Spatial Feature Extractor
        self.vit = MiniViT(n_metrics=n_metrics, embed_dim=vit_embed_dim)

        # 2. Fusion Layer (ViT CLS token + Accuracy + FLOPs)
        # Dimension: vit_embed_dim + 2 scalars
        fused_dim = vit_embed_dim + 2

        # Shared MLP for higher-level reasoning after fusion
        self.shared_mlp = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 3. Output Heads
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, metric_grid, scalars, mask):
        """
        Args:
            metric_grid: Tensor of shape (B, n_metrics, 12, 12)
            scalars: Tensor of shape (B, 2) containing [Accuracy, FLOPs_Ratio]
            mask: Tensor of shape (B, 144) indicating active heads (1) and pruned heads (0)
        """
        # Ensure correct dimensionality if unbatched (e.g., single step in env)
        if metric_grid.dim() == 3:
            metric_grid = metric_grid.unsqueeze(0)
            scalars = scalars.unsqueeze(0)
            mask = mask.unsqueeze(0)

        # 1. Extract global feature vector from the grid
        cls_feature = self.vit(metric_grid)  # Shape: (B, vit_embed_dim)

        # 2. Concatenate with scalars
        fused_state = torch.cat([cls_feature, scalars], dim=-1)  # Shape: (B, fused_dim)

        # 3. Process through shared MLP
        x = self.shared_mlp(fused_state)

        # 4. Actor Head (with masking)
        logits = self.actor(x)
        # Mask out already pruned heads to prevent the agent from selecting them
        logits = logits.masked_fill(mask == 0, -1e9)
        probs = F.softmax(logits, dim=-1)

        # 5. Critic Head
        value = self.critic(x)

        return Categorical(probs), value


def save_training_plots(history, save_path):
    """
    Generates and saves a figure with Return, Entropy, and KL Divergence.
    """
    epochs = range(1, len(history["episodic_return"]) + 1)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Episodic Return
    axs[0].plot(epochs, history["episodic_return"], color="tab:blue", linewidth=1.5)
    axs[0].set_title("Total Episodic Return")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Sum of Rewards")
    axs[0].grid(True, alpha=0.3)

    # 2. Entropy (Exploration)
    axs[1].plot(epochs, history["entropy"], color="tab:green", linewidth=1.5)
    axs[1].set_title("Policy Entropy")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Entropy Score")
    axs[1].grid(True, alpha=0.3)

    # 3. KL Divergence (Stability)
    axs[2].plot(epochs, history["approx_kl"], color="tab:red", linewidth=1.5)
    axs[2].axhline(
        y=0.02, color="gray", linestyle="--", label="Target Max"
    )  # Reference line
    axs[2].set_title("Approx. KL Divergence")
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("KL")
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # Close to free up memory


# ----------------- TRAINING LOGIC -----------------


def train_ppo(args):
    plot_dir = os.path.join(args.save_dir, "figure_metrics")
    os.makedirs(plot_dir, exist_ok=True)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Parse the dataset name from the checkpoint filename (e.g., "dino_imagenet100_latest.pth")
    filename = os.path.basename(args.checkpoint)
    match = re.search(r"dino_(.*?)_(latest|epoch)", filename)

    if match:
        dataset_name = match.group(1)
        print(f"[Auto-Detect] Extracted dataset '{dataset_name}' from checkpoint name.")
    else:
        # Fallback just in case the file was renamed manually
        dataset_name = "imagenet100"
        print(
            f"[Warning] Could not auto-detect dataset from '{filename}'. Defaulting to '{dataset_name}'."
        )

    # Initialize Model and Data Loaders
    _, val_loader, num_classes = get_loaders(
        dataset_name, args.data_dir, args.batch_size, num_workers=2
    )
    model = DinoClassifier(device=device, num_classes=num_classes).to(device)

    if os.path.exists(args.checkpoint):
        model.load_state_dict(
            torch.load(args.checkpoint, map_location=device)["model_state_dict"]
        )

    env = PruningEnv(model, val_loader, device, max_pruning=args.max_pruning)
    policy = AdvancedActorCritic(n_metrics=7).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    eps_clip = 0.2
    gamma = 0.99

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    best_reward = -float("inf")

    # Initialize history for plotting
    history = {"episodic_return": [], "entropy": [], "approx_kl": []}

    print("Starting PPO Training Loop...")
    for episode in range(args.episodes):
        print(f"\n--- Episode {episode+1}/{args.episodes} ---")
        state_dict = env.reset()
        grids, scalars, masks, actions, log_probs, rewards, values = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        # Episode: Sequentially prune heads
        for t in range(args.max_pruning):
            grid = state_dict["metric_grid"]
            scalar = state_dict["scalars"]
            mask = state_dict["mask"]

            dist, value = policy(grid, scalar, mask)
            action = dist.sample()

            next_state_dict, reward, done = env.step(action.item())

            grids.append(grid)
            scalars.append(scalar)
            masks.append(mask)
            actions.append(action)
            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(reward)

            state_dict = next_state_dict
            if done:
                break

        # Metric 1: Total Return (Not discounted)
        total_return = sum(rewards)
        history["episodic_return"].append(total_return)

        # 2. Compute Returns and Advantages
        returns = []
        discounted_reward = 0
        for r in reversed(rewards):
            discounted_reward = r + (gamma * discounted_reward)
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        values = torch.cat(values).squeeze()
        advantages = returns - values.detach()
        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3. PPO Update Step
        old_grids = torch.stack(grids)
        old_scalars = torch.stack(scalars)
        old_masks = torch.stack(masks)
        old_actions = torch.stack(actions)
        old_log_probs = torch.stack(log_probs).detach()

        # METRIC 2-3: KL divergence and Entropy
        epoch_kls = []
        epoch_entropies = []

        for _ in range(5):  # Optimize for K epochs
            dist, val = policy(old_grids, old_scalars, old_masks)
            new_log_probs = dist.log_prob(old_actions)
            entropy = dist.entropy().mean()

            # --- CALCULATE METRICS ---

            # Entropy: Measures how much the agent is exploring
            entropy = dist.entropy().mean()

            # Approximate KL Divergence: log(q/p) where q is new policy and p is old
            # A common robust formula: mean((log_p_old - log_p_new) + (exp(log_p_new - log_p_old) - 1))
            log_ratio = new_log_probs - old_log_probs
            approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()

            epoch_kls.append(approx_kl.item())
            epoch_entropies.append(entropy.item())

            # --- LOSS CALCULATION ---

            # Policy Loss (Clipped)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value Loss (MSE)
            critic_loss = F.mse_loss(val.squeeze(), returns)

            # Total Loss
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Log the average of the K epochs for this episode
        history["entropy"].append(np.mean(epoch_entropies))
        history["approx_kl"].append(np.mean(epoch_kls))

        # --- SAVE LOGIC ---
        avg_reward = sum(rewards) / len(rewards)

        # Save "Latest" model
        torch.save(
            policy.state_dict(),
            os.path.join(
                args.save_dir, f"surgeon_ppo_latest_{args.max_pruning}prune.pth"
            ),
        )

        # Save "Best" model based on total episode reward
        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save(
                policy.state_dict(),
                os.path.join(
                    args.save_dir, f"surgeon_ppo_best_{args.max_pruning}prune.pth"
                ),
            )
            print(f"   *** New Best Reward! ({best_reward:.4f}) ***")

        print(
            f"Final Step Acc: {env.current_acc:.2f}% | Avg Reward: {avg_reward:.4f} | Loss: {loss.item():.4f}"
        )

        if (episode + 1) % 10 == 0:
            plot_path = os.path.join(plot_dir, f"metrics_ep_{episode+1}.png")
            save_training_plots(history, plot_path)
            print(f"   -> Plot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AttentionSurgeon RL Training")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained DINOv2 linear probe",
    )
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints_rl",
        help="Where to save RL agent weights",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_pruning", type=int, default=72)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    train_ppo(args)
