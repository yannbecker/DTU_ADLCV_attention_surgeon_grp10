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
import numpy as np

from classification import DinoClassifier, validate, get_loaders
from rl_utils import get_flops_ratio

# ----------------- ENVIRONMENT -----------------


class PruningEnv:
    def __init__(self, model, dataloader, device, max_pruning=72):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.max_pruning = max_pruning  # Up to 72 heads

        self.num_heads = 144
        self.base_flops = self.num_heads  # Simplified proxy for FLOPs
        self.reset()

    def reset(self):
        self.mask = torch.ones(self.num_heads).to(self.device)
        self.steps = 0
        # Get initial accuracy for the state
        _, self.current_acc = validate(
            self.model, self.dataloader, nn.CrossEntropyLoss(), self.device
        )
        return self._get_state()

    def _get_state(self):
        # State: 144-dim binary mask + task accuracy + FLOPs count
        flops_ratio = get_flops_ratio(self.mask)
        state = torch.cat(
            [
                self.mask,
                torch.tensor([self.current_acc / 100.0, flops_ratio]).to(self.device),
            ]
        )
        return state

    def step(self, action_idx):
        # Apply pruning (Action: index of the next head to prune)
        self.mask[action_idx] = 0
        self.steps += 1

        # Sync the mask with the model
        self.model.set_mask(self.mask)

        # Evaluate new accuracy after pruning
        _, new_acc = validate(
            self.model, self.dataloader, nn.CrossEntropyLoss(), self.device
        )

        # Calculate Reward: accuracy * (1 - (current FLOPs / base FLOPs))
        flops_ratio = get_flops_ratio(self.mask)
        reward = (new_acc / 100.0) * (1.0 - flops_ratio)

        print(
            f"   -> Step {self.steps}: Pruned Head {action_idx} | New Acc: {new_acc:.2f}% | FLOPs Ratio: {flops_ratio:.4f} | Active Heads: {int(self.mask.sum().item())}"
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


# ----------------- TRAINING LOGIC -----------------


def train_ppo(args):
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Initialize Model and Data
    model = DinoClassifier(device=device, num_classes=10).to(device)
    if os.path.exists(args.checkpoint):
        model.load_state_dict(
            torch.load(args.checkpoint, map_location=device)["model_state_dict"]
        )

    train_loader, val_loader = get_loaders(
        args.data_dir, args.batch_size, num_workers=2
    )

    env = PruningEnv(model, val_loader, device, max_pruning=args.max_pruning)
    policy = ActorCritic().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    eps_clip = 0.2
    gamma = 0.99

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    best_reward = -float("inf")

    for episode in range(args.episodes):
        state = env.reset()
        states, actions, log_probs, rewards, values, masks = [], [], [], [], [], []

        # Episode: Sequentially prune heads
        for t in range(args.max_pruning):
            mask = state[:144]
            dist, value = policy(state, mask)
            action = dist.sample()

            next_state, reward, done = env.step(action.item())

            states.append(state)
            actions.append(action)
            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(reward)
            masks.append(mask)

            state = next_state
            if done:
                break

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
        old_states = torch.stack(states)
        old_actions = torch.stack(actions)
        old_log_probs = torch.stack(log_probs).detach()
        old_masks = torch.stack(masks)

        for _ in range(5):  # Optimize for K epochs
            dist, val = policy(old_states, old_masks)
            new_log_probs = dist.log_prob(old_actions)
            entropy = dist.entropy().mean()

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

        # --- SAVE LOGIC ---
        avg_reward = sum(rewards) / len(rewards)

        # Save "Latest" model
        checkpoint_path = os.path.join(args.save_dir, "surgeon_ppo_latest.pth")
        torch.save(
            {
                "episode": episode,
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": args,
            },
            checkpoint_path,
        )

        # Save "Best" model based on total episode reward
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_path = os.path.join(args.save_dir, "surgeon_ppo_best.pth")
            torch.save(policy.state_dict(), best_path)
            print(f"   *** New Best Reward! Saved to {best_path} ***")

        print(
            f"Episode {episode} | Final Reward: {rewards[-1]:.4f} | Final Acc: {env.current_acc:.2f}| Loss: {loss.item():.4f}%"
        )


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
