import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os

from classification import DinoClassifier, validate, get_loaders
from agent import AdvancedActorCritic, PruningEnv


class PhysicalPruner:
    @staticmethod
    def prune_model(model, mask_12x12):
        """
        Physically resizes the QKV and Proj Linear layers of a DINOv2 model.
        mask_12x12: torch.Tensor of shape (12, 12) with 1s (keep) and 0s (prune).
        """
        head_dim = 64
        total_pruned = 0

        # 1. Iterate through each of the 12 transformer blocks
        for layer_idx in range(12):
            attn = model.transformer.blocks[layer_idx].attn
            layer_mask = mask_12x12[layer_idx]

            # Find the indices of the heads we are keeping
            keep_indices = layer_mask.nonzero().squeeze(-1).tolist()
            num_kept = len(keep_indices)

            if num_kept == 12:
                continue  # No pruning needed for this layer

            if num_kept == 0:
                raise ValueError(
                    f"Cannot completely prune layer {layer_idx}. Layer collapse detected."
                )

            total_pruned += 12 - num_kept

            # ---------------------------------------------------------
            # 2. Slice the QKV Matrix
            # QKV is packed as [Q_all, K_all, V_all]
            # ---------------------------------------------------------
            qkv_mask = torch.zeros(3 * 12 * head_dim, dtype=torch.bool)
            for h in keep_indices:
                qkv_mask[0 * 768 + h * head_dim : 0 * 768 + (h + 1) * head_dim] = (
                    True  # Q
                )
                qkv_mask[1 * 768 + h * head_dim : 1 * 768 + (h + 1) * head_dim] = (
                    True  # K
                )
                qkv_mask[2 * 768 + h * head_dim : 2 * 768 + (h + 1) * head_dim] = (
                    True  # V
                )

            old_qkv = attn.qkv
            new_qkv = nn.Linear(
                768, 3 * num_kept * head_dim, bias=(old_qkv.bias is not None)
            )
            new_qkv.weight.data = old_qkv.weight.data[qkv_mask, :]
            if old_qkv.bias is not None:
                new_qkv.bias.data = old_qkv.bias.data[qkv_mask]

            attn.qkv = new_qkv

            # ---------------------------------------------------------
            # 3. Slice the Output Projection (proj) Matrix
            # ---------------------------------------------------------
            proj_mask = torch.zeros(12 * head_dim, dtype=torch.bool)
            for h in keep_indices:
                proj_mask[h * head_dim : (h + 1) * head_dim] = True

            old_proj = attn.proj
            new_proj = nn.Linear(
                num_kept * head_dim, 768, bias=(old_proj.bias is not None)
            )
            new_proj.weight.data = old_proj.weight.data[:, proj_mask]
            if old_proj.bias is not None:
                new_proj.bias.data = old_proj.bias.data

            attn.proj = new_proj

            # ---------------------------------------------------------
            # 4. Update Params & Monkey-Patch Forward Pass
            # ---------------------------------------------------------
            attn.num_heads = num_kept
            PhysicalPruner._patch_attention_forward(attn, head_dim)

        # 5. Remove the simulated forward hooks to prevent conflicts
        if hasattr(model, "hooks"):
            for h in model.hooks:
                h.remove()
            model.hooks = []

        print(
            f"  [+] Physical surgery complete. Permanently removed {total_pruned} heads."
        )

    @staticmethod
    def _patch_attention_forward(attn_module, head_dim=64):
        """
        Replaces the native DINOv2 forward pass with a robust version that
        dynamically adapts to the new irregular number of heads per layer using
        PyTorch 2.0 Scaled Dot-Product Attention for max speed.
        """
        import types

        def new_forward(self, x):
            B, N, C = x.shape

            # qkv output: (B, N, 3 * num_kept_heads * 64)
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv.unbind(0)

            # Efficient attention computation
            x = F.scaled_dot_product_attention(q, k, v)

            # Flatten back to (B, N, num_kept_heads * 64)
            x = x.transpose(1, 2).reshape(B, N, self.num_heads * head_dim)

            # Project back to 768
            x = self.proj(x)
            return x

        # Bind the new method directly to this specific attention instance
        attn_module.forward = types.MethodType(new_forward, attn_module)

    @staticmethod
    def generate_sequence_tensor(agent, env, max_prune):
        """
        Runs the agent to generate a 1D tensor embedding the pruning order.
        Value 0 = Kept.
        Value K = Pruned at step K.
        """
        sequence_tensor = torch.zeros(144, dtype=torch.int32)
        state = env.reset()

        with torch.no_grad():
            for step in range(1, max_prune + 1):
                dist, _ = agent(state["metric_grid"], state["scalars"], state["mask"])
                action = dist.probs.argmax().item()

                sequence_tensor[action] = step
                state, _, done = env.step(action)
                if done:
                    break

        return sequence_tensor

    @staticmethod
    def get_mask_for_step(sequence_tensor, step):
        """
        Decodes the sequence tensor into a 12x12 binary mask for a specific step.
        Heads pruned at <= 'step' are 0 (pruned).
        All other heads are 1 (kept).
        """
        mask_1d = (sequence_tensor == 0) | (sequence_tensor > step)
        return mask_1d.float().view(12, 12)

    @staticmethod
    def yield_sequential_models(base_model, sequence_tensor):
        """
        Generator designed for the environmental study script.
        Yields: (step_number, physically_pruned_model)
        """
        import copy

        max_step = sequence_tensor.max().item()

        for step in range(1, max_step + 1):
            # Create a fresh copy of the unpruned base model to avoid index shifting!
            model_copy = copy.deepcopy(base_model)

            # Decode the binary mask for this exact step
            mask_12x12 = PhysicalPruner.get_mask_for_step(sequence_tensor, step)

            # Physically prune the fresh copy
            PhysicalPruner.prune_model(model_copy, mask_12x12)

            yield step, model_copy


def count_parameters(model):
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad or not p.requires_grad
    )


def main(args):
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"\n--- Starting Physical Pruning Validation Pipeline on {device} ---")

    # 1. Load Data
    _, val_loader, num_classes = get_loaders(
        args.dataset, args.data_dir, args.batch_size, num_workers=2
    )

    # 2. Load the Task DINO Backbone
    model = DinoClassifier(device=device, num_classes=num_classes).to(device)
    if os.path.exists(args.dino_checkpoint):
        model.load_state_dict(
            torch.load(args.dino_checkpoint, map_location=device)["model_state_dict"]
        )
        print(f"[✓] Loaded DINO Backbone: {args.dino_checkpoint}")
    else:
        raise FileNotFoundError(f"DINO Checkpoint {args.dino_checkpoint} not found.")

    params_before = count_parameters(model.transformer)
    print(f"    Base ViT Parameters: {params_before:,}")

    # 3. Load the RL Agent and Extract Mask
    agent = AdvancedActorCritic(n_metrics=7).to(device)
    if os.path.exists(args.agent_checkpoint):
        agent.load_state_dict(torch.load(args.agent_checkpoint, map_location=device))
        agent.eval()
        print(f"[✓] Loaded RL Agent: {args.agent_checkpoint}")
    else:
        raise FileNotFoundError(f"Agent Checkpoint {args.agent_checkpoint} not found.")

    # Run the environment greedily to get the final mask
    env = PruningEnv(model, val_loader, device, max_pruning=args.max_prune)
    state = env.reset()

    print(f"\nRunning RL Agent to prune {args.max_prune} heads...")
    with torch.no_grad():
        for _ in range(args.max_prune):
            # Deterministic selection (argmax) instead of sampling for the final evaluation
            dist, _ = agent(state["metric_grid"], state["scalars"], state["mask"])
            action = dist.probs.argmax()
            state, _, done = env.step(action.item())
            if done:
                break

    final_mask_1d = env.mask.clone()
    final_mask_12x12 = final_mask_1d.view(12, 12)

    # 4. Baseline Evaluation (Simulated Hooks)
    print("\nEvaluating DINO with SIMULATED Pruning (Hooks)...")
    model.set_mask(final_mask_1d)
    _, sim_acc = validate(model, val_loader, nn.CrossEntropyLoss(), device)
    print(f"    -> Simulated Accuracy: {sim_acc:.2f}%")

    # 5. Apply Physical Surgery
    print("\nPerforming Physical Surgery on DINO Matrices...")
    PhysicalPruner.prune_model(model, final_mask_12x12)

    params_after = count_parameters(model.transformer)
    reduction = 100.0 * (params_before - params_after) / params_before
    print(f"    Pruned ViT Parameters: {params_after:,} (Reduced by {reduction:.1f}%)")

    # 6. Final Evaluation (Physically Pruned)
    print("\nEvaluating DINO with PHYSICAL Pruning...")
    _, phys_acc = validate(model, val_loader, nn.CrossEntropyLoss(), device)
    print(f"    -> Physical Accuracy:  {phys_acc:.2f}%")

    # 7. Verification
    print("\n--- Summary ---")
    if abs(sim_acc - phys_acc) < 0.05:
        print("[SUCCESS] Physical accuracy matches Simulated accuracy perfectly.")
    else:
        print("[WARNING] Accuracy mismatch detected. Check tensor slicing logic.")

    # Save the physically pruned model for the CO2/FLOP analysis
    save_path = os.path.join(
        args.save_dir, f"physically_pruned_dino_{args.dataset}.pth"
    )
    torch.save(model.state_dict(), save_path)
    print(f"\nPhysically pruned model saved to {save_path} for CO2/FLOP analysis.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Physical Pruning to DINOv2")
    parser.add_argument(
        "--dino_checkpoint",
        type=str,
        required=True,
        help="Path to trained task DINO weights",
    )
    parser.add_argument(
        "--agent_checkpoint",
        type=str,
        required=True,
        help="Path to trained RL Agent weights",
    )
    parser.add_argument(
        "--dataset", type=str, default="imagenet100", help="Dataset name"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Path to dataset"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="Folder to save the physically pruned model",
    )
    parser.add_argument(
        "--max_prune", type=int, default=72, help="Number of heads to prune"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    main(args)
