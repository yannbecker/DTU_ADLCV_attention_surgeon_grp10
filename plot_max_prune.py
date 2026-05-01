import os
import glob
import argparse
import re
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from classification import DinoClassifier, get_loaders
from agent import AdvancedActorCritic, PruningEnv


def evaluate_agent_trajectory(
    agent_path, x_vals, model, test_loader, criterion, device
):
    """
    Loads ONE master agent and evaluates its performance progressively
    at specific step intervals (x_vals) during a single surgical rollout.
    """
    agent = AdvancedActorCritic(n_metrics=7).to(device)
    agent.load_state_dict(torch.load(agent_path, map_location=device))
    agent.eval()

    max_steps = max(x_vals)
    env = PruningEnv(model, test_loader, device, max_pruning=max_steps)
    state = env.reset()

    trajectory_results = []

    print(f"Unrolling single agent trajectory up to {max_steps} steps...")
    with torch.no_grad():
        for step in range(1, max_steps + 1):
            dist, _ = agent(state["metric_grid"], state["scalars"], state["mask"])
            action = dist.probs.argmax()
            state, _, done = env.step(action.item())

            if step in x_vals:
                print(
                    f"  [Pause Surgery] Evaluating full test set at {step} pruned heads..."
                )
                model.set_mask(env.mask)
                correct, total = 0, 0
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    correct += model(images).argmax(1).eq(labels).sum().item()
                    total += labels.size(0)
                full_acc = 100.0 * correct / total

                trajectory_results.append((step, full_acc))
                print(f"  -> Full Test Acc: {full_acc:.2f}%")

            if done:
                break

    # Add the 0-prune baseline (starting accuracy)
    model.set_mask(torch.ones(12, 12, device=device))
    correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        correct += model(images).argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    trajectory_results.insert(0, (0, 100.0 * correct / total))

    return trajectory_results


def load_baselines_from_cache(cache_path, max_x=50):
    """Extracts all calculated baselines from the baselines.py cache."""
    baseline_results = {}
    if not os.path.exists(cache_path):
        print(f"[!] Baseline cache not found at {cache_path}. Run baselines.py first!")
        return baseline_results

    cache = torch.load(cache_path, map_location="cpu").get("results", {})

    for metric, data_dict in cache.items():
        if not isinstance(data_dict, dict):
            continue

        if metric == "random":
            pts = []
            for step, seed_list in data_dict.items():
                if int(step) <= max_x:
                    valid_seeds = [v for v in seed_list if not math.isnan(v)]
                    if valid_seeds:
                        pts.append((int(step), sum(valid_seeds) / len(valid_seeds)))
            baseline_results[f"Baseline ({metric})"] = sorted(pts)
        else:
            pts = [
                (int(step), acc)
                for step, acc in data_dict.items()
                if int(step) <= max_x
            ]
            baseline_results[f"Baseline ({metric})"] = sorted(pts)

    print(f"[✓] Loaded {len(baseline_results)} baselines from cache.")
    return baseline_results


def main(args):
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    os.makedirs(args.save_dir, exist_ok=True)

    results = {}
    target_x_vals = [i for i in range(2, 51, 2)]

    # 1. Load the Baselines
    if args.baseline_cache:
        baselines = load_baselines_from_cache(args.baseline_cache, max_x=50)
        results.update(baselines)

    # 2. Evaluate the Master Agent
    print(f"\n--- Loading Full Model for Checkpoint Trajectory Evaluation ---")
    train_loader, test_loader, num_classes = get_loaders(
        args.dataset, args.data_dir, args.batch_size, num_workers=2
    )
    model = DinoClassifier(device=device, num_classes=num_classes).to(device)

    if os.path.exists(args.checkpoint):
        model.load_state_dict(
            torch.load(args.checkpoint, map_location=device)["model_state_dict"]
        )
    else:
        raise FileNotFoundError(f"Base weights not found at {args.checkpoint}")

    criterion = nn.CrossEntropyLoss()

    master_agent_path = os.path.join(
        args.agent_dir,
        (
            f"surgeon_ppo_best_segmentation.pth"
            if "seg" in args.dataset
            else f"surgeon_ppo_best_50prune.pth"
        ),
    )

    if os.path.exists(master_agent_path):
        print(f"\nEvaluating Master Agent...")
        agent_data = evaluate_agent_trajectory(
            master_agent_path, target_x_vals, model, test_loader, criterion, device
        )
        results["AttentionSurgeon Agent"] = sorted(agent_data)
    else:
        print(f"[!] Master agent {master_agent_path} not found.")

    # 3. Plotting
    if not results:
        return

    plt.figure(figsize=(12, 7))

    # Make the Agent stand out visually against the baselines
    for label, data in results.items():
        x_vals = [dp[0] for dp in data]
        y_vals = [dp[1] for dp in data]

        if "Agent" in label:
            plt.plot(
                x_vals,
                y_vals,
                label=label,
                color="tab:red",
                marker="o",
                linestyle="-",
                linewidth=3,
                markersize=8,
                zorder=10,
            )
        else:
            plt.plot(
                x_vals,
                y_vals,
                label=label,
                marker="x",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
            )

    plt.title(
        f"AttentionSurgeon vs Baselines ({args.dataset})",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Number of Pruned Heads", fontsize=12)
    plt.ylabel("Validation Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower left", fontsize=10)

    save_path = os.path.join(args.save_dir, f"ultimate_comparison_{args.dataset}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nSuccess! Ultimate comparison graph saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_dir", type=str, default="./checkpoints_rl")
    parser.add_argument("--dataset", type=str, default="imagenet100")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--baseline_cache",
        type=str,
        default="results/baselines/pruning_results.pt",
        help="Path to the baselines.py cache",
    )
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    main(args)
