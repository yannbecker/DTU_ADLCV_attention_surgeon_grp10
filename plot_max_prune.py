import os
import sys
import argparse
import re
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassJaccardIndex

# --- Task 1: Classification ---
from classification import DinoClassifier
from classification import get_loaders as get_loaders_cls

# --- Task 2: Segmentation ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "DPT_segmentation"))
from segmentation import DinoSegmenter
from segmentation import get_loaders as get_loaders_seg

from agent import AdvancedActorCritic, PruningEnv


def parse_agent_log(log_path):
    """Parses the text log file to extract the full test accuracies/mIoUs."""
    parsed_data = {}
    current_agent = "AttentionSurgeon Agent"

    if not os.path.exists(log_path):
        print(f"[!] Log file not found at {log_path}")
        return parsed_data

    with open(log_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # Identify which agent is currently being evaluated
        if "Evaluating Master Agent" in line:
            match = re.search(r"Evaluating Master Agent \((.*?)\)", line)
            if match:
                current_agent = (
                    f"AttentionSurgeon Agent ({match.group(1).capitalize()})"
                )
            if current_agent not in parsed_data:
                parsed_data[current_agent] = []

        # Find the pause step
        if "Evaluating full test set at" in line:
            match_step = re.search(r"at (\d+) pruned heads", line)
            if match_step and i + 1 < len(lines):
                step = int(match_step.group(1))
                next_line = lines[i + 1]
                # Regex handles both "Full Test Acc: XX.XX%" and "Full Test mIoU: XX.XX%"
                match_acc = re.search(r"Full Test (?:Acc|mIoU): ([\d.]+)%", next_line)
                if match_acc:
                    acc = float(match_acc.group(1))
                    parsed_data[current_agent].append((step, acc))

    return parsed_data


def apply_mask(model, mask, task):
    """Safely applies the mask depending on the backbone architecture."""
    if task == "segmentation":
        model.pretrained.model.mask = mask.float()
    else:
        model.mask = mask.float()


def evaluate_agent_trajectory(agent_path, x_vals, model, test_loader, device, task):
    """Loads ONE master agent and evaluates its performance progressively."""
    agent = AdvancedActorCritic(n_metrics=7).to(device)
    agent.load_state_dict(torch.load(agent_path, map_location=device))
    agent.eval()

    max_steps = max(x_vals)
    env = PruningEnv(model, test_loader, device, task=task, max_pruning=max_steps)
    state = env.reset()

    trajectory_results = []
    print(f"Unrolling single agent trajectory up to {max_steps} steps...")

    def eval_model():
        model.eval()
        if task == "segmentation":
            metric = MulticlassJaccardIndex(num_classes=150, ignore_index=-100).to(
                device
            )
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    labels[(labels < 0) | (labels >= 150)] = -100
                    _, outputs = model(images, features=False)
                    metric.update(outputs, labels)
            res = metric.compute().item() * 100.0
            metric.reset()
            return res
        else:
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    correct += model(images).argmax(1).eq(labels).sum().item()
                    total += labels.size(0)
            return 100.0 * correct / total

    with torch.no_grad():
        for step in range(1, max_steps + 1):
            dist, _ = agent(state["metric_grid"], state["scalars"], state["mask"])
            action = dist.probs.argmax()
            state, _, done = env.step(action.item())

            if step in x_vals:
                print(
                    f"  [Pause Surgery] Evaluating full test set at {step} pruned heads..."
                )
                apply_mask(model, env.mask, task)
                full_acc = eval_model()
                trajectory_results.append((step, full_acc))
                metric_name = "mIoU" if task == "segmentation" else "Acc"
                print(f"  -> Full Test {metric_name}: {full_acc:.2f}%")

            if done:
                break

    # Add the 0-prune baseline
    apply_mask(model, torch.ones(12, 12, device=device), task)
    full_acc = eval_model()
    trajectory_results.insert(0, (0, full_acc))

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
    baselines = {}
    if args.baseline_cache:
        baselines = load_baselines_from_cache(args.baseline_cache, max_x=50)
        results.update(baselines)

    # 2. Evaluate the Master Agent (FAST LOG PARSE vs HEAVY MODEL LOAD)
    if args.agent_log:
        print(f"\n--- FAST MODE: Parsing log file {args.agent_log} ---")
        agent_results = parse_agent_log(args.agent_log)

        # Grab Step 0 accuracy from baselines to anchor the curve
        step_0_acc = None
        for b_name, b_data in baselines.items():
            if b_data and b_data[0][0] == 0:
                step_0_acc = b_data[0][1]
                break

        for agent_name, agent_data in agent_results.items():
            if step_0_acc is not None and (not agent_data or agent_data[0][0] != 0):
                agent_data.insert(0, (0, step_0_acc))
            results[agent_name] = agent_data
            print(f"[✓] Extracted {len(agent_data)} points for {agent_name}")

    else:
        print(f"\n--- Loading Full Model for Checkpoint Trajectory Evaluation ---")

        # Load correct architecture based on task
        if args.task == "classification":
            train_loader, test_loader, num_classes = get_loaders_cls(
                args.dataset, args.data_dir, args.batch_size, num_workers=2
            )
            model = DinoClassifier(device=device, num_classes=num_classes).to(device)
        else:
            train_loader, test_loader = get_loaders_seg(
                args.data_dir, args.batch_size, num_workers=2, use_features=False
            )
            model = DinoSegmenter(device, num_classes=150).to(device)
            model.transformer = model.pretrained.model

        if os.path.exists(args.checkpoint):
            ckpt = torch.load(args.checkpoint, map_location=device)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"], strict=False)
            else:
                model.load_state_dict(ckpt, strict=False)
        else:
            raise FileNotFoundError(f"Base weights not found at {args.checkpoint}")

        # Fix PyTorch alias registration
        if args.task == "segmentation":
            model.transformer = model.pretrained.model

        master_agent_path = os.path.join(
            args.agent_dir,
            (
                f"surgeon_ppo_best_segmentation.pth"
                if "seg" in args.dataset or args.task == "segmentation"
                else f"surgeon_ppo_best_50prune.pth"
            ),
        )

        if os.path.exists(master_agent_path):
            print(f"\nEvaluating Master Agent...")
            agent_data = evaluate_agent_trajectory(
                master_agent_path, target_x_vals, model, test_loader, device, args.task
            )
            results["AttentionSurgeon Agent"] = sorted(agent_data)
        else:
            print(f"[!] Master agent {master_agent_path} not found.")

    # 3. Plotting
    if not results:
        print("No data to plot!")
        return

    plt.figure(figsize=(12, 7))

    for label, data in results.items():
        x_vals = [dp[0] for dp in data]
        y_vals = [dp[1] for dp in data]

        if "Agent" in label:
            # Differentiate Best vs Latest if both are parsed
            color = "tab:red" if "Best" in label else "tab:orange"
            if "Best" not in label and "Latest" not in label:
                color = "tab:red"

            plt.plot(
                x_vals,
                y_vals,
                label=label,
                color=color,
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

    # Dynamic Axis Label
    metric_label = (
        "Validation mIoU (%)"
        if args.task == "segmentation"
        else "Validation Accuracy (%)"
    )

    plt.xlabel("Number of Pruned Heads", fontsize=12)
    plt.ylabel(metric_label, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower left", fontsize=10)

    save_path = os.path.join(
        args.save_dir, f"ultimate_comparison_{args.dataset}_{args.task}.png"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nSuccess! Ultimate comparison graph saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "segmentation"],
        default="classification",
    )
    parser.add_argument("--agent_dir", type=str, default="./checkpoints_rl")
    parser.add_argument("--dataset", type=str, default="imagenet100")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/dino_imagenet100_latest.pth"
    )
    parser.add_argument(
        "--baseline_cache",
        type=str,
        default="results/baselines/pruning_results_classification.pt",
    )
    parser.add_argument(
        "--agent_log",
        type=str,
        default=None,
        help="Path to a .out log to parse instantly.",
    )
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    main(args)
