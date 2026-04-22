import os
import glob
import argparse
import re
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from classification import DinoClassifier, get_loaders
from baseline import compute_head_magnitudes, get_pruning_mask, evaluate
from agent import AdvancedActorCritic, PruningEnv


def parse_out_file(filepath):
    best_acc = None
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if "*** New Best Reward!" in line:
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    match = re.search(r"Final Step Acc: ([\d.]+)%", next_line)
                    if match:
                        best_acc = float(match.group(1))
    except Exception as e:
        print(f"[!] Error reading {filepath}: {e}")
    return best_acc


def evaluate_agent(agent_path, max_prune, model, test_loader, criterion, device):
    """Loads an RL agent, performs greedy pruning, and evaluates on the full test set."""
    agent = AdvancedActorCritic(n_metrics=7).to(device)
    agent.load_state_dict(torch.load(agent_path, map_location=device))
    agent.eval()

    # We use the test_loader so the proxy is drawn from the validation set
    env = PruningEnv(model, test_loader, device, max_pruning=max_prune)
    state = env.reset()

    with torch.no_grad():
        for _ in range(max_prune):
            dist, _ = agent(state["metric_grid"], state["scalars"], state["mask"])
            action = dist.probs.argmax()  # Greedy deterministic rollout
            state, _, done = env.step(action.item())
            if done:
                break

    # Evaluate the final mask on the FULL test set
    model.set_mask(env.mask)
    _, full_acc = evaluate(model, test_loader, criterion, device)
    return full_acc


def main(args):
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    os.makedirs(args.save_dir, exist_ok=True)

    # Store results dynamically: { 'series_name': [(x1, y1), (x2, y2)] }
    results = {}
    all_x_vals = set()  # Keep track of all evaluated max_prune values for the baseline

    # ---------------------------------------------------------
    # MODE: LOGS
    # ---------------------------------------------------------
    if args.mode == "logs":
        search_pattern = os.path.join(args.log_dir, f"*_{args.dataset}_*prune.out")
        files = glob.glob(search_pattern)
        print(f"Found {len(files)} log files.")

        log_data = []
        for filepath in files:
            match = re.search(r"_(\d+)prune\.out$", os.path.basename(filepath))
            if match:
                max_prune = int(match.group(1))
                best_acc = parse_out_file(filepath)
                if best_acc is not None:
                    log_data.append((max_prune, best_acc))
                    all_x_vals.add(max_prune)

        if log_data:
            results["Logs (Proxy Acc)"] = sorted(log_data)

    # ---------------------------------------------------------
    # MODE: CHECKPOINTS (Best and/or Latest)
    # ---------------------------------------------------------
    elif args.mode == "checkpoints":
        print(f"\n--- Loading Full Model for Checkpoint Evaluation ---")
        train_loader, test_loader, num_classes = get_loaders(
            args.dataset, args.data_dir, args.batch_size, num_workers=2
        )
        model = DinoClassifier(device=device, num_classes=num_classes).to(device)

        if os.path.exists(args.checkpoint):
            model.load_state_dict(
                torch.load(args.checkpoint, map_location=device)["model_state_dict"]
            )
            print(f"[✓] Loaded DINO weights: {args.checkpoint}")
        else:
            raise FileNotFoundError(f"Base DINO weights not found at {args.checkpoint}")

        criterion = nn.CrossEntropyLoss()

        for atype in args.agent_types:
            # Expected format: surgeon_ppo_best_30prune.pth
            pattern = os.path.join(args.agent_dir, f"surgeon_ppo_{atype}_*prune.pth")
            files = glob.glob(pattern)
            print(f"Found {len(files)} '{atype}' agents.")

            agent_data = []
            for f in files:
                match = re.search(
                    rf"surgeon_ppo_{atype}_(\d+)prune\.pth", os.path.basename(f)
                )
                if match:
                    max_prune = int(match.group(1))
                    print(f"Evaluating {atype} agent for max_prune = {max_prune}...")
                    full_acc = evaluate_agent(
                        f, max_prune, model, test_loader, criterion, device
                    )
                    agent_data.append((max_prune, full_acc))
                    all_x_vals.add(max_prune)
                    print(f"  -> Full Test Acc: {full_acc:.2f}%")

            if agent_data:
                results[f"Agent ({atype.capitalize()})"] = sorted(agent_data)

    # ---------------------------------------------------------
    # BASELINE (Magnitude Sweep)
    # ---------------------------------------------------------
    if args.baseline and all_x_vals:
        print(f"\n--- Running Baseline Sweep ---")
        # Ensure model is loaded if not already loaded by checkpoints mode
        if args.mode == "logs":
            train_loader, test_loader, num_classes = get_loaders(
                args.dataset, args.data_dir, args.batch_size, num_workers=2
            )
            model = DinoClassifier(device=device, num_classes=num_classes).to(device)
            model.load_state_dict(
                torch.load(args.checkpoint, map_location=device)["model_state_dict"]
            )
            criterion = nn.CrossEntropyLoss()

        print("Computing head magnitudes on calibration set...")
        magnitudes = compute_head_magnitudes(
            model, train_loader, num_batches=20, device=device
        )

        baseline_data = []
        for n_prune in sorted(list(all_x_vals)):
            print(f"Evaluating Baseline for max_prune = {n_prune}...")
            head_mask = get_pruning_mask(magnitudes, n_prune)
            model.set_mask(head_mask.view(-1).float())
            _, acc = evaluate(model, test_loader, criterion, device)
            baseline_data.append((n_prune, acc))

        results["Magnitude Baseline"] = baseline_data

    # ---------------------------------------------------------
    # PLOTTING
    # ---------------------------------------------------------
    if not results:
        print("No data to plot. Exiting.")
        return

    plt.figure(figsize=(10, 6))

    # Styling dictionaries for consistency
    colors = {
        "Logs (Proxy Acc)": "tab:blue",
        "Agent (Best)": "tab:orange",
        "Agent (Latest)": "tab:green",
        "Magnitude Baseline": "tab:red",
    }
    markers = {
        "Logs (Proxy Acc)": "x",
        "Agent (Best)": "o",
        "Agent (Latest)": "d",
        "Magnitude Baseline": "s",
    }
    lines = {
        "Logs (Proxy Acc)": ":",
        "Agent (Best)": "-",
        "Agent (Latest)": "-",
        "Magnitude Baseline": "--",
    }

    for label, data in results.items():
        x_vals = [dp[0] for dp in data]
        y_vals = [dp[1] for dp in data]

        plt.plot(
            x_vals,
            y_vals,
            label=label,
            color=colors.get(label, "black"),
            marker=markers.get(label, "o"),
            linestyle=lines.get(label, "-"),
            linewidth=2,
            markersize=8,
        )

        # Annotate points (staggered slightly to avoid overlap)
        y_offset = -15 if "Baseline" in label else 10
        for x, y in zip(x_vals, y_vals):
            plt.annotate(
                f"{y:.1f}%",
                (x, y),
                textcoords="offset points",
                xytext=(0, y_offset),
                ha="center",
                fontsize=8,
                color=colors.get(label, "black"),
            )

    plt.title(
        f"Accuracy Collapse vs. Pruned Heads ({args.dataset})",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Max Pruned Heads (Out of 144)", fontsize=12)
    plt.ylabel("Full Validation Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower left")

    save_path = os.path.join(
        args.save_dir, f"max_prune_sweep_comparison_{args.dataset}.png"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nSuccess! Comparison graph saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate RL Agents and Plot Pruning Pareto Curves."
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["logs", "checkpoints"],
        default="logs",
        help="Parse .out logs (fast) OR load agent checkpoints and evaluate full dataset (slow but accurate).",
    )
    parser.add_argument(
        "--agent_types",
        type=str,
        nargs="+",
        choices=["best", "latest"],
        default=["latest"],
        help="Which checkpoints to evaluate if mode=checkpoints. Can pass both: --agent_types best latest",
    )

    # Paths
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Folder with .out files (for logs mode)",
    )
    parser.add_argument(
        "--agent_dir",
        type=str,
        default="./checkpoints_rl",
        help="Folder with .pth agent weights (for checkpoints mode)",
    )
    parser.add_argument("--dataset", type=str, default="imagenet100")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/dino_imagenet100_latest.pth",
        help="Base DINO weights",
    )
    parser.add_argument("--save_dir", type=str, default="./results")

    # Evaluation options
    parser.add_argument(
        "--baseline", action="store_true", help="Run the magnitude baseline sweep."
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    main(args)
