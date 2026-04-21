import os
import glob
import argparse
import re
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from classification import DinoClassifier, get_loaders
from baseline import compute_head_magnitudes, get_pruning_mask, evaluate


def parse_out_file(filepath):
    """
    Parses a single .out file to find the final accuracy associated
    with the last '*** New Best Reward!' occurrence.
    """
    best_acc = None
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if "*** New Best Reward!" in line:
                # Check the immediately following line for the accuracy
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    # Regex to extract the floating point number right before the '%'
                    match = re.search(r"Final Step Acc: ([\d.]+)%", next_line)
                    if match:
                        best_acc = float(match.group(1))
    except Exception as e:
        print(f"[!] Error reading {filepath}: {e}")

    return best_acc


def main(args):
    # 1. Build the search pattern for the specified folder and dataset
    search_pattern = os.path.join(args.log_dir, f"*_{args.dataset}_*prune.out")
    files = glob.glob(search_pattern)

    if not files:
        print(f"No files matching pattern '{search_pattern}' found in {args.log_dir}.")
        return

    print(f"Found {len(files)} log files. Parsing...")

    data_points = []

    # 2. Extract max_prune and best accuracy from each file
    for filepath in files:
        filename = os.path.basename(filepath)
        match = re.search(r"_(\d+)prune\.out$", filename)

        if not match:
            print(
                f"[-] Skipping {filename}: Could not extract max_prune value from name."
            )
            continue

        max_prune = int(match.group(1))
        best_acc = parse_out_file(filepath)

        if best_acc is not None:
            data_points.append((max_prune, best_acc))
            print(f"[✓] {filename} -> max_prune: {max_prune}, Best Acc: {best_acc}%")
        else:
            print(f"[-] {filename} -> No 'New Best Reward!' found.")

    if not data_points:
        print("No valid data points successfully parsed. Exiting.")
        return

    # 3. Sort data by max_prune (x-axis) so the line plots correctly
    data_points.sort(key=lambda x: x[0])

    x_vals = [dp[0] for dp in data_points]
    rl_y_vals = [dp[1] for dp in data_points]

    # 4. Generate the Plot
    os.makedirs(args.save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(
        x_vals,
        rl_y_vals,
        marker="o",
        linestyle="-",
        color="tab:blue",
        linewidth=2,
        markersize=8,
    )

    # Annotate RL points
    for x, y in zip(x_vals, rl_y_vals):
        plt.annotate(
            f"{y:.1f}%",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color="tab:blue",
        )

    if args.baseline:
        print(f"\n--- Running Baseline Sweep for max_prune values: {x_vals} ---")
        device = torch.device(
            args.device
            if args.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load Dataset and Model
        train_loader, test_loader, num_classes = get_loaders(
            args.dataset, args.data_dir, args.batch_size, num_workers=2
        )
        model = DinoClassifier(device=device, num_classes=num_classes).to(device)

        # Load Weights
        if os.path.exists(args.checkpoint):
            ckpt = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded classifier weights from {args.checkpoint}")
        else:
            print(
                f"[!] Baseline Checkpoint {args.checkpoint} not found. Baseline will be meaningless."
            )

        criterion = nn.CrossEntropyLoss()

        # Compute Magnitudes once
        print("Computing head magnitudes on calibration set...")
        magnitudes = compute_head_magnitudes(
            model, train_loader, calib_batches=20, device=device
        )

        baseline_y_vals = []

        # Sweep through the exact same x_vals as the RL agent
        for n_prune in x_vals:
            print(f"Evaluating Baseline for max_prune = {n_prune}...")

            # Get boolean mask (12x12) from baseline logic
            head_mask = get_pruning_mask(magnitudes, n_prune)

            # Map the boolean mask (True=keep) to the float mask (1.0=keep) required by the hook
            flat_mask = head_mask.view(-1).float()
            model.set_mask(flat_mask)

            # Evaluate the full validation set
            _, acc = evaluate(model, test_loader, criterion, device)
            baseline_y_vals.append(acc)
            print(f"  -> Baseline Acc: {acc:.2f}%")

        # Plot Baseline Data
        plt.plot(
            x_vals,
            baseline_y_vals,
            marker="s",
            linestyle="--",
            color="tab:red",
            linewidth=2,
            markersize=8,
            label="Magnitude Baseline",
        )

        # Annotate Baseline points below the line
        for x, y in zip(x_vals, baseline_y_vals):
            plt.annotate(
                f"{y:.1f}%",
                (x, y),
                textcoords="offset points",
                xytext=(0, -15),
                ha="center",
                fontsize=9,
                color="tab:red",
            )

    # Styling for readability
    plt.title(
        f"Accuracy Collapse vs. Pruned Heads ({args.dataset})",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Max Pruned Heads (Out of 144)", fontsize=12)
    plt.ylabel("Best Validation Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower left")

    save_path = os.path.join(args.save_dir, f"max_prune_sweep_{args.dataset}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nSuccess! Graph saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse output logs and plot Accuracy vs Max Prune."
    )
    parser.add_argument(
        "--log_dir", type=str, required=True, help="Folder containing the .out files"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet100",
        help="Dataset name to filter files (e.g., imagenet100)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results",
        help="Folder to save the resulting plot",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="If set, runs and plots the magnitude baseline curve.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Path to dataset (required if --baseline is used)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/dino_imagenet100_latest.pth",
        help="Path to DINO weights (required if --baseline is used)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for baseline validation"
    )
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    main(args)
