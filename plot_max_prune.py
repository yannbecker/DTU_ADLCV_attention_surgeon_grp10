import os
import glob
import argparse
import re
import matplotlib.pyplot as plt


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
    y_vals = [dp[1] for dp in data_points]

    # 4. Generate the Plot
    os.makedirs(args.save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(
        x_vals,
        y_vals,
        marker="o",
        linestyle="-",
        color="tab:blue",
        linewidth=2,
        markersize=8,
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

    # Annotate points directly on the graph for easy reading
    for x, y in zip(x_vals, y_vals):
        plt.annotate(
            f"{y:.1f}%",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

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

    args = parser.parse_args()
    main(args)
