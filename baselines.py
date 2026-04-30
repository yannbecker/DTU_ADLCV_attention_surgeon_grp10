"""
baselines.py — AttentionSurgeon: modular pruning baseline comparison
=====================================================================
Compares multiple attention-head pruning strategies for classification.
Runs a single HeadCensus pass (all 7 metrics at once), then sweeps
each strategy across requested pruning levels and plots the curves.

Usage
-----
python baselines.py \\
    --checkpoint  checkpoints/dino_classifier_latest.pth \\
    --dataset     imagenet100 \\
    --data_dir    $BLACKHOLE/imagenet100 \\
    --metrics     magnitude entropy distance taylor random greedy \\
    --prune_steps 0 12 24 36 48 60 72 84 96 \\
    --calib_batches 20 \\
    --save_dir    results/baselines

Available metrics
-----------------
  magnitude         L2 norm of value-vector outputs (QKV hook pass)
  entropy           Attention entropy              (HeadCensus idx 0)
  distance          Mean attention distance        (HeadCensus idx 1)
  rollout           Attention rollout flow         (HeadCensus idx 2)
  depth             Layer-depth proxy              (HeadCensus idx 3)
  residual_contrib  Per-head proj contribution L2  (HeadCensus idx 4)
  taylor            Taylor importance |act×grad|   (HeadCensus idx 5)
  intra_layer_rank  Within-layer Taylor rank       (HeadCensus idx 6)
  random            Uniform random scores
  greedy            Not implemented — legend placeholder only

Convention: higher score = more important = keep.
Pruning always removes the lowest-scoring heads first.
"""

import os
import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from classification import DinoClassifier, get_loaders
from head_activation import HeadCensus

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_LAYERS  = 12
NUM_HEADS   = 12
HEAD_DIM    = 64
TOTAL_HEADS = NUM_LAYERS * NUM_HEADS  # 144

# Maps metric name → index in HeadCensus.head_metrics[idx, layer, head]
METRIC_IDX: dict[str, int] = {
    "entropy":          0,
    "distance":         1,
    "rollout":          2,
    "depth":            3,
    "residual_contrib": 4,
    "taylor":           5,
    "intra_layer_rank": 6,
}

NOT_IMPLEMENTED: set[str] = {"greedy"}

ALL_METRICS: list[str] = (
    list(METRIC_IDX) + ["magnitude", "random"] + sorted(NOT_IMPLEMENTED)
)


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_magnitude(
    model: DinoClassifier,
    loader,
    calib_batches: int,
    device,
) -> torch.Tensor:
    """Hook QKV projections, accumulate mean L2 norm of value outputs → (12,12)."""
    accum = [torch.zeros(NUM_HEADS, device=device) for _ in range(NUM_LAYERS)]
    hooks = []

    def make_hook(layer_idx: int):
        def _hook(_module, _inp, output):
            B, N, _ = output.shape
            v = output.reshape(B, N, 3, NUM_HEADS, HEAD_DIM)[:, :, 2].permute(0, 2, 1, 3)
            accum[layer_idx] += v.detach().norm(p=2, dim=(0, 2, 3))
        return _hook

    for i in range(NUM_LAYERS):
        hooks.append(model.transformer.blocks[i].attn.qkv.register_forward_hook(make_hook(i)))

    model.eval()
    with torch.no_grad():
        for b_idx, (images, _) in enumerate(tqdm(loader, desc="  [magnitude] calibrate", leave=False)):
            if b_idx >= calib_batches:
                break
            model(images.to(device))

    for h in hooks:
        h.remove()

    return torch.stack(accum) / calib_batches  # (12, 12)


def score_random(device) -> torch.Tensor:
    return torch.rand(NUM_LAYERS, NUM_HEADS, device=device)


def get_all_scores(
    metrics: list[str],
    model: DinoClassifier,
    loader,
    calib_batches: int,
    device,
) -> dict[str, torch.Tensor | None]:
    """
    Compute importance scores for every requested metric.

    HeadCensus runs at most once (covers all 7 census metrics in a single pass).
    Magnitude needs its own QKV-hook pass. Random scores are sampled on-the-fly.
    Not-implemented strategies map to None.
    """
    scores: dict[str, torch.Tensor | None] = {}

    census_metrics = [m for m in metrics if m in METRIC_IDX]
    need_census    = bool(census_metrics)

    if need_census:
        print(f"  [census] collecting {census_metrics} in one pass...")
        criterion = nn.CrossEntropyLoss()
        census = HeadCensus(model, device)
        census.run_census(loader, num_batches=calib_batches)
        for m in census_metrics:
            scores[m] = census.head_metrics[METRIC_IDX[m]].clone()  # (12, 12)

    if "magnitude" in metrics:
        print("  [magnitude] running QKV calibration pass...")
        scores["magnitude"] = score_magnitude(model, loader, calib_batches, device)

    if "random" in metrics:
        scores["random"] = score_random(device)

    for m in metrics:
        if m in NOT_IMPLEMENTED:
            scores[m] = None

    return scores


# ── Pruning mask ──────────────────────────────────────────────────────────────

def get_pruning_mask(importance: torch.Tensor, n_prune: int) -> torch.Tensor:
    """
    Bool mask (12, 12): True = keep, False = prune.
    Zeroes the n_prune heads with the lowest importance scores.
    """
    flat    = importance.flatten()
    indices = flat.argsort()                               # ascending: weakest first
    mask    = torch.ones(TOTAL_HEADS, dtype=torch.bool, device=importance.device)
    mask[indices[:n_prune]] = False
    return mask.reshape(NUM_LAYERS, NUM_HEADS)


# ── Model mask helpers ────────────────────────────────────────────────────────

def apply_mask(model: DinoClassifier, head_mask: torch.Tensor) -> None:
    """Push a (12,12) bool/float mask into the model; hooks read it at forward time."""
    model.mask = head_mask.float().to(next(model.parameters()).device)


def reset_mask(model: DinoClassifier) -> None:
    """Restore the full unpruned mask without touching any weights."""
    device = next(model.parameters()).device
    model.mask = torch.ones(NUM_LAYERS, NUM_HEADS, device=device)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, loader, device) -> float:
    """Returns top-1 accuracy (%) on the given loader."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(1)
            correct += preds.eq(labels).sum().item()
            total   += labels.size(0)
    return 100.0 * correct / total


# ── Sweep ─────────────────────────────────────────────────────────────────────

def sweep_strategy(
    importance: torch.Tensor,
    model: DinoClassifier,
    loader,
    prune_steps: list[int],
    device,
) -> list[float]:
    """
    Evaluate accuracy at each pruning level.
    The model mask is updated in-place and restored after each level.
    No weights are modified.
    """
    results: list[float] = []
    for n_prune in tqdm(prune_steps, desc="    levels", leave=False):
        if n_prune > 0:
            apply_mask(model, get_pruning_mask(importance, n_prune))
        else:
            reset_mask(model)
        results.append(evaluate(model, loader, device))
    reset_mask(model)
    return results


# ── Plot ──────────────────────────────────────────────────────────────────────

_STYLES: dict[str, dict] = {
    "magnitude":         dict(linestyle="-",  marker="o"),
    "entropy":           dict(linestyle="-",  marker="s"),
    "distance":          dict(linestyle="-",  marker="^"),
    "rollout":           dict(linestyle="-",  marker="v"),
    "depth":             dict(linestyle="-",  marker="D"),
    "residual_contrib":  dict(linestyle="-",  marker="P"),
    "taylor":            dict(linestyle="-",  marker="*", markersize=8),
    "intra_layer_rank":  dict(linestyle="-",  marker="X"),
    "random":            dict(linestyle="--", marker="."),
    "greedy":            dict(linestyle=":",  marker=""),
}


def plot_results(
    results: dict[str, list[float] | None],
    prune_steps: list[int],
    save_dir: str,
) -> None:
    pct = [100.0 * n / TOTAL_HEADS for n in prune_steps]

    fig, ax = plt.subplots(figsize=(9, 5))
    for metric, accs in results.items():
        kw = _STYLES.get(metric, {})
        if accs is None:
            ax.plot([], [], label=f"{metric} (not implemented)", **kw)
        else:
            ax.plot(pct, accs, label=metric, **kw)

    ax.set_xlabel("Heads pruned (%)")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("AttentionSurgeon — Pruning strategy comparison")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "pruning_curves.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AttentionSurgeon — Modular pruning baseline comparison"
    )
    parser.add_argument("--checkpoint",    type=str, default="checkpoints/dino_classifier_latest.pth")
    parser.add_argument("--dataset",       type=str, default="imagenet100",
                        help="cifar10, imagenet100, …")
    parser.add_argument("--data_dir",      type=str, default="./data")
    parser.add_argument("--num_workers",   type=int, default=2)
    parser.add_argument("--batch_size",    type=int, default=32)
    parser.add_argument("--calib_batches", type=int, default=20,
                        help="Train-set batches used to estimate head importance.")
    parser.add_argument("--metrics",       nargs="+", default=["magnitude", "random"],
                        choices=ALL_METRICS,
                        help=f"Strategies to compare. Available: {ALL_METRICS}")
    parser.add_argument("--prune_steps",   nargs="+", type=int,
                        default=[0, 12, 24, 36, 48, 60, 72, 84, 96],
                        help="Number of heads removed at each evaluation point.")
    parser.add_argument("--device",        type=str, default=None)
    parser.add_argument("--save_dir",      type=str, default="results/baselines")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nAttentionSurgeon — Baseline comparison")
    print(f"  device      : {device}")
    print(f"  metrics     : {args.metrics}")
    print(f"  prune_steps : {args.prune_steps}")

    # ── Data & model ──────────────────────────────────────────────────────────
    train_loader, test_loader, num_classes = get_loaders(
        args.dataset, args.data_dir, args.batch_size, args.num_workers
    )
    model = DinoClassifier(device=device, num_classes=num_classes).to(device)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  checkpoint  : {args.checkpoint}")
    else:
        print(f"  Warning: '{args.checkpoint}' not found — using random linear head.")

    # ── Score all metrics (HeadCensus runs at most once) ──────────────────────
    print("\n[1/3] Computing head importance scores...")
    all_scores = get_all_scores(args.metrics, model, train_loader, args.calib_batches, device)

    # ── Sweep each strategy over pruning levels ───────────────────────────────
    print("\n[2/3] Sweeping pruning levels...")
    results: dict[str, list[float] | None] = {}

    for metric in args.metrics:
        importance = all_scores.get(metric)
        if importance is None:
            print(f"  {metric}: not implemented — skipped")
            results[metric] = None
            continue

        print(f"  {metric}:")
        accs = sweep_strategy(importance, model, test_loader, args.prune_steps, device)
        results[metric] = accs
        for n, acc in zip(args.prune_steps, accs):
            print(f"    prune={n:3d} ({100*n//TOTAL_HEADS:2d}%)  acc={acc:.2f}%")

    # ── Plot & save ───────────────────────────────────────────────────────────
    print("\n[3/3] Plotting and saving...")
    plot_results(results, args.prune_steps, args.save_dir)

    torch.save(
        {"results": results, "prune_steps": args.prune_steps, "metrics": args.metrics},
        os.path.join(args.save_dir, "pruning_results.pt"),
    )
    print(f"  Raw data  → {args.save_dir}/pruning_results.pt")


if __name__ == "__main__":
    main()