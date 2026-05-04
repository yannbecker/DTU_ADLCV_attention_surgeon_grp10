"""
baselines.py — AttentionSurgeon: modular pruning baseline comparison
=====================================================================
Compares multiple attention-head pruning strategies for classification.
Runs a single HeadCensus pass (all 7 metrics at once), then sweeps
each strategy across requested pruning levels and plots the curves.

Caching
-------
Results are saved as {metric: {n_prune: value}} in pruning_results.pt.
On re-run, already-computed (metric, n_prune) pairs are reused — only
the missing steps are evaluated. Use --force_recalc to override.
By default entropy and random are always recomputed:
  - entropy: direction was wrong before inversion was added
  - random: new prune_steps need fresh seed runs

Usage
-----
python3 baselines.py \\
    --checkpoint  checkpoints/dino_imagenet100_latest.pth \\
    --dataset     imagenet100 \\
    --data_dir    /dtu/datasets1/imagenet_object_localization_patched2019 \\
    --metrics     magnitude entropy distance taylor random greedy \\
    --prune_steps 0 2 4 6 8 10 12 24 36 48 60 72 84 96 \\
    --calib_batches 20 \\
    --n_random    10\\
    --proxy_batches 1 \\
    --force_recalc entropy random \\
    --device      cuda \\
    --save_dir    results/baselines

One line command: 
python3 baselines.py --checkpoint checkpoints/dino_imagenet100_latest.pth --dataset imagenet100 --data_dir /dtu/datasets1/imagenet_object_localization_patched2019 --metrics magnitude entropy distance taylor random greedy --prune_steps 0 2 4 6 8 10 12 24 36 48 60 72 84 96 --calib_batches 20 --n_random 10 --proxy_batches 1 --force_recalc entropy random --device cuda --save_dir results/baselines

Available metrics
-----------------
  magnitude         L2 norm of value-vector outputs (QKV hook pass)
  entropy           Attention entropy — inverted: prune highest (least specialised)
  distance          Mean attention distance        (HeadCensus idx 1)
  rollout           Attention rollout flow         (HeadCensus idx 2)
  depth             Layer-depth proxy              (HeadCensus idx 3)
  residual_contrib  Per-head proj contribution L2  (HeadCensus idx 4)
  taylor            Taylor importance |act×grad|   (HeadCensus idx 5)
  intra_layer_rank  Within-layer Taylor rank       (HeadCensus idx 6)
  random            Uniform random scores — averaged over --n_random seeds
  greedy            Iterative least-damage removal (proxy-loss-guided)

Convention: higher score = more important = keep.
Pruning always removes the lowest-scoring heads first.
Exception: entropy is negated before ranking so high-entropy (uniform)
heads are pruned first.
"""

import math
import os
import sys
import argparse
from typing import Any

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# --- Task 1: Classification ---
from classification import DinoClassifier
from classification import get_loaders as get_loaders_cls

# --- Task 2: Segmentation ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "DPT_segmentation"))
from segmentation import DinoSegmenter
from segmentation import get_loaders as get_loaders_seg
from torchmetrics.classification import MulticlassJaccardIndex

from head_activation import HeadCensus

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_LAYERS = 12
NUM_HEADS = 12
HEAD_DIM = 64
TOTAL_HEADS = NUM_LAYERS * NUM_HEADS  # 144

METRIC_IDX: dict[str, int] = {
    "entropy": 0,
    "distance": 1,
    "rollout": 2,
    "depth": 3,
    "residual_contrib": 4,
    "taylor": 5,
    "intra_layer_rank": 6,
}

# Negated before ranking: high raw value → low importance → prune first
INVERT_METRICS: set[str] = {"entropy"}

ALL_METRICS: list[str] = list(METRIC_IDX) + ["magnitude", "random", "greedy"]


# ── Scoring ───────────────────────────────────────────────────────────────────


def score_magnitude(
    model: DinoClassifier, loader, calib_batches: int, device, task: str
) -> torch.Tensor:
    """Hook QKV projections, accumulate mean L2 norm of value outputs → (12,12)."""
    accum = [torch.zeros(NUM_HEADS, device=device) for _ in range(NUM_LAYERS)]
    hooks = []

    def make_hook(layer_idx: int):
        def _hook(_module, _inp, output):
            B, N, _ = output.shape
            v = output.reshape(B, N, 3, NUM_HEADS, HEAD_DIM)[:, :, 2].permute(
                0, 2, 1, 3
            )
            accum[layer_idx] += v.detach().norm(p=2, dim=(0, 2, 3))

        return _hook

    for i in range(NUM_LAYERS):
        hooks.append(
            model.transformer.blocks[i].attn.qkv.register_forward_hook(make_hook(i))
        )

    model.eval()
    with torch.no_grad():
        for b_idx, batch in enumerate(
            tqdm(loader, desc="  [magnitude] calibrate", leave=False)
        ):
            if b_idx >= calib_batches:
                break

            # Dynamically grab images (always the first element) to avoid unpacking errors
            images = batch[0].to(device)

            if task == "segmentation":
                model(images, features=False)
            else:
                model(images)

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
    task: str,
) -> dict[str, torch.Tensor]:
    """
    Compute importance scores for every requested static metric.
    HeadCensus runs at most once (all 7 census metrics in one pass).
    INVERT_METRICS are negated so "prune lowest" is always correct.
    """
    scores: dict[str, torch.Tensor] = {}
    census_metrics = [m for m in metrics if m in METRIC_IDX]

    if census_metrics:
        print(f"  [census] collecting {census_metrics} in one pass...")
        census = HeadCensus(model, device)
        census.run_census(loader, num_batches=calib_batches)
        for m in census_metrics:
            raw = census.head_metrics[METRIC_IDX[m]].clone()
            scores[m] = -raw if m in INVERT_METRICS else raw

    if "magnitude" in metrics:
        print("  [magnitude] running QKV calibration pass...")
        scores["magnitude"] = score_magnitude(
            model, loader, calib_batches, device, task
        )

    return scores


# ── Pruning mask ──────────────────────────────────────────────────────────────


def get_pruning_mask(importance: torch.Tensor, n_prune: int) -> torch.Tensor:
    """
    Bool mask (12, 12): True = keep, False = prune.
    Removes the n_prune lowest-importance heads.
    Guarantees at least one head survives per layer.
    """
    flat = importance.flatten()
    indices = flat.argsort()
    mask = torch.ones(TOTAL_HEADS, dtype=torch.bool, device=importance.device)
    mask[indices[:n_prune]] = False
    mask = mask.reshape(NUM_LAYERS, NUM_HEADS)

    for l in range(NUM_LAYERS):
        if not mask[l].any():
            mask[l, importance[l].argmax()] = True

    return mask


# ── Model mask helpers ────────────────────────────────────────────────────────


def apply_mask(model: nn.Module, head_mask: torch.Tensor, task: str) -> None:
    device = next(model.parameters()).device
    # Both DinoClassifier and DinoSegmenter read from model.mask
    model.mask = head_mask.float().to(device)


def reset_mask(model: nn.Module, task: str) -> None:
    device = next(model.parameters()).device
    # Reset to all ones
    model.mask = torch.ones(NUM_LAYERS, NUM_HEADS, device=device)


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate(model: nn.Module, loader, device, task: str) -> float:
    model.eval()
    if task == "classification":
        correct, total = 0, 0
        with torch.no_grad():
            for batch in loader:
                # Safely extract images and labels regardless of batch tuple length
                images, labels = batch[0].to(device), batch[1].to(device)
                correct += model(images).argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        return 100.0 * correct / total
    else:
        # Segmentation: compute Jaccard Index (mIoU)
        metric = MulticlassJaccardIndex(num_classes=150, ignore_index=-100).to(device)
        with torch.no_grad():
            for batch in loader:
                # Safely extract images and labels
                images, labels = batch[0].to(device), batch[1].to(device)
                labels[(labels < 0) | (labels >= 150)] = -100
                _, outputs = model(images, features=False)
                metric.update(outputs, labels)
        res = metric.compute().item() * 100.0
        metric.reset()
        return res


def evaluate_proxy_loss(
    model: nn.Module, loader, criterion, proxy_batches: int, device, task: str
) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if b_idx >= proxy_batches:
                break

            # Safely extract images and labels
            images, labels = batch[0].to(device), batch[1].to(device)

            if task == "segmentation":
                labels[(labels < 0) | (labels >= 150)] = -100
                _, outputs = model(images, features=False)
                loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            total += loss.item()
            n += 1
    return total / max(n, 1)


# ── Cache I/O ─────────────────────────────────────────────────────────────────


def load_cache(path: str) -> dict[str, dict[int, Any]]:
    """
    Load saved results and return cache[metric][n_prune] = value.
      - Most metrics: value is float (accuracy %)
      - random:       value is list[float], one entry per seed

    Handles both:
      - Old format: {"results": {metric: list[float]}, "prune_steps": [...]}
      - New format: {"results": {metric: {n_prune: value}}}
    """
    if not os.path.exists(path):
        return {}

    data = torch.load(path, map_location="cpu", weights_only=False)
    raw = data.get("results", {})

    # Detect format by inspecting the first non-None value
    first = next((v for v in raw.values() if v is not None), None)
    if first is None:
        return {}

    if isinstance(first, dict):
        # New format — keys are already ints (saved via torch which preserves types)
        return {
            m: {int(k): v for k, v in d.items()}
            for m, d in raw.items()
            if d is not None
        }

    # Old list format — convert using the stored prune_steps
    old_steps: list[int] = data.get("prune_steps", [])
    cache: dict[str, dict[int, Any]] = {}
    for metric, result in raw.items():
        if result is None:
            continue
        if isinstance(result[0], list):
            # random: result[seed_idx][step_idx] → cache[n_prune] = [acc_s0, acc_s1, ...]
            cache[metric] = {
                n: [result[s][i] for s in range(len(result))]
                for i, n in enumerate(old_steps)
            }
        else:
            cache[metric] = {n: result[i] for i, n in enumerate(old_steps)}

    n_entries = sum(len(v) for v in cache.values())
    print(f"  [cache] loaded {n_entries} (metric, step) entries from {path}")
    return cache


def save_cache(path: str, cache: dict[str, dict[int, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"results": cache, "format": "dict"}, path)


# ── Compute helpers ───────────────────────────────────────────────────────────


def compute_steps(
    importance: torch.Tensor,
    model: DinoClassifier,
    loader,
    steps: list[int],
    device,
    task: str,
) -> dict[int, float]:
    """Evaluate accuracy at each n_prune in steps. Returns {n_prune: acc}."""
    out: dict[int, float] = {}
    for n in tqdm(steps, desc="    levels", leave=False):
        (
            apply_mask(model, get_pruning_mask(importance, n), task)
            if n > 0
            else reset_mask(model, task)
        )
        out[n] = evaluate(model, loader, device, task)
    reset_mask(model, task)
    return out


# ── Greedy sweep ──────────────────────────────────────────────────────────────


def sweep_greedy(
    model: DinoClassifier,
    proxy_loader,
    test_loader,
    prune_steps: list[int],
    proxy_batches: int,
    device,
    task: str,
) -> dict[int, float]:
    """
    Iterative least-damage greedy pruning.

    At each step, every remaining active head is tried; the one whose removal
    causes the smallest proxy-loss increase is permanently pruned.
    Accuracy on test_loader is recorded only at checkpoint steps (prune_steps).

    Uses loss on proxy_batches batches for head selection: more stable than
    accuracy with very few samples. proxy_batches=1 (32 images) is enough for
    head ranking and keeps the runtime manageable.

    Returns {n_prune: accuracy} for all steps in prune_steps.
    Greedy cannot fill individual steps: re-run from scratch if any step is missing.
    """
    criterion = nn.CrossEntropyLoss()
    prune_set = set(prune_steps)
    max_prune = max(prune_steps)
    current_mask = torch.ones(NUM_LAYERS, NUM_HEADS, dtype=torch.bool, device=device)
    results: dict[int, float] = {}

    criterion = (
        nn.CrossEntropyLoss(ignore_index=-100)
        if task == "segmentation"
        else nn.CrossEntropyLoss()
    )
    prune_set = set(prune_steps)
    max_prune = max(prune_steps)
    current_mask = torch.ones(NUM_LAYERS, NUM_HEADS, dtype=torch.bool, device=device)
    results: dict[int, float] = {}

    if 0 in prune_set:
        reset_mask(model, task)
        results[0] = evaluate(model, test_loader, device, task)

    for step in tqdm(range(1, max_prune + 1), desc="    greedy steps"):
        best_loss, best_l, best_h = float("inf"), -1, -1

        for l in range(NUM_LAYERS):
            if current_mask[l].sum() <= 1:
                continue
            for h in range(NUM_HEADS):
                if not current_mask[l, h]:
                    continue
                trial = current_mask.clone()
                trial[l, h] = False
                apply_mask(model, trial.float(), task)
                loss = evaluate_proxy_loss(
                    model, proxy_loader, criterion, proxy_batches, device, task
                )
                if loss < best_loss:
                    best_loss, best_l, best_h = loss, l, h

        if best_l == -1:
            print(
                f"  [greedy] early stop at step {step}: all remaining heads are layer-sole."
            )
            break

        current_mask[best_l, best_h] = False
        apply_mask(model, current_mask.float(), task)

        if step in prune_set:
            results[step] = evaluate(model, test_loader, device, task)

    reset_mask(model, task)
    return results


# ── Plot ──────────────────────────────────────────────────────────────────────

_STYLES: dict[str, dict] = {
    "magnitude": dict(linestyle="-", marker="o"),
    "entropy": dict(linestyle="-", marker="s"),
    "distance": dict(linestyle="-", marker="^"),
    "rollout": dict(linestyle="-", marker="v"),
    "depth": dict(linestyle="-", marker="D"),
    "residual_contrib": dict(linestyle="-", marker="P"),
    "taylor": dict(linestyle="-", marker="*", markersize=8),
    "intra_layer_rank": dict(linestyle="-", marker="X"),
    "random": dict(linestyle="--", marker="."),
    "greedy": dict(linestyle=":", marker="^", markersize=6),
}


def plot_results(
    results: dict[str, list[float] | list[list[float]] | None],
    prune_steps: list[int],
    save_dir: str,
    task: str,
) -> None:
    """
    results values:
      list[float]        → single line
      list[list[float]]  → mean ± std band (random, multiple seeds)
      None               → legend-only placeholder
    """
    pct = [100.0 * n / TOTAL_HEADS for n in prune_steps]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [p["color"] for p in plt.rcParams["axes.prop_cycle"]]

    for idx, (metric, data) in enumerate(results.items()):
        col = colors[idx % len(colors)]
        kw = _STYLES.get(metric, {})

        if data is None:
            ax.plot([], [], label=f"{metric} (not implemented)", color=col, **kw)
        elif isinstance(data[0], list):
            arr = np.array(data)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            ax.plot(pct, mean, label=f"{metric} (n={len(data)})", color=col, **kw)
            ax.fill_between(pct, mean - std, mean + std, alpha=0.2, color=col)
        else:
            ax.plot(pct, data, label=metric, color=col, **kw)

    metric_name = "mIoU" if task == "segmentation" else "Top-1 Accuracy"
    ax.set_xlabel("Heads pruned (%)")
    ax.set_ylabel(f"{metric_name} (%)")
    ax.set_title(f"AttentionSurgeon — Pruning strategy comparison ({task})")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f"pruning_curves_{task}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AttentionSurgeon — Modular pruning baseline comparison"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "segmentation"],
        default="classification",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/dino_classifier_latest.pth"
    )
    parser.add_argument("--dataset", type=str, default="imagenet100")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--calib_batches",
        type=int,
        default=20,
        help="Train-set batches used to estimate head importance scores.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["magnitude", "random"],
        choices=ALL_METRICS,
        help=f"Strategies to compare. Available: {ALL_METRICS}",
    )
    parser.add_argument(
        "--prune_steps",
        nargs="+",
        type=int,
        default=[0, 12, 24, 36, 48, 60, 72, 84, 96],
        help="Number of heads removed at each evaluation point.",
    )
    parser.add_argument(
        "--n_random",
        type=int,
        default=5,
        help="Seeds to average for the random strategy.",
    )
    parser.add_argument(
        "--proxy_batches",
        type=int,
        default=1,
        help="Batches per head-trial in greedy (1 = fast, higher = more stable).",
    )
    parser.add_argument(
        "--force_recalc",
        nargs="*",
        default=["entropy", "random"],
        choices=ALL_METRICS,
        help="Metrics to recompute even if cached. Default: entropy random.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="results/baselines")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    force_recalc = set(args.force_recalc or [])
    cache_path = os.path.join(args.save_dir, f"pruning_results_{args.task}.pt")

    print(f"\nAttentionSurgeon — Baseline comparison ({args.task})")
    print(f"  device        : {device}")
    print(f"  metrics       : {args.metrics}")
    print(f"  prune_steps   : {args.prune_steps}")
    print(f"  force_recalc  : {sorted(force_recalc)}")

    # ── Load existing cache ────────────────────────────────────────────────────
    cache = load_cache(cache_path)

    # ── Initialize Correct Task Model ─────────────────────────────────────────
    if args.task == "classification":
        train_loader, test_loader, num_classes = get_loaders_cls(
            args.dataset, args.data_dir, args.batch_size, args.num_workers
        )
        model = DinoClassifier(device=device, num_classes=num_classes).to(device)
    else:
        train_loader, test_loader = get_loaders_seg(
            args.data_dir,
            args.batch_size,
            num_workers=args.num_workers,
            use_features=False,
        )
        model = DinoSegmenter(device, num_classes=150).to(device)
        model.transformer = model.pretrained.model  # Alias for initialization safety

    # ── Load Weights ──────────────────────────────────────────────────────────
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        print(f"  checkpoint    : {args.checkpoint}")
    else:
        print(f"  Warning: '{args.checkpoint}' not found.")

    # Apply the alias AFTER weights are loaded (fixes PyTorch state_dict issue)
    if args.task == "segmentation":
        model.transformer = model.pretrained.model

    # ── Score static metrics (HeadCensus at most once) ────────────────────────
    static_metrics = [m for m in args.metrics if m not in ("random", "greedy")]
    # Only need to run census/magnitude for metrics that have missing or forced steps
    need_score = [
        m
        for m in static_metrics
        if m in force_recalc or any(n not in cache.get(m, {}) for n in args.prune_steps)
    ]

    static_metrics = [m for m in args.metrics if m not in ("random", "greedy")]
    need_score = [
        m
        for m in static_metrics
        if m in force_recalc or any(n not in cache.get(m, {}) for n in args.prune_steps)
    ]

    print("\n[1/3] Computing head importance scores...")
    all_scores = get_all_scores(
        need_score, model, train_loader, args.calib_batches, device, args.task
    )

    print("\n[2/3] Sweeping pruning levels...")
    plot_data: dict[str, list[float] | list[list[float]] | None] = {}

    for metric in args.metrics:
        if metric not in ("random", "greedy"):
            importance = all_scores.get(metric)
            if importance is None and metric not in cache:
                plot_data[metric] = None
                continue

            if metric in force_recalc:
                missing = args.prune_steps
                cache.pop(metric, None)
            else:
                missing = [
                    n for n in args.prune_steps if n not in cache.get(metric, {})
                ]

            if missing:
                print(f"  {metric}: computing {len(missing)} step(s)...")
                new_vals = compute_steps(
                    importance, model, test_loader, missing, device, args.task
                )
                cache.setdefault(metric, {}).update(new_vals)

            accs = [cache[metric][n] for n in args.prune_steps]
            plot_data[metric] = accs
            for n, acc in zip(args.prune_steps, accs):
                print(f"    prune={n:3d} ({100*n//TOTAL_HEADS:2d}%)  res={acc:.2f}%")

        elif metric == "random":
            print(f"  random ({args.n_random} seeds):")
            if "random" in force_recalc:
                cache.pop("random", None)
            cache.setdefault("random", {})

            runs: list[list[float]] = []
            for seed in range(args.n_random):
                torch.manual_seed(seed)
                imp = score_random(device)
                missing = [
                    n
                    for n in args.prune_steps
                    if len(cache["random"].get(n, [])) <= seed
                ]

                if missing:
                    new_vals = compute_steps(
                        imp, model, test_loader, missing, device, args.task
                    )
                    for n, acc in new_vals.items():
                        lst = cache["random"].setdefault(n, [])
                        while len(lst) < seed:
                            lst.append(float("nan"))
                        lst.append(acc)

                seed_accs = [cache["random"][n][seed] for n in args.prune_steps]
                runs.append(seed_accs)
                valid = [a for a in seed_accs if not math.isnan(a)]
                print(f"    seed {seed}  mean={sum(valid)/len(valid):.2f}%")

            plot_data["random"] = runs

        elif metric == "greedy":
            cached_g = cache.get("greedy", {})
            all_cached = all(n in cached_g for n in args.prune_steps)

            if "greedy" not in force_recalc and all_cached:
                plot_data["greedy"] = [cached_g[n] for n in args.prune_steps]
            else:
                new_results = sweep_greedy(
                    model,
                    train_loader,
                    test_loader,
                    args.prune_steps,
                    args.proxy_batches,
                    device,
                    args.task,
                )
                cache["greedy"] = new_results
                accs = [new_results.get(n, float("nan")) for n in args.prune_steps]
                plot_data["greedy"] = accs
                for n, acc in zip(args.prune_steps, accs):
                    print(
                        f"    prune={n:3d} ({100*n//TOTAL_HEADS:2d}%)  res={acc:.2f}%"
                    )

    save_cache(cache_path, cache)
    print(f"\n  Cache saved → {cache_path}")
    print("\n[3/3] Plotting...")
    plot_results(plot_data, args.prune_steps, args.save_dir, args.task)


if __name__ == "__main__":
    main()
