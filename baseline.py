"""
baseline.py — Magnitude-based pruning baseline 
=====================================================================
Prunes the N attention heads with the lowest value-vector magnitude
across a calibration set, then evaluates classification accuracy.

Pruning strategy:
  - Score each head by the mean L2 norm of its value-token outputs
    (same metric used in head_activation.py → metrics["magnitude"])
  - Zero out the output-projection columns of the N lowest-scoring heads
    (model.transformer.blocks[l].attn.proj.weight[:, h*64:(h+1)*64] = 0)

Not based on RL or any training.
"""

import os
import argparse

import torch
import torch.nn as nn
from tqdm import tqdm

from classification import DinoClassifier, get_loaders

# ─── Constants (DINOv2 ViT-B/14) ──────────────────────────────────────────────
NUM_LAYERS  = 12
NUM_HEADS   = 12
HEAD_DIM    = 768 // NUM_HEADS   # 64
TOTAL_HEADS = NUM_LAYERS * NUM_HEADS  # 144
SEQ_LEN     = 1 + (224 // 14) ** 2   # 1 CLS + 256 patches = 257


# ─── 1. MAGNITUDE SCORING ─────────────────────────────────────────────────────

def compute_head_magnitudes(model: nn.Module, dataloader, num_batches: int, device) -> torch.Tensor:
    """
    Registers forward hooks on every block's QKV projection to extract
    value tokens, then computes the mean L2 norm per head over the
    calibration set.

    Returns
    -------
    magnitudes : Tensor (NUM_LAYERS, NUM_HEADS)
        Higher value → head is more "active" → less likely to be pruned.
    """
    accum = [torch.zeros(NUM_HEADS, device=device) for _ in range(NUM_LAYERS)]
    hooks = []

    def make_hook(layer_idx: int):
        def _hook(module, _input, output):
            # output: (B, N, 3 * embed_dim)  from nn.Linear(768, 2304)
            B, N, _ = output.shape
            qkv = output.reshape(B, N, 3, NUM_HEADS, HEAD_DIM)
            v   = qkv[:, :, 2, :, :].permute(0, 2, 1, 3)  # (B, H, N, head_dim)
            # Aggregate: L2 norm summed over B, N, head_dim → (H,)
            accum[layer_idx] += v.detach().norm(p=2, dim=(0, 2, 3))
        return _hook
    
    for i in range(NUM_LAYERS):
        h = model.transformer.blocks[i].attn.qkv.register_forward_hook(make_hook(i))
        hooks.append(h)

    model.eval()
    with torch.no_grad():
        for b_idx, (images, _) in enumerate(tqdm(dataloader, desc="[Calibration] scoring heads")):
            if b_idx >= num_batches:
                break
            model(images.to(device))

    for h in hooks:
        h.remove()

    magnitudes = torch.stack(accum) / num_batches  # (12, 12)
    return magnitudes


# ─── 2. PRUNING MASK ──────────────────────────────────────────────────────────

def get_pruning_mask(magnitudes: torch.Tensor, n_prune: int) -> torch.Tensor:
    """
    Returns a bool mask (NUM_LAYERS, NUM_HEADS):
      True  → keep this head
      False → prune this head

    Handles magnitude ties via explicit index selection so that exactly
    n_prune heads are always removed.
    """
    flat    = magnitudes.flatten()                # (144,)
    indices = flat.argsort()                      # ascending: lowest first
    mask    = torch.ones(TOTAL_HEADS, dtype=torch.bool, device=magnitudes.device)
    mask[indices[:n_prune]] = False               # mark bottom-n as pruned
    return mask.reshape(NUM_LAYERS, NUM_HEADS)


# ─── 3. APPLY PRUNING ─────────────────────────────────────────────────────────

def prune_heads(model: nn.Module, head_mask: torch.Tensor) -> None:
    """
    Zeros the output-projection columns that correspond to pruned heads.
    head_mask: (NUM_LAYERS, NUM_HEADS)  — False means prune.

    Concretely: for head h in layer l,
      model.transformer.blocks[l].attn.proj.weight[:, h*HEAD_DIM:(h+1)*HEAD_DIM] = 0
    This nullifies the head's contribution to the block output regardless
    of the Q/K/V computation.
    """
    with torch.no_grad():
        for l in range(NUM_LAYERS):
            for h in range(NUM_HEADS):
                if not head_mask[l, h]:
                    c0 = h * HEAD_DIM
                    c1 = c0 + HEAD_DIM
                    model.transformer.blocks[l].attn.proj.weight[:, c0:c1] = 0.0
    pruned = int((~head_mask).sum())
    print(f"  Zeroed projection columns for {pruned} heads.")


# ─── 4. EVALUATION ────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, loader, criterion, device) -> tuple[float, float]:
    """Returns (avg_loss, accuracy_%)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            correct    += outputs.max(1)[1].eq(labels).sum().item()
            total      += labels.size(0)
    return total_loss / len(loader), 100.0 * correct / total


# ─── 5. FLOPS ESTIMATE ────────────────────────────────────────────────────────

def attention_flops(n_active_heads: int, seq_len: int = SEQ_LEN) -> dict:
    """
    Estimates attention-layer FLOPs for the QK^T and AV operations only.
    Each head: 2 x seq_len² x head_dim multiplications.
    """
    flops_per_head  = 2 * seq_len * seq_len * HEAD_DIM
    flops_full      = TOTAL_HEADS  * flops_per_head
    flops_remaining = n_active_heads * flops_per_head
    ratio_remaining = n_active_heads / TOTAL_HEADS
    return {
        "flops_full"      : flops_full,
        "flops_remaining" : flops_remaining,
        "ratio_remaining" : ratio_remaining,
        "reduction_pct"   : (1.0 - ratio_remaining) * 100.0,
    }


# ─── 6. SUMMARY PRINTER ───────────────────────────────────────────────────────

def print_pruned_summary(head_mask: torch.Tensor, magnitudes: torch.Tensor) -> None:
    print("\n── Pruned heads (sorted by magnitude, lowest first) ─────────────")
    pruned_idx = (~head_mask).nonzero(as_tuple=False).tolist()
    pruned_idx.sort(key=lambda rc: magnitudes[rc[0], rc[1]].item())
    for layer, head in pruned_idx:
        mag = magnitudes[layer, head].item()
        print(f"  Layer {layer:2d}  Head {head:2d}  magnitude = {mag:.4f}")


# ─── 7. MAIN ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AttentionSurgeon — Magnitude pruning baseline"
    )
    parser.add_argument("--data_dir",      type=str,   default="./data")
    parser.add_argument("--checkpoint",    type=str,   default="checkpoints/dino_classifier_latest.pth")
    parser.add_argument("--num_classes",   type=int,   default=10)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--num_workers",   type=int,   default=2)
    parser.add_argument("--calib_batches", type=int,   default=20,
                        help="Number of train-set batches used to estimate head magnitudes.")
    parser.add_argument("--n_prune",       type=int,   default=72,
                        help="Number of heads to prune (default: 72 = 50%% of 144).")
    parser.add_argument("--device",        type=str,   default=None)
    parser.add_argument("--save_dir",      type=str,   default="results")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  AttentionSurgeon — Magnitude baseline")
    print(f"  Device : {device}  |  pruning {args.n_prune}/{TOTAL_HEADS} heads")
    print(f"{'='*60}\n")

    # ── Load model ──────────────────────────────────────────────────────────────
    model = DinoClassifier(device=device, num_classes=args.num_classes).to(device)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint : {args.checkpoint}")
    else:
        print(f"Warning: checkpoint not found at '{args.checkpoint}'. Using random linear head.")

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size, args.num_workers)
    criterion = nn.CrossEntropyLoss()

    # ── Baseline accuracy (without pruning) ──────────────────────────────
    print("\n[Step 1/4] Evaluating full model (no pruning)...")
    _, acc_before = evaluate(model, test_loader, criterion, device)
    print(f"  Accuracy BEFORE pruning : {acc_before:.2f}%")

    # ── Calibration pass: compute per-head magnitudes ───────────────────────────
    print("\n[Step 2/4] Computing head magnitudes on calibration set...")
    magnitudes = compute_head_magnitudes(model, train_loader, args.calib_batches, device)
    print(f"  Magnitude — min: {magnitudes.min():.4f}  "
          f"max: {magnitudes.max():.4f}  mean: {magnitudes.mean():.4f}")

    # ── Build mask and apply pruning ─────────────────────────────────────────────
    print(f"\n[Step 3/4] Pruning {args.n_prune} lowest-magnitude heads...")
    head_mask = get_pruning_mask(magnitudes, n_prune=args.n_prune)
    prune_heads(model, head_mask)

    # ── Evaluate pruned model ────────────────────────────────────────────────────
    print("\n[Step 4/4] Evaluating pruned model...")
    _, acc_after = evaluate(model, test_loader, criterion, device)

    # ── Compute FLOPs & reward ───────────────────────────────────────────────────
    n_active = int(head_mask.sum())
    flops    = attention_flops(n_active)
    # Reward: acc × FLOPs_saved_ratio  (higher ↔ accurate AND efficient)
    reward   = (acc_after / 100.0) * (1.0 - flops["ratio_remaining"])

    # ── Report ───────────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  Accuracy  BEFORE  : {acc_before:.2f}%")
    print(f"  Accuracy  AFTER   : {acc_after:.2f}%")
    print(f"  Accuracy  drop    : {acc_before - acc_after:.2f}%")
    print(f"  Heads kept        : {n_active}/{TOTAL_HEADS}")
    print(f"  Attn FLOPs (full) : {flops['flops_full']:,}")
    print(f"  Attn FLOPs (kept) : {flops['flops_remaining']:,}")
    print(f"  FLOPs reduction   : {flops['reduction_pct']:.1f}%")
    print(f"  Reward (acc x delta FLOP) : {reward:.4f}")
    print(f"{'─'*50}\n")

    print_pruned_summary(head_mask, magnitudes)

    # ── Save results ──────────────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, "baseline_magnitude.pt")
    torch.save(
        {
            "acc_before"      : acc_before,
            "acc_after"       : acc_after,
            "acc_drop"        : acc_before - acc_after,
            "n_pruned"        : args.n_prune,
            "n_active"        : n_active,
            "flops_reduction" : flops["reduction_pct"],
            "reward"          : reward,
            "head_mask"       : head_mask.cpu(),
            "magnitudes"      : magnitudes.cpu(),
        },
        out_path,
    )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
