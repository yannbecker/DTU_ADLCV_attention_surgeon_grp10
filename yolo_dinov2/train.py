# =============================================================================
# yolo_dinov2/train.py — AttentionSurgeon: training script for
#                        DINOv2 ViT-B/14-reg + YOLOv8 Detect head (518×518)
# =============================================================================
# Architecture:
#   DINOv2 ViT-B/14-reg (frozen)          → (B, 768, 37, 37)
#   Single-scale Detect head (trainable)  stride=14, 37×37 feature grid
#
# Training strategy:
#   - Raw COCO images (518×518 float32 [0,1]) so real augmentation is applied.
#   - Backbone normalises internally; we only optimise the Detect head.
#   - Loss: v8DetectionLoss (ultralytics). Always call loss_raw.sum() before
#     .backward() because different ultralytics versions return either a scalar
#     or a 3-element [box, cls, dfl] vector.
#   - Linear LR warmup → cosine decay; early stopping with burn-in period
#     (mAP does not appear until ~15–20 epochs).
#
# Usage:
#   python -m yolo_dinov2.train --datadir $BLACKHOLE/COCO
# =============================================================================

from __future__ import annotations

import os
import argparse

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .detector import DinoPrunableDetector, build_criterion, _build_batch_dict
from .dataset  import get_coco_loaders_v2, IDX_TO_CAT, IMG_SIZE

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    COCO_OK = True
except ImportError:
    COCO_OK = False

try:
    from torchvision.ops import batched_nms, box_iou
    NMS_OK = True
except ImportError:
    NMS_OK = False


# =============================================================================
# 1.  TRAINING LOOP
# =============================================================================

def train_one_epoch(
    model:     DinoPrunableDetector,
    criterion,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    scaler:    amp.GradScaler | None = None,
) -> tuple[float, float, float, float]:
    """
    One training epoch over raw COCO images.

    Returns:
        (avg_total_loss, avg_box_loss, avg_cls_loss, avg_dfl_loss)
    """
    model.train()
    totals = torch.zeros(4)   # [total, box, cls, dfl]
    n = len(loader)

    for batch in tqdm(loader, desc="Train", leave=False):
        imgs, targets = batch
        # non_blocking=True overlaps H→D transfer with previous GPU work
        # (only effective because pin_memory=True is set in the DataLoader).
        imgs = imgs.to(device, non_blocking=True)            # (B, 3, 518, 518)

        # Backbone is frozen — run it without autograd to save memory and time.
        # AMP is applied to the whole forward (backbone + head) because the
        # backbone runs under no_grad anyway; autocast just enables TF32/FP16
        # for the frozen backbone's matmuls which are the main compute cost.
        # head([feat]) below still builds a graph through the head parameters.
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(scaler is not None)):
                feat = model.extract_features(imgs)          # (B, 768, 37, 37)
        feat = feat.detach()

        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(scaler is not None)):
            preds = model.head([feat])                       # training: list of tensors

            batch_dict = _build_batch_dict(targets, device)
            loss_raw, items = criterion(preds, batch_dict)

        # v8DetectionLoss may return a scalar or a 3-element vector depending
        # on the ultralytics version.  Always reduce to scalar before backward.
        loss: torch.Tensor = loss_raw.sum()

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(list(model.head.parameters()), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.head.parameters()), max_norm=10.0)
            optimizer.step()

        totals[0] += loss.item()
        totals[1:] += items.detach().cpu()[:3]

    return (
        (totals[0] / n).item(),
        (totals[1] / n).item(),
        (totals[2] / n).item(),
        (totals[3] / n).item(),
    )


# =============================================================================
# 1b. VALIDATION LOSS  (train-mode subset pass — avoids full epoch overhead)
# =============================================================================

def _compute_val_loss(
    model:     DinoPrunableDetector,
    loader:    DataLoader,
    criterion,
    device:    torch.device,
    n_batches: int = 50,
) -> float:
    """
    Estimate validation loss by running the Detect head in train mode over the
    first ``n_batches`` batches of *loader*.

    Why train mode?  v8DetectionLoss needs the raw pre-decode feature tensors
    returned by the Detect head.  In eval mode the head returns decoded boxes.

    BN corruption guard: all BatchNorm layers are pinned to eval mode so their
    running statistics are not updated during this pass.

    Returns:
        Average total loss over the sampled batches, or 0.0 if loader is empty.
    """
    model.train()
    # Freeze BN running stats — we don't want val batches shifting them
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()

    total = 0.0
    count = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            imgs, targets = batch
            imgs = imgs.to(device, non_blocking=True)

            feat  = model.extract_features(imgs).detach()
            preds = model.head([feat])
            batch_dict = _build_batch_dict(targets, device)
            loss_raw, _ = criterion(preds, batch_dict)
            total += loss_raw.sum().item()
            count += 1

    model.eval()
    return total / count if count > 0 else 0.0


# =============================================================================
# 2.  DECODE + NMS
# =============================================================================

def decode_v8_output(
    raw_out,
    conf_thres: float = 0.01,
    iou_thres:  float = 0.45,
    max_det:    int   = 300,
) -> list[torch.Tensor]:
    """
    Decode the ultralytics Detect head output and apply per-class NMS.

    The Detect head in eval mode returns ``(y, x_dict)`` where ``y`` has
    shape ``(B, 4 + nc, total_anchors)``:
        - channels 0:4  — xyxy boxes in *input-image pixel space* (518 px)
        - channels 4:   — class probabilities, already sigmoid-ed by Detect

    Args:
        raw_out:    Output of DinoPrunableDetector in eval mode.
                    Either ``(y, x_dict)`` or just ``y``.
        conf_thres: Minimum class score to keep a candidate box.
        iou_thres:  IoU threshold for NMS.
        max_det:    Maximum detections per image.

    Returns:
        List of length B.  Each element is a FloatTensor of shape (N, 6):
        ``[x1, y1, x2, y2, score, cls_idx]``  with coords in 518-px space.

    Raises:
        ImportError: if torchvision is not available.
    """
    if not NMS_OK:
        raise ImportError(
            "torchvision is required for NMS: pip install torchvision"
        )

    # Accept both (y, x_dict) tuple (eval mode) and plain tensor (training mode)
    pred: torch.Tensor = raw_out[0] if isinstance(raw_out, (tuple, list)) else raw_out

    results: list[torch.Tensor] = []

    for p in pred:                           # iterate over batch dimension (B, 4+nc, A)
        boxes   = p[:4].T                    # (A, 4)  xyxy in 518-px space
        probs   = p[4:]                      # (nc, A) already sigmoid
        scores, cls_ids = probs.max(dim=0)   # (A,) each

        mask = scores > conf_thres
        if not mask.any():
            results.append(torch.zeros((0, 6), device=pred.device))
            continue

        b = boxes[mask]
        s = scores[mask]
        c = cls_ids[mask]

        keep = batched_nms(b, s, c, iou_thres)[:max_det]

        det = torch.cat(
            [b[keep], s[keep, None], c[keep, None].float()],
            dim=1,
        )                                    # (N, 6)
        results.append(det)

    return results


# =============================================================================
# 3.  EVALUATION  (COCO mAP)
# =============================================================================

def evaluate_v2(
    model:   DinoPrunableDetector,
    loader:  DataLoader,
    device:  torch.device,
    datadir: str | None = None,
) -> tuple[float, float, float]:
    """
    Official COCOeval mAP + Mean IoU on the val split.

    Boxes from the Detect head are in 518-px space (xyxy).  We scale them
    back to original image coordinates before adding to coco_results so that
    COCOeval compares in the correct coordinate system.

    Mean IoU: for each GT box, the max IoU with any predicted box is computed
    (in 518-px space); these are averaged over all GT instances in the split.
    Images with no predictions contribute 0 IoU for each of their GT boxes.

    Returns:
        (mAP@0.5,  mAP@0.5:0.95,  mean_iou)
        All 0.0 if pycocotools is unavailable or no detections are made.
    """
    if not COCO_OK:
        print("pycocotools not found — mAP skipped.")
        return 0.0, 0.0, 0.0

    model.eval()
    coco_results: list[dict] = []

    # Resolve COCO ground truth object
    coco_gt: COCO | None = getattr(loader.dataset, "coco", None)
    if coco_gt is None and datadir is not None:
        ann_file = os.path.join(datadir, "annotations", "instances_val2017.json")
        if os.path.exists(ann_file):
            coco_gt = COCO(ann_file)
        else:
            print(f"Annotation file not found at {ann_file} — mAP skipped.")

    # Mean IoU accumulators (computed in 518-px space)
    iou_sum:   float = 0.0
    iou_count: int   = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            imgs, targets = batch
            imgs = imgs.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                out  = model(imgs)                       # eval: (y, x_dict)
            # Lower threshold → denser recall → more accurate COCOeval PR curve.
            # COCOeval sweeps its own thresholds so submitting more detections
            # (with low score) only helps; the extra FPs are penalised by the
            # precision side of the curve.
            dets = decode_v8_output(out, conf_thres=0.001, iou_thres=0.45, max_det=300)

            for det, tgt in zip(dets, targets):
                # ── Mean IoU ────────────────────────────────────────────────
                # GT boxes are stored normalised [0,1] xyxy; scale to 518-px.
                gt_norm = tgt["boxes"]                              # (G, 4)
                if gt_norm.shape[0] > 0:
                    gt_px = gt_norm.clone()
                    gt_px[:, [0, 2]] *= IMG_SIZE
                    gt_px[:, [1, 3]] *= IMG_SIZE
                    gt_px = gt_px.to(device)

                    if det.shape[0] > 0 and NMS_OK:
                        iou_mat = box_iou(gt_px, det[:, :4])       # (G, N_pred)
                        max_iou = iou_mat.max(dim=1).values         # (G,)
                        iou_sum   += max_iou.sum().item()
                    # else: no predictions → IoU = 0 for every GT box
                    iou_count += gt_norm.shape[0]

                # ── COCOeval ─────────────────────────────────────────────────
                if coco_gt is None:
                    continue

                img_id   = tgt["image_id"]
                img_info = coco_gt.imgs[img_id]

                # Scale from 518-px space back to original image coordinates
                sx = img_info["width"]  / IMG_SIZE
                sy = img_info["height"] / IMG_SIZE

                for row in det.tolist():
                    x1, y1, x2, y2, score, cls_idx = row
                    coco_results.append({
                        "image_id":    img_id,
                        "category_id": IDX_TO_CAT[int(cls_idx)],
                        "bbox":        [
                            x1 * sx,
                            y1 * sy,
                            (x2 - x1) * sx,
                            (y2 - y1) * sy,
                        ],
                        "score": score,
                    })

    mean_iou = iou_sum / iou_count if iou_count > 0 else 0.0

    if not coco_results:
        print("Warning: no detections above conf_thres=0.001 — mAP is 0.")
        model.train()
        return 0.0, 0.0, mean_iou

    if coco_gt is None:
        model.train()
        return 0.0, 0.0, mean_iou

    coco_dt   = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map50_95 = float(coco_eval.stats[0])   # IoU=0.50:0.95
    map50    = float(coco_eval.stats[1])   # IoU=0.50

    model.train()   # restore training mode — eval() is never undone otherwise
    return map50, map50_95, mean_iou


# =============================================================================
# 4.  MAIN
# =============================================================================

def main(args: argparse.Namespace) -> None:
    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # Let cuDNN auto-tune convolution algorithms for this fixed input shape.
    # All convolutions in the Detect head receive (B, 768, 37, 37) every batch,
    # so the one-time benchmark pays for itself within the first few batches.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DinoPrunableDetector(device=device).to(device)
    # head.stride is not a buffer in ultralytics Detect — move manually
    model.head.stride = model.head.stride.to(device)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    # Load checkpoint weights early so model params are correct before optimizer init
    _resume_ckpt = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        _resume_ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(_resume_ckpt["model_state_dict"])
        print(f"Resumed model weights from: {args.checkpoint}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = get_coco_loaders_v2(
        args.datadir, args.batch_size, args.num_workers
    )
    print(
        f"Train: {len(train_loader.dataset)} images | "
        f"Val: {len(val_loader.dataset)} images"
    )

    # ── Optimiser: only Detect head — backbone is frozen ─────────────────────
    optimizer = torch.optim.AdamW(
        list(model.head.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    # Linear warmup for warmup_epochs, then cosine decay to eta_min=1e-5
    warmup_epochs = args.warmup_epochs
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs - warmup_epochs),
        eta_min=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_epochs],
    )

    criterion = build_criterion(model.head, device)

    # AMP GradScaler — enabled only on CUDA (not MPS or CPU).
    # The scaler is passed into train_one_epoch; eval runs under autocast without
    # a scaler because no backward pass occurs there.
    use_amp = (device.type == "cuda")
    scaler  = amp.GradScaler(enabled=use_amp)

    # ── Training state ────────────────────────────────────────────────────────
    train_losses: list[float] = []
    map50_list:   list[float] = []

    burn_in    = max(warmup_epochs, 15)
    best_map50 = 0.0
    no_improve = 0
    start_epoch = 0

    # Restore optimizer / scheduler / early-stopping state from checkpoint
    if _resume_ckpt is not None:
        if "optimizer_state" in _resume_ckpt:
            optimizer.load_state_dict(_resume_ckpt["optimizer_state"])
        if "scheduler_state" in _resume_ckpt:
            scheduler.load_state_dict(_resume_ckpt["scheduler_state"])
        if "scaler_state" in _resume_ckpt and use_amp:
            scaler.load_state_dict(_resume_ckpt["scaler_state"])
        best_map50  = _resume_ckpt.get("best_map50",  0.0)
        no_improve  = _resume_ckpt.get("no_improve",  0)
        start_epoch = _resume_ckpt.get("epoch",       0)
        print(f"Resumed training state: epoch={start_epoch}, best_mAP={best_map50:.4f}")

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        t_loss, box_l, cls_l, dfl_l = train_one_epoch(
            model, criterion, train_loader, optimizer, device, scaler=scaler
        )

        # Evaluate every 5 epochs during the burn-in period (mAP is ~0 anyway
        # and a full 5 k-image val pass at 518×518 is slow).  After burn-in,
        # evaluate every epoch so early stopping can react promptly.
        if epoch >= burn_in or epoch % 5 == 4:
            map50, map50_95, mean_iou = evaluate_v2(
                model, val_loader, device, datadir=args.datadir
            )
            val_loss = _compute_val_loss(
                model, val_loader, criterion, device, n_batches=50
            )
        else:
            map50, map50_95, mean_iou, val_loss = 0.0, 0.0, 0.0, 0.0

        scheduler.step()

        train_losses.append(t_loss)
        map50_list.append(map50)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:02d}/{args.epochs} | "
            f"LR {current_lr:.2e} | "
            f"Train {t_loss:.3f} (box {box_l:.3f} cls {cls_l:.3f} dfl {dfl_l:.3f}) | "
            f"Val loss {val_loss:.3f} | "
            f"mAP@0.5 {map50:.4f}  mAP@0.5:0.95 {map50_95:.4f} | "
            f"MeanIoU {mean_iou:.4f}"
        )

        # Save latest checkpoint — includes optimizer/scheduler so resume works
        ckpt = {
            "epoch":              epoch + 1,
            "model_state_dict":   model.state_dict(),
            "optimizer_state":    optimizer.state_dict(),
            "scheduler_state":    scheduler.state_dict(),
            "scaler_state":       scaler.state_dict(),
            "best_map50":         best_map50,
            "no_improve":         no_improve,
            "map50":              map50,
            "map50_95":           map50_95,
            "val_loss":           val_loss,
            "mean_iou":           mean_iou,
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, "dinov2_coco_latest.pth"))

        # Save best checkpoint and update patience counter
        if map50 > best_map50 + 1e-4:
            best_map50 = map50
            no_improve  = 0
            ckpt["best_map50"] = best_map50
            torch.save(ckpt, os.path.join(args.checkpoint_dir, "dinov2_coco_best.pth"))
            print(f"  Best mAP@0.5: {best_map50:.4f} saved.")
        elif epoch >= burn_in:
            # Only increment patience counter once the burn-in period has ended
            no_improve += 1

        if no_improve >= args.patience:
            print(f"Early stop: no improvement for {args.patience} epochs.")
            break

    # ── Training curves ───────────────────────────────────────────────────────
    os.makedirs("figure", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, len(train_losses) + 1)

    ax1.plot(ep, train_losses)
    ax1.set_title("Total Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2.plot(ep, map50_list, color="green")
    ax2.set_title("mAP@0.5")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mAP")

    plt.tight_layout()
    plt.savefig("figure/trainval_curve_dinov2.jpg")
    print("Curves saved → figure/trainval_curve_dinov2.jpg")


# =============================================================================
# 5.  CLI
# =============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="AttentionSurgeon — DINOv2 ViT-B/14-reg + YOLOv8 "
                    "single-scale detection on COCO (518×518)"
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    p.add_argument(
        "--datadir",
        default=os.path.join(os.environ.get("BLACKHOLE", "."), "COCO"),
        help="COCO root directory (must contain train2017/, val2017/, annotations/).",
    )
    p.add_argument(
        "--checkpoint_dir",
        default="checkpoints/",
        help="Directory to save checkpoint .pth files.",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a checkpoint to resume training from.",
    )

    # ── Hyperparameters ───────────────────────────────────────────────────────
    p.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Maximum number of training epochs.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=16,
        # 518×518 raw images use considerably more VRAM than 224×224 cached
        # features, so default is lower than the v8 baseline.
        help="Images per batch (default 16 — lower than v8 due to 518px VRAM cost).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Peak learning rate for AdamW (after linear warmup).",
    )
    p.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Number of linear LR warmup epochs (lr/10 → lr).",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early-stopping patience (epochs without mAP improvement). "
             "Patience counter only starts after the burn-in period (max(warmup, 15)).",
    )

    # ── Hardware ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader worker processes.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Force compute device (e.g. cuda, mps, cpu). Auto-detected if omitted.",
    )

    main(p.parse_args())
