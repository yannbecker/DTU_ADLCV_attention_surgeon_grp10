# =============================================================================
# object_detection_v8.py — AttentionSurgeon: anchor-free detection (YOLOv8)
# =============================================================================
# References:
#   [1] Jocher et al., Ultralytics YOLOv8, 2023  — Detect head, v8DetectionLoss,
#       TaskAlignedAssigner, C2f neck block
#   [2] Oquab et al., DINOv2, TMLR 2024          — frozen ViT-B/14 backbone
#   [3] Li et al., ViTDet, ECCV 2022             — simple feature pyramid on ViT
#
# Architecture
# ────────────
#   DINOv2 ViT-B/14  (frozen, 12 layers × 12 heads, patch 14px)
#        │
#        │  patch tokens → (B, 768, 16, 16)
#        ▼
#   FPN Neck  (trainable)                         [ViTDet-style top-down]
#     lateral  Conv(768→256, 1×1)  → P3  (B, 256, 16×16)   stride 14
#     upsample(×2) + C2f(256→128) → P2  (B, 128, 32×32)   stride  7
#     upsample(×2) + C2f(128→64)  → P1  (B,  64, 64×64)   stride  3.5
#        ▼
#   ultralytics Detect(nc=80, ch=(64,128,256))    [anchor-free, DFL regression]
#        ▼
#   v8DetectionLoss  (TaskAlignedAssigner + CIoU + DFL + BCE)
#
# Pipeline (same two steps as object_detection_essai.py):
#   Step 1 — precompute DINOv2 features once
#       python object_detection_v8.py --mode precompute \
#           --datadir  $BLACKHOLE/COCO  --feat_dir $BLACKHOLE/COCO_features
#   Step 2 — train neck + head on cached features
#       python object_detection_v8.py --mode train \
#           --feat_dir $BLACKHOLE/COCO_features  --checkpoint_dir $BLACKHOLE/checkpoints
# =============================================================================

import os
import types
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# ── Ultralytics anchor-free components ───────────────────────────────────────
try:
    from ultralytics.nn.modules.head import Detect
    from ultralytics.nn.modules.block import C2f
    from ultralytics.utils.loss import v8DetectionLoss
    ULTRALYTICS_OK = True
except ImportError:
    ULTRALYTICS_OK = False
    raise ImportError(
        "ultralytics is required: pip install ultralytics\n"
        "On the HPC: pip install ultralytics --user"
    )

# NMS via torchvision — no ultralytics dependency for inference
from torchvision.ops import batched_nms

# ── Project imports ───────────────────────────────────────────────────────────
from classification import DinoClassifier
from object_detection_essai import (
    # Data pipeline — identical to the YOLOv3 version, no need to rewrite
    COCODetectionDataset, CachedFeaturesDataset,
    collate_fn, collate_fn_cached,
    get_coco_loaders, get_cached_loaders,
    precompute_features,
    # Shared constants
    COCO_CAT_IDS, CAT_TO_IDX, IDX_TO_CAT,
    DINO_MEAN, DINO_STD,
    IMG_SIZE, GRID_SIZE, FEAT_DIM,
    NUM_CLASSES, NUM_LAYERS, NUM_HEADS, HEAD_DIM,
)

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    COCO_EVAL_AVAILABLE = True
except ImportError:
    COCO_EVAL_AVAILABLE = False

# =============================================================================
# 1.  CONSTANTS
# =============================================================================

# FPN output channels (small, medium, large stride)
NECK_CHANNELS = (64, 128, 256)

# Strides for the three FPN levels on a 224×224 input.
# DINOv2 ViT-B/14 produces 16×16 patches → each patch covers 14px → stride 14.
# Two ×2 upsamples give stride 7 and 3.5.
STRIDES = torch.tensor([3.5, 7.0, 14.0])


# =============================================================================
# 2.  FPN NECK
# =============================================================================

class FPNNeck(nn.Module):
    """
    Top-down feature pyramid on a single DINOv2 feature map.

    DINOv2 outputs one flat spatial feature map (16×16). We upsample twice to
    create three scales that feed the anchor-free Detect head, following the
    "simple feature pyramid" design from ViTDet (Li et al., ECCV 2022).

    Input:   (B, 768, 16, 16)
    Outputs: [(B,  64, 64, 64),   ← P1, stride 3.5  — small objects
              (B, 128, 32, 32),   ← P2, stride 7    — medium objects
              (B, 256, 16, 16)]   ← P3, stride 14   — large objects
    """

    def __init__(self, in_ch: int = FEAT_DIM, out_chs: tuple = NECK_CHANNELS):
        super().__init__()
        c_small, c_med, c_large = out_chs  # 64, 128, 256

        # Reduce backbone channels to c_large
        self.lateral = nn.Sequential(
            nn.Conv2d(in_ch, c_large, 1, bias=False),
            nn.BatchNorm2d(c_large),
            nn.SiLU(inplace=True),
        )

        # 16×16 → 32×32:  C2f preserves spatial size, reduces channels
        self.p2 = C2f(c_large, c_med, n=1, shortcut=False)

        # 32×32 → 64×64
        self.p1 = C2f(c_med, c_small, n=1, shortcut=False)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        p3 = self.lateral(x)                                          # (B, 256, 16, 16)
        p2 = self.p2(F.interpolate(p3, scale_factor=2, mode="nearest"))  # (B, 128, 32, 32)
        p1 = self.p1(F.interpolate(p2, scale_factor=2, mode="nearest"))  # (B,  64, 64, 64)
        return [p1, p2, p3]   # ascending stride order — expected by Detect


# =============================================================================
# 3.  DinoDetectorV8
# =============================================================================

class DinoDetectorV8(DinoClassifier):
    """
    DINOv2 ViT-B/14 backbone (frozen) + FPN neck + YOLOv8 Detect head (trainable).

    Inherits from DinoClassifier (same as DinoDetector):
        set_mask(), register_pruning_hooks(), get_intra_layer_ranks() — unchanged.
        get_taylor_importance() — overridden to use v8 loss signal.
    """

    def __init__(self, device="cpu"):
        super().__init__(device=device, num_classes=1)
        del self.classifier

        self.neck = FPNNeck(in_ch=FEAT_DIM, out_chs=NECK_CHANNELS)
        self.head = Detect(nc=NUM_CLASSES, ch=NECK_CHANNELS)

        # Set strides — required by v8DetectionLoss and inference decoding.
        # Not a buffer in ultralytics Detect, so we move it to device manually
        # in main() after model.to(device).
        self.head.stride = STRIDES.clone()

    # ── Backbone feature extraction (precompute step only) ───────────────────

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """DINOv2 backbone → (B, 768, 16, 16). Called once during precompute."""
        x = self.transformer.prepare_tokens_with_masks(images)
        for blk in self.transformer.blocks:
            x = blk(x)
        x = self.transformer.norm(x)
        patches = x[:, 5:, :]                                      # (B, 256, 768)
        B = patches.size(0)
        return patches.permute(0, 2, 1).reshape(B, FEAT_DIM, GRID_SIZE, GRID_SIZE)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        Accepts raw images (B, 3, 224, 224) or cached features (B, 768, 16, 16).

        Training:  returns dict {"boxes", "scores", "feats"} from Detect head.
        Inference: returns (decoded_preds, raw_dict) from Detect head.
        """
        feat = self.extract_features(x) if x.shape[1] == 3 else x
        fpn  = self.neck(feat)       # [p1, p2, p3]
        return self.head(fpn)

    # ── Taylor importance for the RL agent ───────────────────────────────────

    def get_taylor_importance(
        self,
        images: torch.Tensor,
        targets: list[dict],
    ) -> torch.Tensor:
        """
        Gradient-based head importance via the YOLOv8 loss signal.
        Pass raw images (not cached features) so gradients flow through the backbone.
        """
        self.train()
        taylor_scores = torch.zeros(NUM_LAYERS, NUM_HEADS, device=images.device)
        activations, grads = {}, {}

        def _save(name):
            def hook(module, input, output):
                output.requires_grad_(True)
                activations[name] = output.detach()
                output.register_hook(lambda g: grads.update({name: g}))
            return hook

        temp_hooks = [
            self.transformer.blocks[i].attn.register_forward_hook(_save(f"layer{i}"))
            for i in range(NUM_LAYERS)
        ]

        with torch.set_grad_enabled(True):
            preds    = self.forward(images)
            batch    = _targets_to_batch(targets, images.device)
            criterion = _build_criterion(self.head, images.device)
            loss, _  = criterion(preds, batch)
            self.zero_grad()
            loss.backward()

        for i in range(NUM_LAYERS):
            key = f"layer{i}"
            if key in grads:
                act  = activations[key].view(*activations[key].shape[:2], NUM_HEADS, HEAD_DIM)
                grad = grads[key].view(*grads[key].shape[:2], NUM_HEADS, HEAD_DIM)
                taylor_scores[i] = (act * grad).abs().sum(-1).mean((0, 1))

        for h in temp_hooks:
            h.remove()
        return taylor_scores


# =============================================================================
# 4.  LOSS HELPERS
# =============================================================================

def _build_criterion(detect_head: Detect, device: torch.device) -> v8DetectionLoss:
    """
    Instantiate v8DetectionLoss without a full ultralytics model.

    v8DetectionLoss.__init__ needs:
        next(model.parameters()).device   → device
        model.args.{box, cls, dfl}        → loss hyperparameters
        model.model[-1].{stride, nc, reg_max} → from Detect head
        getattr(model, "class_weights")   → optional, set to None
    """
    _dummy = nn.Parameter(torch.empty(0, device=device))
    adapter = types.SimpleNamespace(
        args=types.SimpleNamespace(box=7.5, cls=0.5, dfl=1.5),
        model=[detect_head],        # model[-1] == detect_head
        class_weights=None,
    )
    adapter.parameters = lambda: iter([_dummy])
    return v8DetectionLoss(adapter)


def _targets_to_batch(targets: list[dict], device: torch.device) -> dict:
    """
    Convert list-of-dicts targets to ultralytics batch dict.

    Input  boxes: normalized xyxy [0, 1]
    Output bboxes: normalized xywh [0, 1]  (ultralytics convention)
    """
    batch_idx_list, cls_list, bboxes_list = [], [], []

    for i, tgt in enumerate(targets):
        boxes  = tgt["boxes"].to(device)    # (N, 4) normalized xyxy
        labels = tgt["labels"].to(device)   # (N,) long
        n = labels.numel()
        if n == 0:
            continue
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        w  =  boxes[:, 2] - boxes[:, 0]
        h  =  boxes[:, 3] - boxes[:, 1]
        batch_idx_list.append(torch.full((n,), i, dtype=torch.float32, device=device))
        cls_list.append(labels.float())
        bboxes_list.append(torch.stack([cx, cy, w, h], dim=1))

    if not batch_idx_list:
        return {
            "batch_idx": torch.zeros(0, device=device),
            "cls":       torch.zeros(0, device=device),
            "bboxes":    torch.zeros((0, 4), device=device),
        }
    return {
        "batch_idx": torch.cat(batch_idx_list),
        "cls":       torch.cat(cls_list),
        "bboxes":    torch.cat(bboxes_list),
    }


# =============================================================================
# 5.  TRAINING LOOP
# =============================================================================

def train_one_epoch(
    model:     DinoDetectorV8,
    criterion: v8DetectionLoss,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
) -> tuple[float, float, float, float]:
    """
    One training epoch.
    Returns (total_loss, box_loss, cls_loss, dfl_loss) averages.
    """
    model.train()
    totals = torch.zeros(4)   # [total, box, cls, dfl]

    for batch_data, targets in tqdm(loader, desc="Train", leave=False):
        batch_data = batch_data.to(device)
        preds      = model(batch_data)          # training mode → dict

        batch = _targets_to_batch(targets, device)
        loss, items = criterion(preds, batch)   # items: [box, cls, dfl]

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(model.neck.parameters()) + list(model.head.parameters()),
            max_norm=10.0,
        )
        optimizer.step()

        totals[0] += loss.item()
        totals[1:] += items.detach().cpu()

    n = len(loader)
    return (totals[0]/n).item(), (totals[1]/n).item(), (totals[2]/n).item(), (totals[3]/n).item()


# =============================================================================
# 6.  DECODE + NMS  (no ultralytics ops dependency)
# =============================================================================

def _decode_and_nms(
    raw_out,
    conf_thres: float = 0.01,
    iou_thres:  float = 0.45,
    max_det:    int   = 300,
) -> list[torch.Tensor]:
    """
    Decode Detect eval output and apply per-class NMS.

    raw_out: (y, preds_dict) tuple from Detect in eval mode, or just y.
        y: (B, 4+nc, total_anchors)
           channels  0:4  — xyxy boxes in input-image pixel space (224px)
           channels  4:   — class probabilities (already sigmoid-ed by Detect)

    Returns: list of (N, 6) float tensors  [x1, y1, x2, y2, score, cls_idx]
             one tensor per image in the batch, N = kept detections.
    """
    pred = raw_out[0] if isinstance(raw_out, (tuple, list)) else raw_out

    results = []
    for p in pred:                          # iterate over batch dimension
        boxes   = p[:4].T                  # (total_anchors, 4) xyxy
        probs   = p[4:]                    # (nc, total_anchors) — already sigmoid
        scores, cls_ids = probs.max(dim=0) # (total_anchors,) each

        mask = scores > conf_thres
        if not mask.any():
            results.append(torch.zeros((0, 6), device=pred.device))
            continue

        b, s, c = boxes[mask], scores[mask], cls_ids[mask]
        keep    = batched_nms(b, s, c, iou_thres)[:max_det]
        results.append(torch.cat([b[keep], s[keep, None], c[keep, None].float()], dim=1))

    return results


# =============================================================================
# 7.  EVALUATION
# =============================================================================

def evaluate(
    model:   DinoDetectorV8,
    loader:  DataLoader,
    device:  torch.device,
    datadir: str | None = None,
) -> tuple[float, float]:
    """
    COCOeval mAP.  Returns (mAP@0.5, mAP@0.5:0.95).
    Decoding and NMS use torchvision.ops.batched_nms — no ultralytics ops needed.
    """
    if not COCO_EVAL_AVAILABLE:
        print("pycocotools not found — mAP skipped.")
        return 0.0, 0.0

    model.eval()
    coco_results = []

    coco_gt = getattr(loader.dataset, "coco", None)
    if coco_gt is None and datadir:
        ann_file = os.path.join(datadir, "annotations", "instances_val2017.json")
        if os.path.exists(ann_file):
            coco_gt = COCO(ann_file)
        else:
            print(f"Annotation file not found at {ann_file} — mAP skipped.")

    with torch.no_grad():
        for batch_data, targets in tqdm(loader, desc="Eval", leave=False):
            batch_data = batch_data.to(device)
            out = model(batch_data)    # eval mode: (decoded_preds, raw_dict)

            # out: (decoded_preds, raw_dict) from Detect eval mode
            # decoded_preds: (B, 4+nc, total_anchors) in 224px pixel space
            dets = _decode_and_nms(out, conf_thres=0.01, iou_thres=0.45, max_det=300)
            # dets: list of (N, 6) tensors  [x1, y1, x2, y2, score, cls_idx]  (224px)

            if coco_gt is None:
                continue

            for det, tgt in zip(dets, targets):
                img_id   = tgt["image_id"]
                img_info = coco_gt.imgs[img_id]
                sx = img_info["width"]  / IMG_SIZE   # 224px → original image space
                sy = img_info["height"] / IMG_SIZE

                for row in det.tolist():
                    x1, y1, x2, y2, score, cls_idx = row
                    coco_results.append({
                        "image_id":    img_id,
                        "category_id": IDX_TO_CAT[int(cls_idx)],
                        "bbox":  [x1*sx, y1*sy, (x2-x1)*sx, (y2-y1)*sy],
                        "score": score,
                    })

    map50, map50_95 = 0.0, 0.0
    if coco_gt is not None and coco_results:
        coco_dt   = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        map50_95 = float(coco_eval.stats[0])
        map50    = float(coco_eval.stats[1])
    elif not coco_results:
        print("Warning: no detections above conf_thres=0.01 — mAP is 0.")

    return map50, map50_95


# =============================================================================
# 7.  MAIN
# =============================================================================

def main(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    model = DinoDetectorV8(device=device).to(device)
    model.head.stride = model.head.stride.to(device)   # not a buffer — move manually

    # ── Precompute mode ───────────────────────────────────────────────────────
    if args.mode == "precompute":
        assert args.datadir and args.feat_dir, "--datadir and --feat_dir required"
        precompute_features(
            model, args.datadir, args.feat_dir, device,
            args.batch_size, args.num_workers,
        )
        print("Done. Run --mode train --feat_dir ...")
        return

    # ── Data loaders ─────────────────────────────────────────────────────────
    if args.feat_dir and os.path.isdir(args.feat_dir):
        train_loader, val_loader = get_cached_loaders(
            args.feat_dir, args.batch_size, args.num_workers,
        )
    else:
        assert args.datadir, "Provide --feat_dir or --datadir."
        train_loader, val_loader = get_coco_loaders(
            args.datadir, args.batch_size, args.num_workers,
        )
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Resumed: {args.checkpoint}")

    # ── Optimise neck + head only (backbone is frozen) ───────────────────────
    trainable  = list(model.neck.parameters()) + list(model.head.parameters())
    optimizer  = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    criterion  = _build_criterion(model.head, device)

    train_losses, map50_list = [], []
    best_map50 = 0.0
    no_improve = 0

    for epoch in range(args.epochs):
        t_loss, box_l, cls_l, dfl_l = train_one_epoch(
            model, criterion, train_loader, optimizer, device
        )
        map50, map50_95 = evaluate(model, val_loader, device, datadir=args.datadir)
        scheduler.step()

        train_losses.append(t_loss)
        map50_list.append(map50)

        print(
            f"Epoch {epoch+1:02d}/{args.epochs} | "
            f"Loss {t_loss:.3f}  box {box_l:.3f}  cls {cls_l:.3f}  dfl {dfl_l:.3f} | "
            f"mAP@0.5 {map50:.4f}  mAP@0.5:0.95 {map50_95:.4f}"
        )

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "map50": map50,
            "map50_95": map50_95,
        }, os.path.join(args.checkpoint_dir, "dinov8_coco_latest.pth"))

        if map50 > best_map50 + 1e-4:
            best_map50 = map50
            no_improve  = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "map50": map50,
                "map50_95": map50_95,
            }, os.path.join(args.checkpoint_dir, "dinov8_coco_best.pth"))
            print(f"  ↑ Best mAP@0.5: {best_map50:.4f} saved.")
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"No improvement for {args.patience} epochs — early stop.")
            break

    # ── Training curves ───────────────────────────────────────────────────────
    os.makedirs("figure", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, len(train_losses) + 1)
    ax1.plot(ep, train_losses)
    ax1.set_title("Total Train Loss"); ax1.set_xlabel("Epoch")
    ax2.plot(ep, map50_list, color="green")
    ax2.set_title("mAP@0.5"); ax2.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig("figure/trainval_curve_coco_v8.jpg")
    print("Curves saved → figure/trainval_curve_coco_v8.jpg")


# =============================================================================
# 8.  CLI
# =============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="AttentionSurgeon — DINOv2 + YOLOv8 anchor-free detection on COCO"
    )
    p.add_argument("--mode", choices=["precompute", "train"], default="train")
    p.add_argument("--datadir",
        default=os.path.join(os.environ.get("BLACKHOLE", "."), "COCO"))
    p.add_argument("--feat_dir",
        default=os.path.join(os.environ.get("BLACKHOLE", "."), "COCO_features"))
    p.add_argument("--checkpoint_dir", default="checkpoints/")
    p.add_argument("--checkpoint",     default=None)
    p.add_argument("--epochs",         type=int,   default=40)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--patience",       type=int,   default=15)
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--device",         default=None)
    main(p.parse_args())
