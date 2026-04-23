# =============================================================================
# object_detection.py — AttentionSurgeon: Detection Task
# =============================================================================
# Architecture:
#   - DinoDetector subclasses DinoClassifier → inherits set_mask,
#     register_pruning_hooks, get_taylor_importance, get_intra_layer_ranks,
#     set_heads_requires_grad for free.
#   - Adds a 3-conv YOLO detection head on top of DINOv2 patch features.
#   - DetectionValidator mirrors FastProxyValidator from rl_utils.py but
#     uses patch tokens (x[:,5:]) and YOLO loss instead of CLS + CrossEntropy.
#
# TWO-STEP PIPELINE (recommended):
#
#   Step 1 — Precompute DINOv2 features ONCE (run only once, ~1-2h):
#       python object_detection.py --mode precompute \
#           --datadir    $BLACKHOLE/COCO \
#           --feat_dir   $BLACKHOLE/COCO_features
#
#   Step 2 — Train det_head on cached features (~10x faster):
#       python object_detection.py --mode train \
#           --feat_dir       $BLACKHOLE/COCO_features \
#           --checkpoint_dir $BLACKHOLE/checkpoints
#
# ONE-STEP PIPELINE (slower, no precompute):
#       python object_detection.py --mode train \
#           --datadir        $BLACKHOLE/COCO \
#           --checkpoint_dir $BLACKHOLE/checkpoints
#
# Usage (RL — in agent.py):
#   from object_detection import DinoDetector, CachedFeaturesDataset, \
#                                collate_fn_cached, DetectionValidator
#   model     = DinoDetector(device=device)
#   model.load_state_dict(torch.load("checkpoints/dino_coco_best.pth")["model_state_dict"])
#   proxy_ds  = CachedFeaturesDataset(feat_dir, split="val", max_samples=500)
#   proxy_loader = DataLoader(proxy_ds, batch_size=32, collate_fn=collate_fn_cached)
#   validator = DetectionValidator(model, proxy_loader, device)
# =============================================================================

import os
import math
import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision.ops import nms
from tqdm.auto import tqdm
from PIL import Image

from classification import DinoClassifier
from rl_utils import get_proxy_loader

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    COCO_EVAL_AVAILABLE = True
except ImportError:
    COCO_EVAL_AVAILABLE = False
    print("Warning: pycocotools not found. Full mAP evaluation disabled.")

# =============================================================================
# 1.  CONSTANTS
# =============================================================================

NUM_LAYERS  = 12
NUM_HEADS   = 12
HEAD_DIM    = 64
FEAT_DIM    = 768
GRID_SIZE   = 16
IMG_SIZE    = 224
NUM_CLASSES = 80
NUM_ANCHORS = 3

ANCHOR_PRIORS = [[0.28, 0.22],
                 [0.38, 0.48],
                 [0.90, 0.78]]

COCO_CAT_IDS = [
     1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]
CAT_TO_IDX = {cat_id: idx for idx, cat_id in enumerate(COCO_CAT_IDS)}
IDX_TO_CAT = {idx: cat_id for cat_id, idx in CAT_TO_IDX.items()}

DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD  = [0.229, 0.224, 0.225]

# =============================================================================
# 2.  MODEL — DinoDetector
# =============================================================================

class DinoDetector(DinoClassifier):
    """
    DINOv2 ViT-B/14 backbone (frozen) + YOLO detection head (trainable).

    Inherits from DinoClassifier:
        - load_model()                  via __init__
        - register_pruning_hooks()      → set_mask() works identically
        - set_mask(mask_1d)             → called by RL agent, unchanged
        - set_heads_requires_grad()     → used by get_taylor_importance
        - get_taylor_importance()       → overridden below (different loss)
        - get_intra_layer_ranks()       → inherited, unchanged
    """

    def __init__(self, device: str | torch.device = "cpu"):
        super().__init__(device=device, num_classes=1)
        del self.classifier

        out_ch = NUM_ANCHORS * (5 + NUM_CLASSES)   # 255
    
        self.det_head = nn.Sequential(
            nn.Conv2d(FEAT_DIM, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(p=0.1),                  # dropout
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(p=0.1),                  # dropout again
            nn.Conv2d(256, out_ch, kernel_size=1),
        )

        self.register_buffer(
            "anchors",
            torch.tensor(ANCHOR_PRIORS, dtype=torch.float32)
        )

    # ── Feature map from raw images (used during precompute only) ────────────

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Run DINOv2 backbone on raw images and return spatial patch feature map.
        Called ONCE during precompute — never during training epochs.

        Token layout for dinov2_vitb14_reg:
            index 0   : CLS token
            index 1–4 : 4 register tokens  (discarded)
            index 5–260: 256 patch tokens  ← spatial features

        Returns: (B, 768, 16, 16)
        """
        x = self.transformer.prepare_tokens_with_masks(images)
        for blk in self.transformer.blocks:
            x = blk(x)
        x = self.transformer.norm(x)

        patches  = x[:, 5:, :]                              # (B, 256, 768)
        B        = patches.size(0)
        feat_map = patches.permute(0, 2, 1)                 # (B, 768, 256)
        feat_map = feat_map.reshape(B, FEAT_DIM, GRID_SIZE, GRID_SIZE)
        return feat_map                                      # (B, 768, 16, 16)

    # ── Forward: accepts EITHER raw images OR pre-cached feature maps ─────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Accepts two input formats:
            - Raw images      (B, 3,   224, 224) → runs backbone + head
            - Cached features (B, 768,  16,  16) → runs head only (fast path)

        Returns: (B, NUM_ANCHORS, GRID_SIZE, GRID_SIZE, 5 + NUM_CLASSES)
                 = (B, 3, 16, 16, 85)
        """
        if x.shape[1] == 3:
            # Raw image path — backbone + head
            feat = self.extract_features(x)
        else:
            # Cached feature map path — head only (10x faster)
            feat = x

        raw = self.det_head(feat)                           # (B, 255, 16, 16)
        B, _, H, W = raw.shape
        return raw.reshape(B, NUM_ANCHORS, H, W, 5 + NUM_CLASSES)

    # ── Taylor importance ─────────────────────────────────────────────────────

    def get_taylor_importance(
        self,
        images: torch.Tensor,
        targets: list[dict],
    ) -> torch.Tensor:
        """
        Computes per-head Taylor importance using the YOLO loss signal.
        NOTE: pass raw images here (not cached features) — gradients must
        flow through the backbone blocks to compute head importance.

        Returns: (12, 12) Taylor score matrix.
        """
        self.eval()
        taylor_scores = torch.zeros(NUM_LAYERS, NUM_HEADS, device=images.device)
        activations, grads = {}, {}

        def save_activation(name):
            def hook(module, input, output):
                output.requires_grad_(True)
                activations[name] = output.detach()
                output.register_hook(lambda g: grads.update({name: g}))
            return hook

        temp_hooks = []
        for i in range(NUM_LAYERS):
            h = self.transformer.blocks[i].attn.register_forward_hook(
                save_activation(f"layer{i}")
            )
            temp_hooks.append(h)

        with torch.set_grad_enabled(True):
            preds = self.forward(images)
            loss  = yolo_loss(preds, targets, self.anchors, images.device)
            self.zero_grad()
            loss.backward()

        for i in range(NUM_LAYERS):
            key = f"layer{i}"
            if key in grads:
                act  = activations[key]
                grad = grads[key]
                act  = act.view(act.size(0),  act.size(1),  NUM_HEADS, HEAD_DIM)
                grad = grad.view(grad.size(0), grad.size(1), NUM_HEADS, HEAD_DIM)
                score = torch.abs(act * grad).sum(dim=-1).mean(dim=(0, 1))
                taylor_scores[i] = score

        for h in temp_hooks:
            h.remove()
        return taylor_scores


# =============================================================================
# 3.  YOLO LOSS
# =============================================================================

def yolo_loss(
    predictions:  torch.Tensor,
    targets:      list[dict],
    anchors:      torch.Tensor,
    device:       torch.device,
    lambda_coord: float = 5.0,
    lambda_cls:   float = 1.0,
    lambda_noobj: float = 0.5,    # ← YOLOv3 paper §2.2
    ignore_thresh: float = 0.5,   # ← YOLOv3 paper §2.2
) -> torch.Tensor:

    B, A, H, W, _ = predictions.shape
    C = NUM_CLASSES

    tgt_obj  = torch.zeros(B, A, H, W,    device=device)
    tgt_xy   = torch.zeros(B, A, H, W, 2, device=device)
    tgt_wh   = torch.zeros(B, A, H, W, 2, device=device)
    tgt_cls  = torch.zeros(B, A, H, W, C, device=device)
    obj_mask = torch.zeros(B, A, H, W,    device=device, dtype=torch.bool)

    for b_idx, tgt in enumerate(targets):
        boxes  = tgt["boxes"].to(device)
        labels = tgt["labels"].to(device)
        if boxes.numel() == 0:
            continue

        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        bw =  boxes[:, 2] - boxes[:, 0]
        bh =  boxes[:, 3] - boxes[:, 1]
        gi = (cx * W).long().clamp(0, W - 1)
        gj = (cy * H).long().clamp(0, H - 1)

        wh_gt      = torch.stack([bw, bh], dim=1).unsqueeze(1)
        wh_an      = anchors.unsqueeze(0)
        inter      = torch.min(wh_gt, wh_an).prod(-1)
        union      = wh_gt.prod(-1) + wh_an.prod(-1) - inter
        best_anchor = (inter / union.clamp(min=1e-6)).argmax(dim=1)

        for n in range(len(boxes)):
            a, gi_, gj_ = best_anchor[n].item(), gi[n].item(), gj[n].item()
            tgt_obj [b_idx, a, gj_, gi_]    = 1.0
            obj_mask[b_idx, a, gj_, gi_]    = True
            tgt_xy  [b_idx, a, gj_, gi_]    = torch.stack([
                cx[n] * W - gi[n].float(),
                cy[n] * H - gj[n].float()])
            tgt_wh  [b_idx, a, gj_, gi_]    = torch.stack([
                torch.log(bw[n] / anchors[a, 0].clamp(min=1e-6)),
                torch.log(bh[n] / anchors[a, 1].clamp(min=1e-6))])
            cls_idx = labels[n].item()
            if 0 <= cls_idx < C:
                tgt_cls[b_idx, a, gj_, gi_, cls_idx] = 1.0

    mse        = nn.MSELoss(reduction="sum")
    bce_elem   = nn.BCELoss(reduction="none")
    bce_logits = nn.BCEWithLogitsLoss()

    # ── Objectness: split positive / negative with ignore threshold ──────────
    pred_obj    = torch.sigmoid(predictions[..., 4])
    obj_loss_map = bce_elem(pred_obj, tgt_obj)          # (B, A, H, W)

    # Ignore background cells with high predicted confidence (YOLOv3 §2.2)
    with torch.no_grad():
        ignore_mask = (~obj_mask) & (pred_obj > ignore_thresh)

    loss_obj   = obj_loss_map[obj_mask].mean() if obj_mask.any() else torch.tensor(0., device=device)
    noobj_mask = ~obj_mask & ~ignore_mask
    loss_noobj = obj_loss_map[noobj_mask].mean() if noobj_mask.any() else torch.tensor(0., device=device)

    # ── Box regression + classification over positive cells only ────────────
    if obj_mask.any():
        pred_xy = torch.stack([
            torch.sigmoid(predictions[..., 0]),
            torch.sigmoid(predictions[..., 1]),
        ], dim=-1)
        pred_wh  = predictions[..., 2:4]
        loss_xy  = mse(pred_xy[obj_mask], tgt_xy[obj_mask]) / B
        loss_wh  = mse(pred_wh[obj_mask], tgt_wh[obj_mask]) / B
        loss_cls = bce_logits(predictions[..., 5:][obj_mask], tgt_cls[obj_mask])
    else:
        loss_xy  = torch.tensor(0., device=device)
        loss_wh  = torch.tensor(0., device=device)
        loss_cls = torch.tensor(0., device=device)

    return (lambda_coord * (loss_xy + loss_wh)
            + loss_obj
            + lambda_noobj * loss_noobj
            + lambda_cls * loss_cls)

# =============================================================================
# 4.  DECODE PREDICTIONS + NMS
# =============================================================================

def decode_predictions(
    raw_preds: torch.Tensor,
    anchors: torch.Tensor,
    conf_thresh: float = 0.25,
    nms_thresh:  float = 0.45,
) -> list[dict]:
    """
    Convert raw (B, 3, 16, 16, 85) output to per-image detection dicts.

    Returns list of dicts:
        'boxes'  : (K, 4) float xyxy pixel-space
        'scores' : (K,)   float  obj * class_score
        'labels' : (K,)   int    contiguous class index 0–79
    """
    B, A, H, W, _ = raw_preds.shape
    cell_w = IMG_SIZE / W
    cell_h = IMG_SIZE / H
    results = []

    for b in range(B):
        boxes_list, scores_list, labels_list = [], [], []

        for a in range(A):
            for gj in range(H):
                for gi in range(W):
                    p   = raw_preds[b, a, gj, gi]
                    obj = torch.sigmoid(p[4]).item()
                    if obj < conf_thresh:
                        continue

                    cls_probs         = torch.softmax(p[5:], dim=0)
                    cls_score, cls_label = cls_probs.max(0)
                    final_score       = obj * cls_score.item()
                    if final_score < conf_thresh:
                        continue

                    cx = (gi + torch.sigmoid(p[0]).item()) * cell_w
                    cy = (gj + torch.sigmoid(p[1]).item()) * cell_h
                    bw = anchors[a, 0].item() * IMG_SIZE * math.exp(
                        max(-10.0, min(10.0, p[2].item())))
                    bh = anchors[a, 1].item() * IMG_SIZE * math.exp(
                        max(-10.0, min(10.0, p[3].item())))

                    boxes_list.append([cx - bw/2, cy - bh/2,
                                       cx + bw/2, cy + bh/2])
                    scores_list.append(final_score)
                    labels_list.append(cls_label.item())

        if boxes_list:
            bx   = torch.tensor(boxes_list, dtype=torch.float32)
            sc   = torch.tensor(scores_list, dtype=torch.float32)
            lb   = torch.tensor(labels_list, dtype=torch.long)
            keep = nms(bx, sc, nms_thresh)
            results.append({"boxes": bx[keep], "scores": sc[keep],
                            "labels": lb[keep]})
        else:
            results.append({
                "boxes":  torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros(0,      dtype=torch.float32),
                "labels": torch.zeros(0,      dtype=torch.long),
            })

    return results


# =============================================================================
# 5.  RAW IMAGE DATASET — COCODetectionDataset
#     Used only during --mode precompute.  Not touched during training.
# =============================================================================

class COCODetectionDataset(Dataset):
    """
    COCO 2017 dataset that loads raw JPEG images + annotations.
    Only used during the precompute step to run DINOv2 once.

    Directory layout (produced by download_data.py):
        $BLACKHOLE/COCO/
            train2017/
            val2017/
            annotations/
                instances_train2017.json
                instances_val2017.json
    """

    _transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(DINO_MEAN, DINO_STD),
    ])

    def __init__(self, root: str, split: str = "val"):
        assert split in ("train", "val")
        self.img_dir  = os.path.join(root, f"{split}2017")
        ann_file      = os.path.join(root, "annotations",
                                     f"instances_{split}2017.json")
        if not COCO_EVAL_AVAILABLE:
            raise ImportError("pycocotools is required.")
        self.coco     = COCO(ann_file)
        self.img_ids  = sorted(self.coco.imgs.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id   = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        img      = Image.open(
            os.path.join(self.img_dir, img_info["file_name"])
        ).convert("RGB")

        orig_w, orig_h = img.size
        img_tensor     = self._transform(img)

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns    = self.coco.loadAnns(ann_ids)

        boxes, labels = [], []
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in CAT_TO_IDX:
                continue
            x, y, w, h = ann["bbox"]
            x1 = max(0., min(1., x / orig_w))
            y1 = max(0., min(1., y / orig_h))
            x2 = max(0., min(1., (x + w) / orig_w))
            y2 = max(0., min(1., (y + h) / orig_h))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(CAT_TO_IDX[cat_id])

        target = {
            "boxes":    torch.tensor(boxes,  dtype=torch.float32)
                        if boxes  else torch.zeros((0, 4), dtype=torch.float32),
            "labels":   torch.tensor(labels, dtype=torch.long)
                        if labels else torch.zeros(0, dtype=torch.long),
            "image_id": img_id,
        }
        return img_tensor, target


def collate_fn(batch):
    images  = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets


# =============================================================================
# 6.  PRECOMPUTE — Run DINOv2 once, save features to disk
# =============================================================================

def precompute_features(
    model:    DinoDetector,
    datadir:  str,
    feat_dir: str,
    device:   torch.device,
    batch_size:  int = 64,
    num_workers: int = 4,
) -> None:
    """
    Pass every COCO image through DINOv2 exactly once and save the resulting
    (768, 16, 16) feature map + target dict as a .pt file.

    Output layout:
        $feat_dir/
            train/
                <image_id>.pt   ← {"feat": Tensor(768,16,16), "target": dict}
            val/
                <image_id>.pt

    Disk usage: ~786 KB per image × 123k images ≈ ~93 GB total.
    Runtime   : ~1–2h on a single V100.
    """
    model.eval()

    for split in ("train", "val"):
        out_dir = os.path.join(feat_dir, split)
        os.makedirs(out_dir, exist_ok=True)

        dataset = COCODetectionDataset(datadir, split=split)
        loader  = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        print(f"\n[Precompute] {split}: {len(dataset)} images → {out_dir}")

        with torch.no_grad():
            for images, targets in tqdm(loader, desc=f"Precompute {split}"):
                images = images.to(device)
                feats  = model.extract_features(images)  # (B, 768, 16, 16)

                for i, tgt in enumerate(targets):
                    img_id   = tgt["image_id"]
                    save_path = os.path.join(out_dir, f"{img_id}.pt")
                    if not os.path.exists(save_path):   # resume-safe
                        torch.save(
                            {"feat": feats[i].cpu(), "target": tgt},
                            save_path,
                        )

        print(f"[Precompute] {split} done.")


# =============================================================================
# 7.  CACHED FEATURES DATASET — used during training (no backbone call)
# =============================================================================

class CachedFeaturesDataset(Dataset):
    """
    Loads pre-computed DINOv2 feature maps from disk.
    Replaces COCODetectionDataset during training — the backbone is never called.

    Each __getitem__ returns:
        feat   : (768, 16, 16) float tensor — DINOv2 patch features
        target : dict  {'boxes', 'labels', 'image_id'}
    """

    def __init__(self, feat_dir: str, split: str = "train",
                 max_samples: int | None = None):
        assert split in ("train", "val")
        self.split_dir = os.path.join(feat_dir, split)
        assert os.path.isdir(self.split_dir), (
            f"Feature directory not found: {self.split_dir}\n"
            f"Run --mode precompute first."
        )
        self.files = sorted(
            os.path.join(self.split_dir, f)
            for f in os.listdir(self.split_dir) if f.endswith(".pt")
        )
        if max_samples is not None:
            self.files = self.files[:max_samples]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data   = torch.load(self.files[idx], weights_only=True)
        feat   = data["feat"]        # (768, 16, 16)
        target = data["target"]
        return feat, target


def collate_fn_cached(batch):
    """Collate for CachedFeaturesDataset — same interface as collate_fn."""
    feats   = torch.stack([b[0] for b in batch])   # (B, 768, 16, 16)
    targets = [b[1] for b in batch]
    return feats, targets


def get_cached_loaders(
    feat_dir:    str,
    batch_size:  int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader) backed by cached feature files.
    Drop-in replacement for get_coco_loaders() — same return signature.
    """
    train_set = CachedFeaturesDataset(feat_dir, split="train")
    val_set   = CachedFeaturesDataset(feat_dir, split="val")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_fn_cached,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_fn_cached,
    )
    return train_loader, val_loader


def get_coco_loaders(
    datadir:     str,
    batch_size:  int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """
    Fallback: raw image loaders (slow path — backbone runs every epoch).
    Used when --feat_dir is not provided.
    """
    train_set = COCODetectionDataset(datadir, split="train")
    val_set   = COCODetectionDataset(datadir, split="val")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn,
    )
    return train_loader, val_loader


# =============================================================================
# 8.  TRAINING AND VALIDATION LOOPS
# =============================================================================

def train_one_epoch(
    model:     DinoDetector,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
) -> float:
    """
    One training epoch. Works with BOTH raw-image and cached-feature loaders:
        - Cached features (B, 768, 16, 16) → model.forward skips backbone
        - Raw images      (B, 3,  224, 224) → model.forward runs backbone

    Only det_head parameters receive gradients — backbone is always frozen.
    Returns: average YOLO loss over the epoch.
    """
    model.train()
    running_loss = 0.0

    for batch, targets in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        preds = model(batch)
        loss  = yolo_loss(preds, targets, model.anchors, device)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.det_head.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(
    model:  DinoDetector,
    loader: DataLoader,
    device: torch.device,
    datadir:  str | None = None,
) -> tuple[float, float, float]:
    """
    Full evaluation with official COCOeval mAP.
    Works with both raw-image and cached-feature loaders.

    Returns: (avg_loss, map50, map50_95)
    """
    if not COCO_EVAL_AVAILABLE:
        print("pycocotools not available — skipping mAP evaluation.")
        return 0.0, 0.0, 0.0

    model.eval()
    running_loss = 0.0
    coco_results = []

    # Resolve COCO ground truth — available on COCODetectionDataset only
    coco_gt = getattr(loader.dataset, "coco", None)
    if coco_gt is None:
        # CachedFeaturesDataset: rebuild COCO object from annotation file
        # (needed for COCOeval — annotations are not stored in .pt files)
        if datadir:
            # Explicit path — guaranteed correct regardless of feat_dir layout
            ann_file = os.path.join(datadir, "annotations", "instances_val2017.json")
        else:
            # Fallback: walk up from feat_dir/val/ → feat_dir/ → blackhole/COCO/
            ann_file = os.path.join(
                os.path.dirname(loader.dataset.split_dir),
                "..", "..", "COCO", "annotations", "instances_val2017.json"
            )
        ann_file = os.path.normpath(ann_file)
        if os.path.exists(ann_file):
            coco_gt = COCO(ann_file)
        else:
            print(f"Warning: annotation file not found at {ann_file}. "
                  f"mAP will be skipped.")
            coco_gt = None

    with torch.no_grad():
        for batch, targets in tqdm(loader, desc="Evaluating", leave=False):
            batch = batch.to(device)
            preds = model(batch)
            loss  = yolo_loss(preds, targets, model.anchors, device)
            running_loss += loss.item()

            if coco_gt is not None:
                detections = decode_predictions(preds, model.anchors)
                for det, tgt in zip(detections, targets):
                    img_id = tgt["image_id"]
                    for k in range(len(det["scores"])):
                        box   = det["boxes"][k]
                        score = det["scores"][k].item()
                        label = det["labels"][k].item()
                        coco_results.append({
                            "image_id":    img_id,
                            "category_id": IDX_TO_CAT[label],
                            "bbox": [box[0].item(), box[1].item(),
                                     (box[2]-box[0]).item(),
                                     (box[3]-box[1]).item()],
                            "score": score,
                        })

    avg_loss        = running_loss / len(loader)
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
        print("Warning: no detections above threshold — mAP is 0.")

    return avg_loss, map50, map50_95


# =============================================================================
# 9.  DETECTION VALIDATOR  (RL agent — mirrors FastProxyValidator)
# =============================================================================

class DetectionValidator:
    """
    Rapid detection evaluator for the RL environment.

    Mirrors FastProxyValidator from rl_utils.py:
        validator.evaluate(mask_1d) → (avg_loss, proxy_map)

    With CachedFeaturesDataset as proxy_loader, each evaluate() call:
        1. Replays transformer blocks (with current mask) on cached tokens
        2. Feeds features through det_head
        3. Returns (yolo_loss, mean_objectness) as the reward signal

    This is the same mechanism used in DetectionValidator._build_cache()
    but now the cache is built from pre-saved .pt files — no backbone needed.
    """

    def __init__(
        self,
        model:        DinoDetector,
        proxy_loader: DataLoader,
        device:       torch.device,
    ):
        self.model  = model
        self.device = device
        self.cached_tokens:  list[torch.Tensor] = []
        self.cached_targets: list[list[dict]]   = []
        self._build_cache(proxy_loader)

    def _build_cache(self, proxy_loader: DataLoader) -> None:
        """
        Pre-compute and store transformer tokens for all proxy images.

        Accepts two loader types:
            - CachedFeaturesDataset: features already computed → wrap back
              into token form by a dummy pass through just prepare_tokens.
              Actually: we store the (768,16,16) features directly and skip
              the token re-wrap entirely during evaluate().
            - COCODetectionDataset: run full backbone to get tokens.

        Internally we store (B, 768, 16, 16) feature tensors (not tokens)
        when the loader is backed by cached features, or full token sequences
        (B, N+5, 768) when backed by raw images.
        """
        self.model.eval()
        n = len(proxy_loader.dataset)
        print(f"Building Detection Proxy Cache for {n} samples...")

        is_cached = isinstance(proxy_loader.dataset, CachedFeaturesDataset)

        with torch.no_grad():
            for batch, targets in proxy_loader:
                batch = batch.to(self.device)
                if is_cached:
                    # batch is already (B, 768, 16, 16) — store directly
                    self.cached_tokens.append(batch)
                else:
                    # batch is (B, 3, 224, 224) — run backbone to get tokens
                    tokens = self.model.transformer.prepare_tokens_with_masks(batch)
                    self.cached_tokens.append(tokens)
                self.cached_targets.append(targets)

        self._is_feat_cache = is_cached
        print("Detection Proxy Cache built successfully.")

    def evaluate(self, mask_1d: torch.Tensor) -> tuple[float, float]:
        """
        Fast evaluation with current pruning mask.

        Args:
            mask_1d: (144,) float tensor — current head pruning mask

        Returns:
            (avg_loss, proxy_map)
            proxy_map = mean sigmoid objectness — fast mAP surrogate
        """
        self.model.eval()
        self.model.set_mask(mask_1d)

        running_loss     = 0.0
        total_objectness = 0.0
        num_batches      = len(self.cached_tokens)

        with torch.no_grad():
            for cached, targets in zip(self.cached_tokens, self.cached_targets):

                if self._is_feat_cache:
                    # Fast path: cached is (B, 768, 16, 16) — go straight to head
                    feat_map = cached
                else:
                    # Token path: replay transformer blocks with current mask
                    x = cached
                    for blk in self.model.transformer.blocks:
                        x = blk(x)
                    x = self.model.transformer.norm(x)
                    patches  = x[:, 5:, :]
                    B        = patches.size(0)
                    feat_map = patches.permute(0, 2, 1).reshape(
                        B, FEAT_DIM, GRID_SIZE, GRID_SIZE
                    )

                raw   = self.model.det_head(feat_map)
                B_    = feat_map.size(0)
                preds = raw.reshape(B_, NUM_ANCHORS, GRID_SIZE, GRID_SIZE,
                                    5 + NUM_CLASSES)

                loss = yolo_loss(preds, targets, self.model.anchors, self.device)
                running_loss     += loss.item()
                total_objectness += torch.sigmoid(preds[..., 4]).mean().item()

        avg_loss  = running_loss     / num_batches
        proxy_map = total_objectness / num_batches
        return avg_loss, proxy_map


# =============================================================================
# 10. MAIN
# =============================================================================

def main(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    model = DinoDetector(device=device).to(device)

    # ── MODE: precompute ──────────────────────────────────────────────────────
    if args.mode == "precompute":
        assert args.datadir,  "--datadir is required for --mode precompute"
        assert args.feat_dir, "--feat_dir is required for --mode precompute"
        precompute_features(
            model=model,
            datadir=args.datadir,
            feat_dir=args.feat_dir,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print("Precompute complete. Now run with --mode train --feat_dir ...")
        return

    # ── MODE: train ───────────────────────────────────────────────────────────
    # Choose cached loader (fast) or raw image loader (slow fallback)
    if args.feat_dir and os.path.isdir(args.feat_dir):
        print(f"Using cached features from: {args.feat_dir}")
        train_loader, val_loader = get_cached_loaders(
            feat_dir=args.feat_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        assert args.datadir, (
            "Provide either --feat_dir (fast) or --datadir (slow fallback)."
        )
        print(f"No feat_dir found — using raw images from: {args.datadir}")
        train_loader, val_loader = get_coco_loaders(
            datadir=args.datadir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    print(f"Train: {len(train_loader.dataset)} samples | "
          f"Val: {len(val_loader.dataset)} samples")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}")

    optimizer = optim.Adam(model.det_head.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    train_losses, val_losses, map50_list = [], [], []
    best_val_loss = float("inf")
    best_map50    = 0.0
    no_improve    = 0

    for epoch in range(args.epochs):
        t_loss              = train_one_epoch(model, train_loader, optimizer, device)
        v_loss, map50, map50_95 = evaluate(model, val_loader, device,
                                   datadir=args.datadir)
        scheduler.step()

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        map50_list.append(map50)

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | "
            f"mAP@0.5: {map50:.4f} | mAP@0.5:0.95: {map50_95:.4f}"
        )

        # Save latest checkpoint every epoch
        torch.save({
            "epoch":            epoch + 1,
            "model_state_dict": model.state_dict(),
            "map50":            map50,
            "map50_95":         map50_95,
        }, os.path.join(args.checkpoint_dir, "dino_coco_latest.pth"))

        # Early stopping on val loss (not mAP — mAP is 0 for first ~15 epochs)
        if v_loss < best_val_loss - 0.01:
            best_val_loss = v_loss
            no_improve    = 0
            if map50 > best_map50:
                best_map50 = map50
            torch.save({
                "epoch":            epoch + 1,
                "model_state_dict": model.state_dict(),
                "map50":            map50,
                "map50_95":         map50_95,
            }, os.path.join(args.checkpoint_dir, "dino_coco_best.pth"))
            print(f"  → Best val loss: {best_val_loss:.4f} | "
                  f"mAP@0.5: {best_map50:.4f} saved.")
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"Plateau for {args.patience} epochs — stopping early.")
            break

    # Training curves
    os.makedirs("figure", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(train_losses) + 1)
    ax1.plot(ep, train_losses, label="Train Loss")
    ax1.plot(ep, val_losses,   label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend()
    ax1.set_title("YOLO Loss")
    ax2.plot(ep, map50_list, label="mAP@0.5", color="green")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("mAP"); ax2.legend()
    ax2.set_title("Validation mAP@0.5")
    plt.tight_layout()
    plt.savefig("figure/trainval_curve_coco.jpg")
    print("Training curves saved to figure/trainval_curve_coco.jpg")


# =============================================================================
# 11. CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AttentionSurgeon — DINOv2 YOLO Detection on COCO"
    )
    parser.add_argument(
        "--mode", choices=["precompute", "train"], default="train",
        help="precompute: run DINOv2 once and save features. "
             "train: train det_head (uses cached features if --feat_dir given)."
    )
    # Paths
    parser.add_argument(
        "--datadir", type=str,
        default=os.path.join(os.environ.get("BLACKHOLE", "."), "COCO"),
        help="COCO root dir. Required for --mode precompute. "
             "Fallback for --mode train if --feat_dir not provided."
    )
    parser.add_argument(
        "--feat_dir", type=str,
        default=os.path.join(os.environ.get("BLACKHOLE", "."), "COCO_features"),
        help="Directory for pre-computed DINOv2 features "
             "(output of --mode precompute, input of --mode train)."
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints/"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to resume training from a saved checkpoint."
    )
    # Hyperparameters
    parser.add_argument("--epochs",     type=int,   default=40)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument(
        "--patience", type=int, default=15,
        help="Early stopping patience. Tracks val loss, not mAP "
             "(mAP stays 0 for the first ~15 epochs)."
    )
    # Hardware
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--device", type=str, default=None,
        help="cuda | mps | cpu  (auto-detected if omitted)"
    )

    args = parser.parse_args()
    main(args)
