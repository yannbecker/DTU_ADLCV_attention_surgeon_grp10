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
# Usage (training):
#   python object_detection.py \
#       --datadir $BLACKHOLE/COCO \
#       --epochs 40 \
#       --batch_size 32 \
#       --lr 1e-3 \
#       --checkpoint_dir checkpoints/
#
# Usage (RL — in agent.py, swap these two lines):
#   from object_detection import DinoDetector, get_coco_loaders, DetectionValidator
#   model   = DinoDetector(device=device)
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

# Reuse existing modules — no reimplementation needed
from classification import DinoClassifier
from rl_utils import get_proxy_loader

# pycocotools is needed only for mAP evaluation
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
HEAD_DIM    = 64          # 768 / 12
FEAT_DIM    = 768
GRID_SIZE   = 16          # 224 / 14 patches per side
IMG_SIZE    = 224
NUM_CLASSES = 80          # COCO
NUM_ANCHORS = 3

# Anchors normalised to [0,1] relative to full image (w, h).
# These approximate the k-means clusters from YOLOv2 on COCO (Redmon 2017).
ANCHOR_PRIORS = [[0.28, 0.22],   # small objects
                 [0.38, 0.48],   # medium objects
                 [0.90, 0.78]]   # large objects

# COCO category IDs are non-contiguous (1–90 with gaps).
# Map them to contiguous 0–79 indices for the network output.
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

# DINOv2 normalisation (ImageNet statistics)
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

    Detection-specific additions:
        - det_head  : 3-layer conv neck + prediction (YOLOv3-style)
        - _features : extracts patch tokens x[:,5:] as a (B,768,16,16) map
        - forward   : _features → det_head → (B,3,16,16,85)
    """

    def __init__(self, device: str | torch.device = "cpu"):
        # super().__init__ calls load_model(), freezes backbone,
        # initialises self.mask (12,12) and attaches pruning hooks.
        # num_classes=1 is a placeholder — self.classifier is deleted below.
        super().__init__(device=device, num_classes=1)

        # Remove the classification linear head — not needed for detection.
        del self.classifier

        # ── Detection head (YOLOv3-style, Redmon & Farhadi 2018) ─────────────
        # Input : (B, 768, 16, 16)  — DINOv2 patch feature map
        # Output: (B, 3*(5+80), 16, 16) = (B, 255, 16, 16)
        #
        # 768→512→256: progressive channel compression (FPN/ViTDet convention).
        # kernel=3 with padding=1: spatial feature mixing without size change.
        # kernel=1 final layer: per-location linear projection to predictions.
        # BatchNorm: YOLOv2 addition for training stability.
        # LeakyReLU(0.1): verbatim from YOLOv1 paper, kept through all versions.
        out_ch = NUM_ANCHORS * (5 + NUM_CLASSES)   # 3 * 85 = 255
        self.det_head = nn.Sequential(
            nn.Conv2d(FEAT_DIM, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, out_ch, kernel_size=1),
        )

        # Anchor priors as a non-trainable buffer (moves with .to(device))
        self.register_buffer(
            "anchors",
            torch.tensor(ANCHOR_PRIORS, dtype=torch.float32)
        )

    # ── Feature extraction ────────────────────────────────────────────────────

    def _features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Run DINOv2 and return the spatial patch feature map.

        Token layout for dinov2_vitb14_reg:
            index 0   : CLS token
            index 1–4 : 4 register tokens  (discarded — artefact prevention)
            index 5–260: 256 patch tokens  ← we use these

        Returns: (B, 768, 16, 16)
        """
        x = self.transformer.prepare_tokens_with_masks(images)
        for blk in self.transformer.blocks:
            x = blk(x)
        x = self.transformer.norm(x)

        # Drop CLS (0) and 4 register tokens (1–4) → 256 patch tokens
        patches = x[:, 5:, :]                          # (B, 256, 768)
        B = patches.size(0)
        # Reshape sequence back to 2-D spatial grid
        feat_map = patches.permute(0, 2, 1)            # (B, 768, 256)
        feat_map = feat_map.view(B, FEAT_DIM, GRID_SIZE, GRID_SIZE)
        return feat_map                                 # (B, 768, 16, 16)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass.

        Returns: (B, NUM_ANCHORS, GRID_SIZE, GRID_SIZE, 5+NUM_CLASSES)
                 = (B, 3, 16, 16, 85)

        Dimension 85:
            [0]    tx  — x offset within cell (apply sigmoid to decode)
            [1]    ty  — y offset within cell (apply sigmoid to decode)
            [2]    tw  — log-scale width  relative to anchor
            [3]    th  — log-scale height relative to anchor
            [4]    obj — objectness logit  (apply sigmoid to decode)
            [5:85] cls — class logits      (apply softmax to decode)
        """
        feat = self._features(images)                  # (B, 768, 16, 16)
        raw  = self.det_head(feat)                     # (B, 255, 16, 16)
        B, _, H, W = raw.shape
        # Reshape to (B, anchors, H, W, 85)
        return raw.reshape(B, NUM_ANCHORS, H, W, 5 + NUM_CLASSES)

    # ── Taylor importance — overrides DinoClassifier ──────────────────────────

    def get_taylor_importance(
        self,
        images: torch.Tensor,
        targets: list[dict],
    ) -> torch.Tensor:
        """
        Computes per-head Taylor importance using the YOLO loss signal.
        Identical logic to DinoClassifier.get_taylor_importance() but uses
        yolo_loss() instead of CrossEntropyLoss.

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
                act  = activations[key]               # (B, N, 768)
                grad = grads[key]                     # (B, N, 768)
                act  = act.view(act.size(0),  act.size(1),  NUM_HEADS, HEAD_DIM)
                grad = grad.view(grad.size(0), grad.size(1), NUM_HEADS, HEAD_DIM)
                score = torch.abs(act * grad).sum(dim=-1).mean(dim=(0, 1))
                taylor_scores[i] = score

        for h in temp_hooks:
            h.remove()
        return taylor_scores   # (12, 12)


# =============================================================================
# 3.  YOLO LOSS  (YOLOv3-style, Redmon & Farhadi 2018)
# =============================================================================

def yolo_loss(
    predictions: torch.Tensor,
    targets: list[dict],
    anchors: torch.Tensor,
    device: torch.device,
    lambda_coord: float = 5.0,
    lambda_cls: float   = 1.0,
) -> torch.Tensor:
    """
    Inspired by YOLOv3 loss over a batch.

    Args:
        predictions : (B, 3, 16, 16, 85) — raw network output
        targets     : list of dicts, one per image:
                        'boxes'  (N,4) xyxy normalised [0,1]
                        'labels' (N,)  contiguous class indices 0–79
        anchors     : (3, 2) anchor (w, h) normalised to full image
        device      : torch device
        lambda_coord: weight for box regression loss  (=5, from YOLOv1 paper)
        lambda_cls  : weight for classification loss  (=1, from YOLOv1 paper)

    Returns: scalar loss tensor (differentiable w.r.t. det_head parameters)

    Loss structure (all three from YOLOv1/v2/v3 papers):
        L = lambda_coord * (L_xy + L_wh)   box regression — MSE
          + lambda_coord * L_obj            objectness     — BCE  (YOLOv3)
          + lambda_cls   * L_cls            classification — BCE  (YOLOv3)
    """
    B, A, H, W, _ = predictions.shape
    C = NUM_CLASSES

    # ── Build target tensors ──────────────────────────────────────────────────
    tgt_obj  = torch.zeros(B, A, H, W,    device=device)
    tgt_xy   = torch.zeros(B, A, H, W, 2, device=device)
    tgt_wh   = torch.zeros(B, A, H, W, 2, device=device)
    tgt_cls  = torch.zeros(B, A, H, W, C, device=device)
    obj_mask = torch.zeros(B, A, H, W,    device=device, dtype=torch.bool)

    for b_idx, tgt in enumerate(targets):
        boxes  = tgt["boxes"].to(device)    # (N, 4) xyxy normalised
        labels = tgt["labels"].to(device)   # (N,)
        if boxes.numel() == 0:
            continue

        # Convert xyxy → cxcywh (all normalised)
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        bw =  boxes[:, 2] - boxes[:, 0]
        bh =  boxes[:, 3] - boxes[:, 1]

        # Grid cell that contains the object centre
        gi = (cx * W).long().clamp(0, W - 1)
        gj = (cy * H).long().clamp(0, H - 1)

        # Anchor assignment: best anchor by (w, h) IoU (YOLOv2 §Dimension Clusters)
        wh_gt = torch.stack([bw, bh], dim=1).unsqueeze(1)   # (N, 1, 2)
        wh_an = anchors.unsqueeze(0)                         # (1, A, 2)
        inter = torch.min(wh_gt, wh_an).prod(-1)             # (N, A)
        union = wh_gt.prod(-1) + wh_an.prod(-1) - inter
        best_anchor = (inter / union.clamp(min=1e-6)).argmax(dim=1)  # (N,)

        for n in range(len(boxes)):
            a   = best_anchor[n].item()
            gi_ = gi[n].item()
            gj_ = gj[n].item()

            tgt_obj[b_idx, a, gj_, gi_] = 1.0
            obj_mask[b_idx, a, gj_, gi_] = True

            # Centre: fractional offset within the responsible cell
            tgt_xy[b_idx, a, gj_, gi_] = torch.stack([
                cx[n] * W - gi[n].float(),
                cy[n] * H - gj[n].float(),
            ])

            # Size: log-ratio relative to anchor (YOLOv2 direct location prediction)
            tgt_wh[b_idx, a, gj_, gi_] = torch.stack([
                torch.log(bw[n] / anchors[a, 0].clamp(min=1e-6)),
                torch.log(bh[n] / anchors[a, 1].clamp(min=1e-6)),
            ])

            # One-hot class vector
            cls_idx = labels[n].item()
            if 0 <= cls_idx < C:
                tgt_cls[b_idx, a, gj_, gi_, cls_idx] = 1.0

    # ── Compute losses ────────────────────────────────────────────────────────
    bce = nn.BCELoss()
    mse = nn.MSELoss()
    bce_logits = nn.BCEWithLogitsLoss()

    # Objectness — BCE over ALL cells (YOLOv3: sigmoid + BCE, not MSE)
    pred_obj = torch.sigmoid(predictions[..., 4])
    loss_obj = bce(pred_obj, tgt_obj)

    # Box regression — MSE over POSITIVE cells only
    if obj_mask.any():
        pred_xy = torch.stack([
            torch.sigmoid(predictions[..., 0]),
            torch.sigmoid(predictions[..., 1]),
        ], dim=-1)
        pred_wh = predictions[..., 2:4]
        loss_xy = mse(pred_xy[obj_mask], tgt_xy[obj_mask])
        loss_wh = mse(pred_wh[obj_mask], tgt_wh[obj_mask])
    else:
        loss_xy = torch.tensor(0.0, device=device)
        loss_wh = torch.tensor(0.0, device=device)

    # Classification — BCE per class over POSITIVE cells (YOLOv3: not softmax CE)
    if obj_mask.any():
        loss_cls = bce_logits(predictions[..., 5:][obj_mask], tgt_cls[obj_mask])
    else:
        loss_cls = torch.tensor(0.0, device=device)

    # Total loss with YOLOv1 weighting (lambda_coord=5 verbatim from paper)
    return lambda_coord * (loss_obj + loss_xy + loss_wh) + lambda_cls * loss_cls


# =============================================================================
# 4.  DECODE PREDICTIONS + NMS
# =============================================================================

def decode_predictions(
    raw_preds: torch.Tensor,
    anchors: torch.Tensor,
    conf_thresh: float = 0.25,
    nms_thresh: float  = 0.45,
) -> list[dict]:
    """
    Convert raw network output (B, 3, 16, 16, 85) to lists of detections.

    Returns list of dicts, one per image:
        'boxes'  : (K, 4) float  — pixel-coordinate xyxy
        'scores' : (K,)   float  — final confidence = obj * class_score
        'labels' : (K,)   int    — contiguous class index 0–79
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

                    cls_probs  = torch.softmax(p[5:], dim=0)
                    cls_score, cls_label = cls_probs.max(0)
                    final_score = obj * cls_score.item()
                    if final_score < conf_thresh:
                        continue

                    # Decode box to pixel coordinates
                    cx = (gi + torch.sigmoid(p[0]).item()) * cell_w
                    cy = (gj + torch.sigmoid(p[1]).item()) * cell_h
                    bw = anchors[a, 0].item() * IMG_SIZE * math.exp(
                        max(-10.0, min(10.0, p[2].item()))
                    )
                    bh = anchors[a, 1].item() * IMG_SIZE * math.exp(
                        max(-10.0, min(10.0, p[3].item()))
                    )

                    boxes_list.append([cx - bw / 2, cy - bh / 2,
                                       cx + bw / 2, cy + bh / 2])
                    scores_list.append(final_score)
                    labels_list.append(cls_label.item())

        if boxes_list:
            bx = torch.tensor(boxes_list, dtype=torch.float32)
            sc = torch.tensor(scores_list, dtype=torch.float32)
            lb = torch.tensor(labels_list, dtype=torch.long)
            keep = nms(bx, sc, nms_thresh)
            results.append({"boxes": bx[keep], "scores": sc[keep], "labels": lb[keep]})
        else:
            results.append({
                "boxes":  torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros(0,      dtype=torch.float32),
                "labels": torch.zeros(0,      dtype=torch.long),
            })

    return results


# =============================================================================
# 5.  DATASET — COCODetectionDataset
# =============================================================================

class COCODetectionDataset(Dataset):
    """
    COCO 2017 detection dataset.

    Reads from the directory layout produced by download_data.py:
        $BLACKHOLE/COCO/
            train2017/          ← JPEG images
            val2017/
            annotations/
                instances_train2017.json
                instances_val2017.json

    Each __getitem__ returns:
        img    : (3, 224, 224) float tensor — DINOv2 normalisation
        target : dict
            'boxes'    : (N, 4) float — xyxy normalised [0, 1]
            'labels'   : (N,)   long  — contiguous class index 0–79
            'image_id' : int
    """

    _transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(DINO_MEAN, DINO_STD),
    ])

    def __init__(self, root: str, split: str = "val"):
        assert split in ("train", "val"), "split must be 'train' or 'val'"
        self.img_dir = os.path.join(root, f"{split}2017")
        ann_file     = os.path.join(root, "annotations",
                                    f"instances_{split}2017.json")

        if not COCO_EVAL_AVAILABLE:
            raise ImportError("pycocotools is required for COCODetectionDataset.")

        self.coco    = COCO(ann_file)
        self.img_ids = sorted(self.coco.imgs.keys())

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        img_id   = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.img_dir, img_info["file_name"])

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        img_tensor = self._transform(img)

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns    = self.coco.loadAnns(ann_ids)

        boxes, labels = [], []
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in CAT_TO_IDX:
                continue
            x, y, w, h = ann["bbox"]          # COCO xywh in pixels
            x1 = x / orig_w
            y1 = y / orig_h
            x2 = (x + w) / orig_w
            y2 = (y + h) / orig_h
            # Clamp to [0, 1] — some annotations slightly exceed image bounds
            x1, y1, x2, y2 = (max(0.0, min(1.0, v))
                               for v in (x1, y1, x2, y2))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(CAT_TO_IDX[cat_id])

        target = {
            "boxes":    torch.tensor(boxes,  dtype=torch.float32)
                        if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels":   torch.tensor(labels, dtype=torch.long)
                        if labels else torch.zeros(0, dtype=torch.long),
            "image_id": img_id,
        }
        return img_tensor, target


def collate_fn(batch):
    """Custom collate: images stacked, targets kept as a list of dicts."""
    images  = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


def get_coco_loaders(
    datadir: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader) for COCO 2017.
    Mirrors get_loaders() from classification.py in signature and style.
    """
    train_set = COCODetectionDataset(datadir, split="train")
    val_set   = COCODetectionDataset(datadir, split="val")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


# =============================================================================
# 6.  TRAINING AND VALIDATION LOOPS
# =============================================================================

def train_one_epoch(
    model: DinoDetector,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    One training epoch.
    Only det_head parameters receive gradients — backbone is frozen.
    Returns: average YOLO loss over the epoch.
    """
    model.train()
    running_loss = 0.0

    for images, targets in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        preds  = model(images)
        loss   = yolo_loss(preds, targets, model.anchors, device)

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping: prevents exploding gradients in early epochs
        nn.utils.clip_grad_norm_(model.det_head.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(
    model: DinoDetector,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """
    Full evaluation with official COCOeval mAP.

    Returns: (avg_loss, map50, map50_95)
        avg_loss  : average YOLO loss on val set
        map50     : mAP @ IoU=0.50
        map50_95  : mAP @ IoU=0.50:0.95  (primary COCO metric)
    """
    if not COCO_EVAL_AVAILABLE:
        print("pycocotools not available — skipping mAP evaluation.")
        return 0.0, 0.0, 0.0

    model.eval()
    running_loss = 0.0
    coco_results = []
    coco_gt      = loader.dataset.coco

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)
            preds  = model(images)
            loss   = yolo_loss(preds, targets, model.anchors, device)
            running_loss += loss.item()

            detections = decode_predictions(preds, model.anchors)
            for det, tgt in zip(detections, targets):
                img_id = tgt["image_id"]
                for k in range(len(det["scores"])):
                    box   = det["boxes"][k]    # xyxy pixel-space
                    score = det["scores"][k].item()
                    label = det["labels"][k].item()
                    coco_results.append({
                        "image_id":    img_id,
                        "category_id": IDX_TO_CAT[label],
                        "bbox": [                        # COCOeval expects xywh
                            box[0].item(),
                            box[1].item(),
                            (box[2] - box[0]).item(),
                            (box[3] - box[1]).item(),
                        ],
                        "score": score,
                    })

    avg_loss = running_loss / len(loader)
    map50, map50_95 = 0.0, 0.0

    if coco_results:
        coco_dt   = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        map50_95 = float(coco_eval.stats[0])   # mAP @ 0.50:0.95
        map50    = float(coco_eval.stats[1])   # mAP @ 0.50
    else:
        print("Warning: no detections above threshold — mAP is 0.")

    return avg_loss, map50, map50_95


# =============================================================================
# 7.  DETECTION VALIDATOR  (for RL agent — mirrors FastProxyValidator)
# =============================================================================

class DetectionValidator:
    """
    Rapid detection evaluator for the RL environment.

    Mirrors the interface of FastProxyValidator from rl_utils.py:
        validator.evaluate(mask_1d) → (avg_loss, proxy_map)

    Key adaptations vs. FastProxyValidator:
        - Caches patch tokens x[:,5:] instead of CLS token
        - Uses YOLO loss instead of CrossEntropyLoss
        - proxy_map = mean objectness (fast surrogate for mAP)
          Full COCOeval runs only at training time, not per RL step.
    """

    def __init__(
        self,
        model: DinoDetector,
        proxy_loader: DataLoader,
        device: torch.device,
    ):
        self.model  = model
        self.device = device
        # Cache: list of (cached_tokens, targets) tuples — one per batch
        self.cached_tokens: list[torch.Tensor] = []
        self.cached_targets: list[list[dict]]  = []
        self._build_cache(proxy_loader)

    def _build_cache(self, proxy_loader: DataLoader) -> None:
        """
        Pre-compute and store transformer tokens for all proxy images.
        Skips patch embedding on every RL step → significant speedup.
        """
        self.model.eval()
        n = len(proxy_loader.dataset)
        print(f"Building Detection Proxy Cache for {n} samples...")

        with torch.no_grad():
            for images, targets in proxy_loader:
                images = images.to(self.device)
                # Store full token sequence — we slice x[:,5:] at eval time
                # so the cache works even if token layout changes.
                tokens = self.model.transformer.prepare_tokens_with_masks(images)
                self.cached_tokens.append(tokens)
                self.cached_targets.append(targets)

        print("Detection Proxy Cache built successfully.")

    def evaluate(
        self,
        mask_1d: torch.Tensor,
    ) -> tuple[float, float]:
        """
        Fast evaluation using cached tokens.

        Args:
            mask_1d: (144,) float tensor — current pruning mask

        Returns:
            (avg_loss, proxy_map)
            proxy_map = mean objectness score — cheap mAP surrogate.
            The RL agent uses this as its per-step reward signal.
        """
        self.model.eval()
        self.model.set_mask(mask_1d)

        running_loss      = 0.0
        total_objectness  = 0.0
        num_batches       = len(self.cached_tokens)

        with torch.no_grad():
            for tokens, targets in zip(self.cached_tokens, self.cached_targets):
                # Re-run transformer blocks with the updated pruning mask
                x = tokens
                for blk in self.model.transformer.blocks:
                    x = blk(x)
                x = self.model.transformer.norm(x)

                # Extract patch tokens (skip CLS + 4 registers)
                patches  = x[:, 5:, :]
                B        = patches.size(0)
                feat_map = patches.permute(0, 2, 1).view(
                    B, FEAT_DIM, GRID_SIZE, GRID_SIZE
                )
                raw   = self.model.det_head(feat_map)
                preds = raw.reshape(B, NUM_ANCHORS, GRID_SIZE, GRID_SIZE, 5 + NUM_CLASSES)

                loss = yolo_loss(preds, targets, self.model.anchors, self.device)
                running_loss     += loss.item()
                # Proxy mAP: mean sigmoid objectness across all cells
                total_objectness += torch.sigmoid(preds[..., 4]).mean().item()

        avg_loss   = running_loss     / num_batches
        proxy_map  = total_objectness / num_batches
        return avg_loss, proxy_map


# =============================================================================
# 8.  MAIN — Training loop with checkpoint + curve saving
# =============================================================================

def main(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = get_coco_loaders(
        datadir=args.datadir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"COCO loaded — train: {len(train_loader.dataset)} images  "
          f"val: {len(val_loader.dataset)} images")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DinoDetector(device=device).to(device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load existing checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}")

    # Only optimise the detection head — backbone stays frozen
    optimizer = optim.Adam(model.det_head.parameters(), lr=args.lr)
    # Cosine annealing: smoothly decays lr from args.lr to 1e-5 over epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    train_losses, val_losses, map50_list = [], [], []
    best_map50 = 0.0
    no_improve = 0

    for epoch in range(args.epochs):
        t_loss = train_one_epoch(model, train_loader, optimizer, device)
        v_loss, map50, map50_95 = evaluate(model, val_loader, device)
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
        ckpt_path = os.path.join(args.checkpoint_dir, "dino_coco_latest.pth")
        torch.save({
            "epoch":            epoch + 1,
            "model_state_dict": model.state_dict(),
            "map50":            map50,
            "map50_95":         map50_95,
        }, ckpt_path)

        # Save best checkpoint
        if map50 > best_map50 + 0.005:
            best_map50 = map50
            no_improve = 0
            best_path  = os.path.join(args.checkpoint_dir, "dino_coco_best.pth")
            torch.save({
                "epoch":            epoch + 1,
                "model_state_dict": model.state_dict(),
                "map50":            map50,
                "map50_95":         map50_95,
            }, best_path)
            print(f"  → New best mAP@0.5: {best_map50:.4f} saved.")
        else:
            no_improve += 1

        # Early stopping: plateau criterion (see training guide)
        if no_improve >= args.patience:
            print(f"Plateau for {args.patience} epochs — stopping early.")
            break

    # ── Training curves ───────────────────────────────────────────────────────
    os.makedirs("figure", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs_ran = range(1, len(train_losses) + 1)
    ax1.plot(epochs_ran, train_losses, label="Train Loss")
    ax1.plot(epochs_ran, val_losses,   label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend()
    ax1.set_title("YOLO Loss")

    ax2.plot(epochs_ran, map50_list, label="mAP@0.5", color="green")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("mAP"); ax2.legend()
    ax2.set_title("Validation mAP@0.5")

    curve_path = "figure/trainval_curve_coco.jpg"
    plt.tight_layout()
    plt.savefig(curve_path)
    print(f"Training curves saved to {curve_path}")


# =============================================================================
# 9.  CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AttentionSurgeon — DINOv2 YOLO Detection on COCO"
    )
    # Paths
    parser.add_argument("--datadir",        type=str,
                        default=os.path.join(os.environ.get("BLACKHOLE", "."), "COCO"),
                        help="Path to COCO root (contains train2017/, val2017/, annotations/)") # export BLACKHOLE=/dtu/blackhole/10/224464
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/",
                        help="Directory to save model checkpoints")
    parser.add_argument("--checkpoint",     type=str, default=None,
                        help="Optional: path to resume from a saved checkpoint")
    # Hyperparameters
    parser.add_argument("--epochs",     type=int,   default=40)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,   default=5,
                        help="Early stopping patience (epochs without mAP improvement)")
    # Hardware
    parser.add_argument("--num_workers", type=int,  default=4)
    parser.add_argument("--device",      type=str,  default=None,
                        help="Force device: cuda, mps, cpu")

    args = parser.parse_args()
    main(args)
