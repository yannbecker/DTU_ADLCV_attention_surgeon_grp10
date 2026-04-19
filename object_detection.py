"""
object_detection.py
AttentionSurgeon — COCO Object Detection Pipeline

Architecture:
  - Frozen DINOv2 ViT-B/14 backbone (with head-pruning mask support via set_mask)
  - YOLO-style detection head plugged on top of DINOv2 patch-token feature maps
  - Linear probe approach: only the detection head is trained; backbone is frozen
  - Integrates with the same set_mask / pruning interface used in classification.py and agent.py

COCO dataset is loaded from $BLACKHOLE/COCO (same layout as download_data.py).

Usage:
  # Train the detection head
  python object_detection.py --mode train --datadir $BLACKHOLE/COCO --epochs 10

  # Evaluate mAP on val2017
  python object_detection.py --mode eval --datadir $BLACKHOLE/COCO --checkpoint checkpoints/dino_detection_latest.pth
"""

import os
import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision.ops import nms
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils import load_model  # same helper used in classification.py

# ---------------------------------------------------------------------------
# Constants — DINOv2 ViT-B/14 on 224x224 images
# ---------------------------------------------------------------------------
NUM_LAYERS   = 12
NUM_HEADS    = 12
HEAD_DIM     = 768 // NUM_HEADS   # 64
TOTAL_HEADS  = NUM_LAYERS * NUM_HEADS  # 144
PATCH_SIZE   = 14
IMG_SIZE     = 224
NUM_PATCHES  = (IMG_SIZE // PATCH_SIZE) ** 2  # 256  (16x16 grid)
FEAT_DIM     = 768                             # DINOv2 embedding dim
GRID_SIZE    = IMG_SIZE // PATCH_SIZE          # 16
NUM_CLASSES  = 80                              # COCO foreground classes


# ---------------------------------------------------------------------------
# 1. DINOv2 backbone wrapper  (mirrors DinoClassifier from classification.py)
# ---------------------------------------------------------------------------

class DinoDetector(nn.Module):
    """
    Frozen DINOv2 backbone + lightweight YOLO-style detection head.

    The backbone exposes the same set_mask() / get_taylor_importance() API
    used by the RL environment (agent.py) and the baseline (baseline.py),
    so the pruning infrastructure works without modification.
    """

    def __init__(self, device, num_classes=NUM_CLASSES, num_anchors=3):
        super().__init__()
        self.device      = device
        self.num_heads   = NUM_HEADS
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # ---- backbone (frozen) ----
        self.transformer = load_model(device)
        for param in self.transformer.parameters():
            param.requires_grad = False

        # ---- head-pruning mask (144,) float — same API as classification.py ----
        self.mask  = torch.ones(NUM_LAYERS, NUM_HEADS, device=device)
        self.hooks = []
        self._register_pruning_hooks()

        # ---- YOLO-style detection head ----
        # Input : patch tokens reshaped to (B, FEAT_DIM, GRID_SIZE, GRID_SIZE)
        # Output: (B, num_anchors * (5 + num_classes), GRID_SIZE, GRID_SIZE)
        out_ch = num_anchors * (5 + num_classes)  # 5 = tx, ty, tw, th, objectness
        self.det_head = nn.Sequential(
            nn.Conv2d(FEAT_DIM, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, out_ch, kernel_size=1),
        )

        # Default anchors (normalised 0-1, w x h) — tuned for COCO on a 16x16 grid
        self.register_buffer(
            "anchors",
            torch.tensor([
                [0.28, 0.22],
                [0.38, 0.48],
                [0.90, 0.78],
            ], dtype=torch.float32),
        )  # shape (num_anchors, 2)

    # ------------------------------------------------------------------
    # Pruning hooks  (identical pattern to classification.py)
    # ------------------------------------------------------------------

    def _register_pruning_hooks(self):
        """Attach forward hooks that zero-out pruned attention-head outputs."""
        for h in self.hooks:
            h.remove()
        self.hooks = []

        def _make_hook(layer_idx):
            def hook_fn(module, input, output):
                mask_layer = self.mask[layer_idx]                    # (12,)
                full_mask  = mask_layer.repeat_interleave(HEAD_DIM).to(output.device)
                return output * full_mask
            return hook_fn

        for i in range(NUM_LAYERS):
            layer  = self.transformer.blocks[i].attn
            handle = layer.register_forward_hook(_make_hook(i))
            self.hooks.append(handle)

    def set_mask(self, mask_1d: torch.Tensor):
        """
        Update the pruning mask.  mask_1d: (144,) — same API as classification.py.
        Called by the RL environment (agent.py) and magnitude baseline (baseline.py).
        """
        self.mask = mask_1d.view(NUM_LAYERS, NUM_HEADS).to(self.device)

    def set_heads_require_grad(self, requires_grad: bool = True):
        for i in range(NUM_LAYERS):
            self.transformer.blocks[i].attn.requires_grad_(requires_grad)

    # ------------------------------------------------------------------
    # Taylor importance  (same as classification.py — used by HeadCensus)
    # ------------------------------------------------------------------

    def get_taylor_importance(self, images, labels_unused, criterion_unused):
        """
        Returns a (12, 12) Taylor importance matrix for the detection backbone.
        Uses the L2-norm of the detection feature map as a surrogate loss so
        the method is self-contained (no ground-truth boxes required at census time).
        """
        self.eval()
        taylor_scores = torch.zeros(NUM_LAYERS, NUM_HEADS, device=self.device)
        activations, grads = {}, {}

        def save_act(name):
            def hook(mod, inp, out):
                out.requires_grad_(True)
                activations[name] = out.detach()
                out.register_hook(lambda g: grads.update({name: g}))
                return out
            return hook

        temp_hooks = []
        for i in range(NUM_LAYERS):
            h = self.transformer.blocks[i].attn.register_forward_hook(save_act(f"layer{i}"))
            temp_hooks.append(h)

        with torch.set_grad_enabled(True):
            feat_map = self._extract_patch_features(images)
            loss_proxy = feat_map.norm()   # differentiable surrogate
            self.zero_grad()
            loss_proxy.backward()

        for i in range(NUM_LAYERS):
            key = f"layer{i}"
            if key in grads:
                act  = activations[key]                               # (B, N, 768)
                grad = grads[key]
                act  = act.view(act.size(0), act.size(1), NUM_HEADS, HEAD_DIM)
                grad = grad.view(grad.size(0), grad.size(1), NUM_HEADS, HEAD_DIM)
                score = torch.abs(act * grad).sum(dim=-1).mean(dim=(0, 1))
                taylor_scores[i] = score

        for h in temp_hooks:
            h.remove()
        return taylor_scores

    def get_intra_layer_ranks(self, taylor_importance_matrix: torch.Tensor):
        """Normalised (0-1) rank matrix — mirrors classification.py."""
        ranks = torch.argsort(
            torch.argsort(taylor_importance_matrix, dim=1), dim=1
        ).float()
        return ranks / (self.num_heads - 1)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _extract_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Run images through the (possibly masked) DINOv2 backbone.
        Returns patch-token features reshaped for 2-D convolution:
            (B, FEAT_DIM, GRID_SIZE, GRID_SIZE)
        """
        x = self.transformer.prepare_tokens_with_masks(images)
        for blk in self.transformer.blocks:
            x = blk(x)
        x = self.transformer.norm(x)
        # Drop the CLS token; keep the 256 patch tokens
        patch_tokens = x[:, 1:, :]                                   # (B, 256, 768)
        B = patch_tokens.size(0)
        feat_map = patch_tokens.permute(0, 2, 1)                      # (B, 768, 256)
        feat_map = feat_map.view(B, FEAT_DIM, GRID_SIZE, GRID_SIZE)
        return feat_map

    def forward(self, images: torch.Tensor):
        """
        Returns raw detection tensor:
            (B, num_anchors, GRID_SIZE, GRID_SIZE, 5 + num_classes)
        """
        feat_map = self._extract_patch_features(images)               # (B, 768, 16, 16)
        raw      = self.det_head(feat_map)                            # (B, A*(5+C), 16, 16)
        B, _, H, W = raw.shape
        A = self.num_anchors
        C = self.num_classes
        raw = raw.view(B, A, 5 + C, H, W).permute(0, 1, 3, 4, 2)    # (B, A, H, W, 5+C)
        return raw


# ---------------------------------------------------------------------------
# 2. YOLO-style loss
# ---------------------------------------------------------------------------

def yolo_loss(predictions, targets, anchors, img_size=IMG_SIZE, device="cpu"):
    """
    Simplified YOLO loss = objectness + bbox regression + classification.

    predictions : (B, A, H, W, 5+C)
    targets     : list of dicts — {'boxes': (N,4) xyxy normalised, 'labels': (N,)}
    anchors     : (A, 2) normalised (w, h)
    """
    B, A, H, W, _ = predictions.shape
    C = predictions.shape[-1] - 5

    tx  = torch.sigmoid(predictions[..., 0])
    ty  = torch.sigmoid(predictions[..., 1])
    tw  = predictions[..., 2]
    th  = predictions[..., 3]
    obj = torch.sigmoid(predictions[..., 4])
    cls = predictions[..., 5:]

    tgt_obj  = torch.zeros(B, A, H, W,     device=device)
    tgt_xy   = torch.zeros(B, A, H, W, 2,  device=device)
    tgt_wh   = torch.zeros(B, A, H, W, 2,  device=device)
    tgt_cls  = torch.zeros(B, A, H, W, C,  device=device)
    obj_mask = torch.zeros(B, A, H, W,     device=device, dtype=torch.bool)

    for b_idx, target in enumerate(targets):
        boxes  = target["boxes"].to(device)
        labels = target["labels"].to(device)
        if boxes.numel() == 0:
            continue

        # xyxy → cxcywh (normalised)
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        bw = boxes[:, 2] - boxes[:, 0]
        bh = boxes[:, 3] - boxes[:, 1]

        gi = (cx * W).long().clamp(0, W - 1)
        gj = (cy * H).long().clamp(0, H - 1)

        # Assign best anchor by (w, h) IoU
        wh_gt = torch.stack([bw, bh], dim=1).unsqueeze(1)           # (N, 1, 2)
        wh_an = anchors.unsqueeze(0)                                  # (1, A, 2)
        inter = torch.min(wh_gt, wh_an).prod(dim=-1)
        union = wh_gt.prod(dim=-1) + wh_an.prod(dim=-1) - inter
        best_anchor = (inter / union.clamp(min=1e-6)).argmax(dim=1)

        for n in range(len(boxes)):
            a   = best_anchor[n].item()
            gi_ = gi[n].item()
            gj_ = gj[n].item()
            tgt_obj[b_idx, a, gj_, gi_]  = 1.0
            tgt_xy [b_idx, a, gj_, gi_]  = torch.stack([
                cx[n] * W - gi[n].float(),
                cy[n] * H - gj[n].float(),
            ])
            tgt_wh [b_idx, a, gj_, gi_]  = torch.stack([
                torch.log(bw[n] / anchors[a, 0].clamp(min=1e-6)),
                torch.log(bh[n] / anchors[a, 1].clamp(min=1e-6)),
            ])
            if labels[n] < C:
                tgt_cls[b_idx, a, gj_, gi_, labels[n]] = 1.0
            obj_mask[b_idx, a, gj_, gi_] = True

    bce = nn.BCELoss()
    mse = nn.MSELoss()
    ce  = nn.BCEWithLogitsLoss()

    loss_obj = bce(obj, tgt_obj)
    loss_xy  = mse(torch.stack([tx, ty], dim=-1)[obj_mask],
                   tgt_xy[obj_mask]) if obj_mask.any() else torch.tensor(0., device=device)
    loss_wh  = mse(torch.stack([tw, th], dim=-1)[obj_mask],
                   tgt_wh[obj_mask]) if obj_mask.any() else torch.tensor(0., device=device)
    loss_cls = ce(cls[obj_mask],
                  tgt_cls[obj_mask]) if obj_mask.any() else torch.tensor(0., device=device)

    total = 5.0 * loss_obj + 5.0 * (loss_xy + loss_wh) + loss_cls
    return total, loss_obj.item(), (loss_xy + loss_wh).item(), loss_cls.item()


# ---------------------------------------------------------------------------
# 3. COCO dataset loader  (reads from $BLACKHOLE/COCO — same as download_data.py)
# ---------------------------------------------------------------------------

DINO_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


class COCODetectionDataset(torch.utils.data.Dataset):
    """
    Lightweight COCO detection dataset.
    Boxes are returned normalised (0-1) in xyxy format.
    Labels are zero-indexed (COCO category id mapped to 0..79).
    """

    def __init__(self, img_dir: str, ann_file: str, transform=DINO_TRANSFORM):
        self.coco      = COCO(ann_file)
        self.img_dir   = img_dir
        self.transform = transform

        # Keep only images with at least one annotation
        self.img_ids = sorted([
            img_id for img_id in self.coco.imgs
            if len(self.coco.getAnnIds(imgIds=img_id)) > 0
        ])

        # Contiguous category mapping (80 COCO classes → 0-indexed)
        cats = sorted(self.coco.cats.keys())
        self.cat_to_idx = {cat: idx for idx, cat in enumerate(cats)}

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        from PIL import Image
        img_id   = self.img_ids[index]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.img_dir, img_info["file_name"])

        img = Image.open(img_path).convert("RGB")
        W0, H0 = img.size

        if self.transform:
            img = self.transform(img)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns    = self.coco.loadAnns(ann_ids)

        boxes, labels = [], []
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            x, y, w, h = ann["bbox"]
            x1 = max(0., min(1., x / W0))
            y1 = max(0., min(1., y / H0))
            x2 = max(0., min(1., (x + w) / W0))
            y2 = max(0., min(1., (y + h) / H0))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(self.cat_to_idx[ann["category_id"]])

        boxes  = torch.tensor(boxes,  dtype=torch.float32) if boxes  else torch.zeros((0, 4))
        labels = torch.tensor(labels, dtype=torch.long)    if labels else torch.zeros((0,), dtype=torch.long)
        return img, {"boxes": boxes, "labels": labels, "image_id": img_id}


def collate_fn(batch):
    images  = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets


def get_loaders(data_dir: str, batch_size: int, num_workers: int):
    """
    Builds train and val DataLoaders from $BLACKHOLE/COCO.
    Mirrors the get_loaders() signature in classification.py.
    """
    train_img_dir = os.path.join(data_dir, "train2017")
    val_img_dir   = os.path.join(data_dir, "val2017")
    train_ann     = os.path.join(data_dir, "annotations", "instances_train2017.json")
    val_ann       = os.path.join(data_dir, "annotations", "instances_val2017.json")

    train_ds = COCODetectionDataset(train_img_dir, train_ann)
    val_ds   = COCODetectionDataset(val_img_dir,   val_ann)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        preds  = model(images)
        loss, _, _, _ = yolo_loss(preds, targets, model.anchors, device=device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


# ---------------------------------------------------------------------------
# 5. Evaluation — mAP via pycocotools  (also used by the RL environment)
# ---------------------------------------------------------------------------

def decode_predictions(raw_preds, anchors, img_size=IMG_SIZE,
                       conf_thresh=0.25, nms_thresh=0.45):
    """
    Convert raw network output to a list of dicts:
        {'boxes': (N,4) xyxy pixel coords, 'scores': (N,), 'labels': (N,)}
    One dict per image in the batch.
    """
    B, A, H, W, _ = raw_preds.shape
    cell_w = img_size / W
    cell_h = img_size / H
    results = []

    for b in range(B):
        all_boxes, all_scores, all_labels = [], [], []
        for a in range(A):
            for gj in range(H):
                for gi in range(W):
                    pred = raw_preds[b, a, gj, gi]
                    obj  = torch.sigmoid(pred[4]).item()
                    if obj < conf_thresh:
                        continue
                    cls_probs    = torch.softmax(pred[5:], dim=0)
                    score, label = cls_probs.max(0)
                    score = (score * obj).item()
                    if score < conf_thresh:
                        continue
                    cx = (gi + torch.sigmoid(pred[0]).item()) * cell_w
                    cy = (gj + torch.sigmoid(pred[1]).item()) * cell_h
                    bw = anchors[a, 0].item() * img_size * math.exp(
                        max(-10., min(10., pred[2].item())))
                    bh = anchors[a, 1].item() * img_size * math.exp(
                        max(-10., min(10., pred[3].item())))
                    x1 = cx - bw / 2;  y1 = cy - bh / 2
                    x2 = cx + bw / 2;  y2 = cy + bh / 2
                    all_boxes.append([x1, y1, x2, y2])
                    all_scores.append(score)
                    all_labels.append(label.item())

        if all_boxes:
            boxes  = torch.tensor(all_boxes,  dtype=torch.float32)
            scores = torch.tensor(all_scores, dtype=torch.float32)
            labels = torch.tensor(all_labels, dtype=torch.long)
            keep   = nms(boxes, scores, nms_thresh)
            results.append({"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]})
        else:
            results.append({
                "boxes":  torch.zeros((0, 4)),
                "scores": torch.zeros(0),
                "labels": torch.zeros(0, dtype=torch.long),
            })
    return results


def evaluate(model, loader, device, conf_thresh=0.25, nms_thresh=0.45):
    """
    Full COCO mAP evaluation via pycocotools.
    Returns (map50_95, map50) as floats.
    Mirrors the validate() signature in classification.py.
    """
    model.eval()
    coco_gt  = loader.dataset.coco
    cat_list = sorted(coco_gt.cats.keys())   # original COCO category ids

    coco_results = []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)
            raw    = model(images)
            dets   = decode_predictions(raw, model.anchors,
                                        conf_thresh=conf_thresh,
                                        nms_thresh=nms_thresh)
            for det, tgt in zip(dets, targets):
                img_id = tgt["image_id"]
                for k in range(len(det["scores"])):
                    x1, y1, x2, y2 = det["boxes"][k].tolist()
                    cat_id = cat_list[int(det["labels"][k])] \
                             if int(det["labels"][k]) < len(cat_list) else 1
                    coco_results.append({
                        "image_id":    img_id,
                        "category_id": cat_id,
                        "bbox":        [x1, y1, x2 - x1, y2 - y1],
                        "score":       float(det["scores"][k]),
                    })

    if not coco_results:
        print("Warning: no detections produced — mAP = 0.0")
        return 0.0, 0.0

    coco_dt   = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return float(coco_eval.stats[0]), float(coco_eval.stats[1])


# ---------------------------------------------------------------------------
# 6. RL environment helpers  (used by agent.py for the detection task)
# ---------------------------------------------------------------------------

class DetectionValidator:
    """
    Lightweight wrapper used by the RL PruningEnv (agent.py) to rapidly
    score the current mask on the detection task.

    Caches pre-processed images so the backbone patch-embedding step is
    only paid once — mirrors FastProxyValidator from rl_utils.py.

    Usage inside agent.py:
        validator = DetectionValidator(model, proxy_loader, device)
        loss, map_proxy = validator.evaluate(mask_1d)
    """

    def __init__(self, model: DinoDetector, proxy_loader: DataLoader, device):
        self.model        = model
        self.proxy_loader = proxy_loader
        self.device       = device
        self._build_cache()

    def _build_cache(self):
        self.cached_images  = []
        self.cached_targets = []
        self.model.eval()
        print(f"Building detection proxy cache "
              f"({len(self.proxy_loader.dataset)} samples)…")
        with torch.no_grad():
            for images, targets in self.proxy_loader:
                self.cached_images.append(images.to(self.device))
                self.cached_targets.extend(targets)
        print("Detection proxy cache ready.")

    def evaluate(self, mask_1d: torch.Tensor):
        """
        Apply mask_1d and run a fast forward pass on cached images.
        Returns (avg_loss, map_proxy) where map_proxy is a cheap
        mean-objectness surrogate (avoids full COCOeval overhead per RL step).
        """
        self.model.set_mask(mask_1d)
        self.model.eval()
        total_loss = 0.0
        all_scores = []

        with torch.no_grad():
            for i, images in enumerate(self.cached_images):
                batch_targets = self.cached_targets[
                    i * images.size(0): (i + 1) * images.size(0)
                ]
                raw  = self.model(images)
                loss, _, _, _ = yolo_loss(
                    raw, batch_targets, self.model.anchors, device=self.device
                )
                total_loss += loss.item()
                dets = decode_predictions(raw, self.model.anchors, conf_thresh=0.25)
                for d in dets:
                    s = d["scores"].mean().item() if d["scores"].numel() > 0 else 0.0
                    all_scores.append(s)

        avg_loss  = total_loss / max(len(self.cached_images), 1)
        map_proxy = float(np.mean(all_scores)) if all_scores else 0.0
        return avg_loss, map_proxy


def get_proxy_loader(base_loader: DataLoader, num_samples: int = 200,
                     seed: int = 42) -> DataLoader:
    """
    Returns a small fixed DataLoader for RL proxy evaluation.
    Mirrors get_proxy_loader() in rl_utils.py.
    """
    dataset     = base_loader.dataset
    num_samples = min(num_samples, len(dataset))
    np.random.seed(seed)
    indices     = np.random.choice(len(dataset), num_samples, replace=False)
    proxy_ds    = Subset(dataset, indices)
    return DataLoader(
        proxy_ds,
        batch_size=base_loader.batch_size,
        shuffle=False,
        num_workers=base_loader.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # ---- Data ----
    train_loader, val_loader = get_loaders(
        args.datadir, args.batch_size, args.num_workers
    )
    print(f"COCO  train: {len(train_loader.dataset)} images  "
          f"val: {len(val_loader.dataset)} images")

    # ---- Model ----
    model = DinoDetector(device, num_classes=NUM_CLASSES).to(device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}")

    # ---- Mode ----
    if args.mode == "train":
        optimizer = optim.Adam(model.det_head.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
        train_losses, map50_list = [], []

        for epoch in range(args.epochs):
            t_loss          = train_one_epoch(model, train_loader, optimizer, device)
            map50_95, map50 = evaluate(model, val_loader, device)
            scheduler.step()
            train_losses.append(t_loss)
            map50_list.append(map50)
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Loss: {t_loss:.4f} | "
                  f"mAP@0.5:0.95: {map50_95:.4f} | mAP@0.5: {map50:.4f}")

            ckpt_path = os.path.join(
                args.checkpoint_dir, "dino_detection_latest.pth"
            )
            torch.save({
                "epoch":            epoch + 1,
                "model_state_dict": model.state_dict(),
                "map50":            map50,
                "map50_95":         map50_95,
            }, ckpt_path)

        # Training curve
        os.makedirs("figure", exist_ok=True)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, args.epochs + 1), train_losses, label="Train Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(range(1, args.epochs + 1), map50_list,
                 label="mAP@0.5", color="orange")
        plt.xlabel("Epoch"); plt.ylabel("mAP"); plt.legend()
        plt.tight_layout()
        plt.savefig("figure/detection_training_curve.jpg")
        print("Saved figure/detection_training_curve.jpg")

    elif args.mode == "eval":
        map50_95, map50 = evaluate(model, val_loader, device)
        print(f"mAP@0.5:0.95 = {map50_95:.4f}  |  mAP@0.5 = {map50:.4f}")

    else:
        raise ValueError(f"Unknown mode: {args.mode}  (choose 'train' or 'eval')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AttentionSurgeon — COCO Object Detection (DINOv2 + YOLO head)"
    )
    # Paths
    parser.add_argument(
        "--datadir", type=str,
        default=os.path.join(os.environ.get("BLACKHOLE", ".data"), "COCO"),
        help="Path to COCO dataset root  (default: $BLACKHOLE/COCO)",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints",
        help="Directory to save / load checkpoints",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a checkpoint to resume from",
    )
    # Mode
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "eval"],
        help="train or eval",
    )
    # Hyperparameters
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int,   default=4)
    # Hardware
    parser.add_argument(
        "--device", type=str, default=None,
        help="Force device (cuda / cpu / mps)",
    )

    args = parser.parse_args()
    main(args)
