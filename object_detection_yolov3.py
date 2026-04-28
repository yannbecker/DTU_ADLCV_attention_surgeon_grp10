import os
import math
import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision.ops import nms
from tqdm.auto import tqdm
from PIL import Image

from classification import DinoClassifier

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    COCO_EVAL_AVAILABLE = True
except ImportError:
    COCO_EVAL_AVAILABLE = False
    print("Warning: pycocotools not found. Full mAP evaluation disabled.")

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_LAYERS  = 12
NUM_HEADS   = 12
HEAD_DIM    = 64
FEAT_DIM    = 768
GRID_SIZE   = 16
IMG_SIZE    = 224
NUM_CLASSES = 80
NUM_ANCHORS = 3

ANCHOR_PRIORS = [[0.28, 0.22], [0.38, 0.48], [0.90, 0.78]]

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


# ── Model ─────────────────────────────────────────────────────────────────────

class DinoDetector(DinoClassifier):
    def __init__(self, device="cpu"):
        super().__init__(device=device, num_classes=1)
        del self.classifier

        out_ch = NUM_ANCHORS * (5 + NUM_CLASSES)   # 255
        self.det_head = nn.Sequential(
            nn.Conv2d(FEAT_DIM, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(256, out_ch, 1),
        )
        self.register_buffer("anchors", torch.tensor(ANCHOR_PRIORS, dtype=torch.float32))

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        # Token layout: [CLS | REG×4 | PATCH×256]; patches are indices 5:261
        x = self.transformer.prepare_tokens_with_masks(images)
        for blk in self.transformer.blocks:
            x = blk(x)
        x = self.transformer.norm(x)
        patches = x[:, 5:, :]                              # (B, 256, 768)
        B = patches.size(0)
        return patches.permute(0, 2, 1).reshape(B, FEAT_DIM, GRID_SIZE, GRID_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x) if x.shape[1] == 3 else x
        raw = self.det_head(feat)                           # (B, 255, 16, 16)
        B, _, H, W = raw.shape
        return raw.reshape(B, NUM_ANCHORS, H, W, 5 + NUM_CLASSES)

    def get_taylor_importance(self, images: torch.Tensor, targets: list) -> torch.Tensor:
        # Pass raw images — gradients must flow through backbone blocks.
        was_training = self.training
        self.train()
        taylor_scores = torch.zeros(NUM_LAYERS, NUM_HEADS, device=images.device)
        activations, grads = {}, {}

        def save_activation(name):
            def hook(module, input, output):
                output.requires_grad_(True)
                activations[name] = output.detach()
                output.register_hook(lambda g: grads.update({name: g}))
            return hook

        temp_hooks = [
            self.transformer.blocks[i].attn.register_forward_hook(save_activation(f"layer{i}"))
            for i in range(NUM_LAYERS)
        ]

        with torch.set_grad_enabled(True):
            preds = self.forward(images)
            loss  = yolo_loss(preds, targets, self.anchors, images.device)
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
        if not was_training:
            self.eval()
        return taylor_scores


# ── YOLO Loss ─────────────────────────────────────────────────────────────────

def yolo_loss(
    predictions:   torch.Tensor,
    targets:       list,
    anchors:       torch.Tensor,
    device:        torch.device,
    lambda_coord:  float = 10.0,   # higher than paper's 5 because reduction="mean"
    lambda_cls:    float = 0.5,
    lambda_noobj:  float = 0.5,
    ignore_thresh: float = 0.5,
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

        wh_gt = torch.stack([bw, bh], dim=1).unsqueeze(1)
        wh_an = anchors.unsqueeze(0)
        inter = torch.min(wh_gt, wh_an).prod(-1)
        union = wh_gt.prod(-1) + wh_an.prod(-1) - inter
        best_anchor = (inter / union.clamp(min=1e-6)).argmax(dim=1)

        for n in range(len(boxes)):
            a, gi_, gj_ = best_anchor[n].item(), gi[n].item(), gj[n].item()
            tgt_obj [b_idx, a, gj_, gi_] = 1.0
            obj_mask[b_idx, a, gj_, gi_] = True
            tgt_xy  [b_idx, a, gj_, gi_] = torch.stack([
                cx[n] * W - gi[n].float(), cy[n] * H - gj[n].float()])
            tgt_wh  [b_idx, a, gj_, gi_] = torch.stack([
                torch.log(bw[n] / anchors[a, 0].clamp(min=1e-6)),
                torch.log(bh[n] / anchors[a, 1].clamp(min=1e-6))])
            cls_idx = labels[n].item()
            if 0 <= cls_idx < C:
                tgt_cls[b_idx, a, gj_, gi_, cls_idx] = 1.0

    mse        = nn.MSELoss(reduction="mean")
    bce_elem   = nn.BCELoss(reduction="none")
    bce_logits = nn.BCEWithLogitsLoss()

    pred_obj     = torch.sigmoid(predictions[..., 4])
    obj_loss_map = bce_elem(pred_obj, tgt_obj)

    with torch.no_grad():
        ignore_mask = (~obj_mask) & (pred_obj > ignore_thresh)

    loss_obj   = obj_loss_map[obj_mask].mean()   if obj_mask.any()  else torch.tensor(0., device=device)
    noobj_mask = ~obj_mask & ~ignore_mask
    loss_noobj = obj_loss_map[noobj_mask].mean() if noobj_mask.any() else torch.tensor(0., device=device)

    if obj_mask.any():
        pred_xy  = torch.stack([torch.sigmoid(predictions[..., 0]),
                                 torch.sigmoid(predictions[..., 1])], dim=-1)
        pred_wh  = predictions[..., 2:4]
        loss_xy  = mse(pred_xy[obj_mask], tgt_xy[obj_mask])
        loss_wh  = mse(pred_wh[obj_mask], tgt_wh[obj_mask])
        loss_cls = bce_logits(predictions[..., 5:][obj_mask], tgt_cls[obj_mask])
    else:
        loss_xy = loss_wh = loss_cls = torch.tensor(0., device=device)

    return (lambda_coord * (loss_xy + loss_wh)
            + loss_obj
            + lambda_noobj * loss_noobj
            + lambda_cls * loss_cls)


# ── Decode + NMS ──────────────────────────────────────────────────────────────

def decode_predictions(
    raw_preds:  torch.Tensor,
    anchors:    torch.Tensor,
    conf_thresh: float = 0.25,
    nms_thresh:  float = 0.45,
) -> list:
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
                    # sigmoid not softmax — trained with BCEWithLogitsLoss per class
                    cls_probs         = torch.sigmoid(p[5:])
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
                    boxes_list.append([cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2])
                    scores_list.append(final_score)
                    labels_list.append(cls_label.item())

        if boxes_list:
            bx   = torch.tensor(boxes_list, dtype=torch.float32)
            sc   = torch.tensor(scores_list, dtype=torch.float32)
            lb   = torch.tensor(labels_list, dtype=torch.long)
            keep = nms(bx, sc, nms_thresh)
            results.append({"boxes": bx[keep], "scores": sc[keep], "labels": lb[keep]})
        else:
            results.append({
                "boxes":  torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros(0,      dtype=torch.float32),
                "labels": torch.zeros(0,      dtype=torch.long),
            })
    return results


# ── Datasets ──────────────────────────────────────────────────────────────────

class COCODetectionDataset(Dataset):
    _transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(DINO_MEAN, DINO_STD),
    ])

    def __init__(self, root: str, split: str = "val"):
        assert split in ("train", "val")
        self.img_dir = os.path.join(root, f"{split}2017")
        ann_file     = os.path.join(root, "annotations", f"instances_{split}2017.json")
        if not COCO_EVAL_AVAILABLE:
            raise ImportError("pycocotools is required.")
        self.coco    = COCO(ann_file)
        self.img_ids = sorted(self.coco.imgs.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id   = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        img      = Image.open(os.path.join(self.img_dir, img_info["file_name"])).convert("RGB")
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
            "boxes":    torch.tensor(boxes,  dtype=torch.float32) if boxes  else torch.zeros((0, 4), dtype=torch.float32),
            "labels":   torch.tensor(labels, dtype=torch.long)    if labels else torch.zeros(0,      dtype=torch.long),
            "image_id": img_id,
        }
        return img_tensor, target


def collate_fn(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


class CachedFeaturesDataset(Dataset):
    def __init__(self, feat_dir: str, split: str = "train", max_samples=None):
        assert split in ("train", "val")
        self.split_dir = os.path.join(feat_dir, split)
        assert os.path.isdir(self.split_dir), \
            f"Feature directory not found: {self.split_dir}\nRun --mode precompute first."
        self.files = sorted(
            os.path.join(self.split_dir, f)
            for f in os.listdir(self.split_dir) if f.endswith(".pt")
        )
        if max_samples is not None:
            self.files = self.files[:max_samples]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=True)
        return data["feat"], data["target"]


def collate_fn_cached(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def get_coco_loaders(datadir, batch_size=32, num_workers=4):
    train_set = COCODetectionDataset(datadir, split="train")
    val_set   = COCODetectionDataset(datadir, split="val")
    return (
        DataLoader(train_set, batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, collate_fn=collate_fn),
        DataLoader(val_set,   batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn),
    )


def get_cached_loaders(feat_dir, batch_size=32, num_workers=4):
    train_set = CachedFeaturesDataset(feat_dir, split="train")
    val_set   = CachedFeaturesDataset(feat_dir, split="val")
    return (
        DataLoader(train_set, batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_cached),
        DataLoader(val_set,   batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_cached),
    )


# ── Precompute ────────────────────────────────────────────────────────────────

def precompute_features(model, datadir, feat_dir, device, batch_size=64, num_workers=4):
    model.eval()
    for split in ("train", "val"):
        out_dir = os.path.join(feat_dir, split)
        os.makedirs(out_dir, exist_ok=True)
        dataset = COCODetectionDataset(datadir, split=split)
        loader  = DataLoader(dataset, batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True, collate_fn=collate_fn)
        print(f"\n[Precompute] {split}: {len(dataset)} images → {out_dir}")
        with torch.no_grad():
            for images, targets in tqdm(loader, desc=f"Precompute {split}"):
                images = images.to(device)
                feats  = model.extract_features(images)
                for i, tgt in enumerate(targets):
                    save_path = os.path.join(out_dir, f"{tgt['image_id']}.pt")
                    if not os.path.exists(save_path):   # resume-safe
                        torch.save({"feat": feats[i].cpu(), "target": tgt}, save_path)
        print(f"[Precompute] {split} done.")


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device):
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


def evaluate(model, loader, device, datadir=None):
    if not COCO_EVAL_AVAILABLE:
        return 0.0, 0.0, 0.0

    model.eval()
    running_loss = 0.0
    coco_results = []

    coco_gt = getattr(loader.dataset, "coco", None)
    if coco_gt is None:
        if datadir:
            ann_file = os.path.join(datadir, "annotations", "instances_val2017.json")
        else:
            ann_file = os.path.normpath(os.path.join(
                os.path.dirname(loader.dataset.split_dir), "..", "..", "COCO",
                "annotations", "instances_val2017.json"))
        coco_gt = COCO(ann_file) if os.path.exists(ann_file) else None
        if coco_gt is None:
            print(f"Warning: annotation file not found at {ann_file}. mAP skipped.")

    with torch.no_grad():
        for batch, targets in tqdm(loader, desc="Evaluating", leave=False):
            batch = batch.to(device)
            preds = model(batch)
            running_loss += yolo_loss(preds, targets, model.anchors, device).item()

            if coco_gt is not None:
                # Low threshold: COCOeval sweeps confidence to build the full PR curve
                dets = decode_predictions(preds, model.anchors, conf_thresh=0.01)
                for det, tgt in zip(dets, targets):
                    img_id   = tgt["image_id"]
                    img_info = coco_gt.imgs[img_id]
                    sx = img_info["width"]  / IMG_SIZE   # convert 224px → original pixel space
                    sy = img_info["height"] / IMG_SIZE
                    for k in range(len(det["scores"])):
                        box = det["boxes"][k]
                        coco_results.append({
                            "image_id":    img_id,
                            "category_id": IDX_TO_CAT[det["labels"][k].item()],
                            "bbox": [box[0].item()*sx, box[1].item()*sy,
                                     (box[2]-box[0]).item()*sx, (box[3]-box[1]).item()*sy],
                            "score": det["scores"][k].item(),
                        })

    avg_loss        = running_loss / len(loader)
    map50, map50_95 = 0.0, 0.0
    if coco_gt is not None and coco_results:
        coco_dt   = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
        map50_95 = float(coco_eval.stats[0])
        map50    = float(coco_eval.stats[1])
    elif not coco_results:
        print("Warning: no detections above threshold — mAP is 0.")
    return avg_loss, map50, map50_95


# ── DetectionValidator (RL agent) ─────────────────────────────────────────────

class DetectionValidator:
    """
    Fast proxy evaluator for the RL pruning agent.
    evaluate(mask_1d) → (avg_loss, proxy_map)

    proxy_map = mean sigmoid objectness — cheap mAP surrogate.
    Works with both CachedFeaturesDataset and COCODetectionDataset proxy loaders.
    """

    def __init__(self, model, proxy_loader, device):
        self.model  = model
        self.device = device
        self.cached_data   = []   # list of (feat_or_tokens, targets) batches
        self._is_feat_cache = isinstance(proxy_loader.dataset, CachedFeaturesDataset)
        self._build_cache(proxy_loader)

    def _build_cache(self, proxy_loader):
        self.model.eval()
        print(f"Building Detection Proxy Cache for {len(proxy_loader.dataset)} samples...")
        with torch.no_grad():
            for batch, targets in proxy_loader:
                batch = batch.to(self.device)
                if self._is_feat_cache:
                    self.cached_data.append((batch, targets))
                else:
                    # Raw images: store pre-patch tokens; replay blocks on each evaluate()
                    tokens = self.model.transformer.prepare_tokens_with_masks(batch)
                    self.cached_data.append((tokens, targets))
        print("Detection Proxy Cache built.")

    def evaluate(self, mask_1d: torch.Tensor):
        self.model.eval()
        self.model.set_mask(mask_1d)
        running_loss = total_obj = 0.0

        with torch.no_grad():
            for cached, targets in self.cached_data:
                if self._is_feat_cache:
                    feat_map = cached
                else:
                    x = cached
                    for blk in self.model.transformer.blocks:
                        x = blk(x)
                    x = self.model.transformer.norm(x)
                    patches  = x[:, 5:, :]
                    B        = patches.size(0)
                    feat_map = patches.permute(0, 2, 1).reshape(B, FEAT_DIM, GRID_SIZE, GRID_SIZE)

                raw   = self.model.det_head(feat_map)
                B_    = feat_map.size(0)
                preds = raw.reshape(B_, NUM_ANCHORS, GRID_SIZE, GRID_SIZE, 5 + NUM_CLASSES)
                running_loss += yolo_loss(preds, targets, self.model.anchors, self.device).item()
                total_obj    += torch.sigmoid(preds[..., 4]).mean().item()

        n = len(self.cached_data)
        return running_loss / n, total_obj / n


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    model = DinoDetector(device=device).to(device)

    if args.mode == "precompute":
        assert args.datadir and args.feat_dir, "--datadir and --feat_dir required"
        precompute_features(model, args.datadir, args.feat_dir, device,
                            args.batch_size, args.num_workers)
        print("Precompute done. Run --mode train --feat_dir ...")
        return

    if args.feat_dir and os.path.isdir(args.feat_dir):
        train_loader, val_loader = get_cached_loaders(args.feat_dir, args.batch_size, args.num_workers)
    else:
        assert args.datadir, "Provide --feat_dir (fast) or --datadir (slow fallback)."
        train_loader, val_loader = get_coco_loaders(args.datadir, args.batch_size, args.num_workers)

    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Resumed: {args.checkpoint}")

    optimizer = optim.Adam(model.det_head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    train_losses, val_losses, map50_list = [], [], []
    best_val_loss = float("inf")
    best_map50    = 0.0
    no_improve    = 0

    for epoch in range(args.epochs):
        t_loss               = train_one_epoch(model, train_loader, optimizer, device)
        v_loss, map50, map50_95 = evaluate(model, val_loader, device, datadir=args.datadir)
        scheduler.step()

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        map50_list.append(map50)

        print(f"Epoch {epoch+1}/{args.epochs} | Train {t_loss:.4f} | Val {v_loss:.4f} | "
              f"mAP@0.5 {map50:.4f} | mAP@0.5:0.95 {map50_95:.4f}")

        torch.save({"epoch": epoch+1, "model_state_dict": model.state_dict(),
                    "map50": map50, "map50_95": map50_95},
                   os.path.join(args.checkpoint_dir, "dino_coco_latest.pth"))

        # Early stop on val loss — mAP is ~0 for first ~15 epochs
        if v_loss < best_val_loss - 0.01:
            best_val_loss = v_loss
            no_improve    = 0
            best_map50    = max(best_map50, map50)
            torch.save({"epoch": epoch+1, "model_state_dict": model.state_dict(),
                        "map50": map50, "map50_95": map50_95},
                       os.path.join(args.checkpoint_dir, "dino_coco_best.pth"))
            print(f"  Best val loss {best_val_loss:.4f} | mAP@0.5 {best_map50:.4f} saved.")
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"Plateau for {args.patience} epochs — early stop.")
            break

    os.makedirs("figure", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(train_losses) + 1)
    ax1.plot(ep, train_losses, label="Train"); ax1.plot(ep, val_losses, label="Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.set_title("YOLO Loss")
    ax2.plot(ep, map50_list, color="green"); ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mAP"); ax2.set_title("Val mAP@0.5")
    plt.tight_layout()
    plt.savefig("figure/trainval_curve_coco_v3.jpg")
    print("Curves saved → figure/trainval_curve_coco_v3.jpg")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="AttentionSurgeon — DINOv2 YOLOv3 detection on COCO")
    p.add_argument("--mode", choices=["precompute", "train"], default="train")
    p.add_argument("--datadir",        default=os.path.join(os.environ.get("BLACKHOLE", "."), "COCO"))
    p.add_argument("--feat_dir",       default=os.path.join(os.environ.get("BLACKHOLE", "."), "COCO_features"))
    p.add_argument("--checkpoint_dir", default="checkpoints/")
    p.add_argument("--checkpoint",     default=None)
    p.add_argument("--epochs",         type=int,   default=40)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--patience",       type=int,   default=15,
                   help="Early stopping patience on val loss.")
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--device",         default=None)
    main(p.parse_args())
