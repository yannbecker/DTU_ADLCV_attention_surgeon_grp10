import os
import math
import types
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision.ops import batched_nms, box_iou
from tqdm.auto import tqdm
from PIL import Image

try:
    from ultralytics.nn.modules.head import Detect
    from ultralytics.nn.modules.block import C2f
    from ultralytics.utils.loss import v8DetectionLoss
except ImportError:
    raise ImportError("ultralytics is required: pip install ultralytics")

from classification import DinoClassifier

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    COCO_EVAL_AVAILABLE = True
except ImportError:
    COCO_EVAL_AVAILABLE = False
    print("Warning: pycocotools not found. mAP evaluation disabled.")

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_LAYERS  = 12
NUM_HEADS   = 12
HEAD_DIM    = 64
FEAT_DIM    = 768
GRID_SIZE   = 16
IMG_SIZE    = 224
NUM_CLASSES = 80

NECK_CHANNELS = (64, 128, 256)
# Strides for the three FPN levels; DINOv2 ViT-B/14 stride=14, two 2× upsamples give 7 and 3.5
STRIDES = torch.tensor([3.5, 7.0, 14.0])

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


# ── FPN Neck ──────────────────────────────────────────────────────────────────

class FPNNeck(nn.Module):
    """
    Top-down simple feature pyramid on the single DINOv2 (16×16) feature map.
    Outputs three scales for the anchor-free Detect head (ViTDet-style).

    Input : (B, 768, 16, 16)
    Output: [(B, 64, 64, 64), (B, 128, 32, 32), (B, 256, 16, 16)]
    """

    def __init__(self, in_ch=FEAT_DIM, out_chs=NECK_CHANNELS):
        super().__init__()
        c_small, c_med, c_large = out_chs
        self.lateral = nn.Sequential(
            nn.Conv2d(in_ch, c_large, 1, bias=False),
            nn.BatchNorm2d(c_large),
            nn.SiLU(inplace=True),
        )
        self.p2 = C2f(c_large, c_med,   n=1, shortcut=False)
        self.p1 = C2f(c_med,   c_small, n=1, shortcut=False)

    def forward(self, x):
        p3 = self.lateral(x)                                          # (B, 256, 16, 16)
        p2 = self.p2(F.interpolate(p3, scale_factor=2, mode="nearest"))  # (B, 128, 32, 32)
        p1 = self.p1(F.interpolate(p2, scale_factor=2, mode="nearest"))  # (B,  64, 64, 64)
        return [p1, p2, p3]   # ascending stride order — expected by Detect


# ── Model ─────────────────────────────────────────────────────────────────────

class DinoDetectorV8(DinoClassifier):
    """DINOv2 ViT-B/14 (frozen) + FPN neck + YOLOv8 anchor-free Detect head."""

    def __init__(self, device="cpu"):
        super().__init__(device=device, num_classes=1)
        del self.classifier

        self.neck = FPNNeck(in_ch=FEAT_DIM, out_chs=NECK_CHANNELS)
        self.head = Detect(nc=NUM_CLASSES, ch=NECK_CHANNELS)
        self.head.stride = STRIDES.clone()
        # bias_init uses 640px as reference; call _fix_cls_bias to correct for 224px
        self.head.bias_init()
        self._fix_cls_bias()
        self._init_neck_weights()

    def _fix_cls_bias(self):
        # ultralytics bias_init uses log(5/nc/(640/s)^2); correct for 224px input
        for i, s in enumerate(STRIDES.tolist()):
            cv3_last = self.head.cv3[i][-1]
            cv3_last.bias.data[:NUM_CLASSES] = math.log(5.0 / NUM_CLASSES / (IMG_SIZE / s) ** 2)

    def _init_neck_weights(self):
        for m in self.neck.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        # Token layout: [CLS | REG×4 | PATCH×256]; patches are indices 5:261
        x = self.transformer.prepare_tokens_with_masks(images)
        for blk in self.transformer.blocks:
            x = blk(x)
        x = self.transformer.norm(x)
        patches = x[:, 5:, :]
        B = patches.size(0)
        return patches.permute(0, 2, 1).reshape(B, FEAT_DIM, GRID_SIZE, GRID_SIZE)

    def forward(self, x: torch.Tensor):
        feat = self.extract_features(x) if x.shape[1] == 3 else x
        return self.head(self.neck(feat))

    def get_taylor_importance(self, images: torch.Tensor, targets: list) -> torch.Tensor:
        # Pass raw images — gradients must flow through backbone blocks.
        was_training = self.training
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
            preds     = self.forward(images)
            criterion = _build_criterion(self.head, images.device)
            loss, _   = criterion(preds, _targets_to_batch(targets, images.device))
            self.zero_grad()
            loss.sum().backward()

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


# ── Loss helpers ──────────────────────────────────────────────────────────────

def _build_criterion(detect_head, device):
    _dummy = nn.Parameter(torch.empty(0, device=device))
    adapter = types.SimpleNamespace(
        args=types.SimpleNamespace(box=7.5, cls=0.5, dfl=1.5),
        model=[detect_head],
        class_weights=None,
    )
    adapter.parameters = lambda: iter([_dummy])
    return v8DetectionLoss(adapter)


def _targets_to_batch(targets: list, device: torch.device) -> dict:
    # Convert list-of-dicts (normalised xyxy) to ultralytics batch dict (normalised xywh)
    batch_idx_list, cls_list, bboxes_list = [], [], []
    for i, tgt in enumerate(targets):
        boxes  = tgt["boxes"].to(device)
        labels = tgt["labels"].to(device)
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
        return {"batch_idx": torch.zeros(0, device=device),
                "cls":       torch.zeros(0, device=device),
                "bboxes":    torch.zeros((0, 4), device=device)}
    return {"batch_idx": torch.cat(batch_idx_list),
            "cls":       torch.cat(cls_list),
            "bboxes":    torch.cat(bboxes_list)}


# ── Feature augmentation ──────────────────────────────────────────────────────

def augment_features(feats: torch.Tensor, targets: list):
    # Random horizontal flip on cached (B, 768, H, W) feature maps (50% per image)
    aug_feats, aug_targets = [], []
    for i in range(feats.size(0)):
        feat = feats[i]
        tgt  = targets[i]
        if random.random() < 0.5:
            feat  = feat.flip(-1).contiguous()   # flip W; contiguous() needed for Conv2d
            boxes = tgt["boxes"].clone()
            if boxes.numel() > 0:
                x1_new = 1.0 - boxes[:, 2]
                x2_new = 1.0 - boxes[:, 0]
                boxes[:, 0] = x1_new
                boxes[:, 2] = x2_new
            tgt = {**tgt, "boxes": boxes}
        aug_feats.append(feat)
        aug_targets.append(tgt)
    return torch.stack(aug_feats), aug_targets


# ── Training loop ─────────────────────────────────────────────────────────────

def _log_grad_norms(model, epoch, step):
    groups = {"neck.lateral": model.neck.lateral, "neck.p2": model.neck.p2,
              "neck.p1": model.neck.p1, "head": model.head}
    lines  = [f"[grad] epoch {epoch:03d}  step {step:05d}"]
    for gname, module in groups.items():
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        if not grads:
            lines.append(f"  {gname:14s}  — no grad")
            continue
        norms   = torch.stack([g.norm() for g in grads])
        maxes   = torch.stack([g.abs().max() for g in grads])
        nan_inf = sum(1 for g in grads if torch.isnan(g).any() or torch.isinf(g).any())
        flag    = " !! NaN/Inf" if nan_inf else ""
        lines.append(f"  {gname:14s}  norm mean={norms.mean():.2e}  max={maxes.max():.2e}  n={len(grads)}{flag}")
    print("\n".join(lines), flush=True)


def train_one_epoch(model, criterion, loader, optimizer, device,
                    epoch=0, debug_grads=False, debug_every=100):
    model.train()
    totals = torch.zeros(4)   # [total, box, cls, dfl]

    for step, (batch_data, targets) in enumerate(tqdm(loader, desc="Train", leave=False)):
        batch_data = batch_data.to(device)
        batch_data, targets = augment_features(batch_data, targets)
        preds    = model(batch_data)
        loss_raw, items = criterion(preds, _targets_to_batch(targets, device))
        # v8DetectionLoss may return a 3-element vector or scalar depending on version
        loss = loss_raw.sum()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(model.neck.parameters()) + list(model.head.parameters()), max_norm=10.0)
        optimizer.step()

        totals[0] += loss.item()
        totals[1:] += items.detach().cpu()[:3]

        if debug_grads and step % debug_every == 0:
            _log_grad_norms(model, epoch, step)

    n = len(loader)
    return (totals[0]/n).item(), (totals[1]/n).item(), (totals[2]/n).item(), (totals[3]/n).item()


# ── Decode + NMS ──────────────────────────────────────────────────────────────

def _decode_and_nms(raw_out, conf_thres=0.01, iou_thres=0.45, max_det=300):
    # raw_out: (B, 4+nc, total_anchors) — channels 0:4 are xyxy in 224px space
    pred = raw_out[0] if isinstance(raw_out, (tuple, list)) else raw_out
    results = []
    for p in pred:
        boxes  = p[:4].T
        scores, cls_ids = p[4:].max(dim=0)
        mask = scores > conf_thres
        if not mask.any():
            results.append(torch.zeros((0, 6), device=pred.device))
            continue
        b, s, c = boxes[mask], scores[mask], cls_ids[mask]
        keep = batched_nms(b, s, c, iou_thres)[:max_det]
        results.append(torch.cat([b[keep], s[keep, None], c[keep, None].float()], dim=1))
    return results


# ── Validation loss (train-mode subset pass) ──────────────────────────────────

def _compute_val_loss(model, loader, criterion, device, n_batches=50):
    """
    Estimate val loss over the first n_batches batches.

    Runs in train mode so the Detect head returns raw tensors (needed by the
    loss).  BN stats are frozen to prevent corruption from val batches.
    """
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i, (batch_data, targets) in enumerate(loader):
            if i >= n_batches:
                break
            batch_data = batch_data.to(device)
            preds = model(batch_data)
            loss_raw, _ = criterion(preds, _targets_to_batch(targets, device))
            total += loss_raw.sum().item()
            count += 1
    model.eval()
    return total / count if count > 0 else 0.0


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, loader, device, datadir=None):
    if not COCO_EVAL_AVAILABLE:
        print("pycocotools not found — mAP skipped.")
        return 0.0, 0.0, 0.0

    model.eval()
    coco_results = []

    coco_gt = getattr(loader.dataset, "coco", None)
    if coco_gt is None and datadir:
        ann_file = os.path.join(datadir, "annotations", "instances_val2017.json")
        if os.path.exists(ann_file):
            coco_gt = COCO(ann_file)
        else:
            print(f"[evaluate] ERROR: annotation file not found at {ann_file}. Returning mAP=0.")
            return 0.0, 0.0, 0.0

    if coco_gt is None:
        print("[evaluate] ERROR: no COCO ground-truth available. Pass --datadir. Returning mAP=0.")
        return 0.0, 0.0, 0.0

    iou_sum:   float = 0.0
    iou_count: int   = 0

    with torch.no_grad():
        for batch_data, targets in tqdm(loader, desc="Eval", leave=False):
            batch_data = batch_data.to(device)
            out  = model(batch_data)
            # Lower threshold — more candidates → more accurate COCOeval PR curve
            dets = _decode_and_nms(out, conf_thres=0.001, iou_thres=0.45, max_det=300)
            for det, tgt in zip(dets, targets):
                # ── Mean IoU ────────────────────────────────────────────────
                gt_norm = tgt["boxes"]                          # (G, 4) normalised xyxy
                if gt_norm.shape[0] > 0:
                    gt_px = gt_norm * IMG_SIZE                  # scale to 224-px space
                    gt_px = gt_px.to(device)
                    if det.shape[0] > 0:
                        iou_mat = box_iou(gt_px, det[:, :4])   # (G, N)
                        iou_sum += iou_mat.max(dim=1).values.sum().item()
                    iou_count += gt_norm.shape[0]

                # ── COCOeval ─────────────────────────────────────────────────
                img_id   = tgt["image_id"]
                img_info = coco_gt.imgs[img_id]
                sx = img_info["width"]  / IMG_SIZE
                sy = img_info["height"] / IMG_SIZE
                for row in det.tolist():
                    x1, y1, x2, y2, score, cls_idx = row
                    coco_results.append({
                        "image_id":    img_id,
                        "category_id": IDX_TO_CAT[int(cls_idx)],
                        "bbox":  [x1*sx, y1*sy, (x2-x1)*sx, (y2-y1)*sy],
                        "score": score,
                    })

    mean_iou        = iou_sum / iou_count if iou_count > 0 else 0.0
    map50 = map50_95 = 0.0
    if coco_results:
        coco_dt   = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
        map50_95 = float(coco_eval.stats[0])
        map50    = float(coco_eval.stats[1])
    else:
        print("Warning: no detections above conf_thres=0.001 — mAP is 0.")
    return map50, map50_95, mean_iou


# ── Datasets (inlined — identical to yolov3 version) ─────────────────────────

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
                    if not os.path.exists(save_path):
                        torch.save({"feat": feats[i].cpu(), "target": tgt}, save_path)
        print(f"[Precompute] {split} done.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    model = DinoDetectorV8(device=device).to(device)
    model.head.stride = model.head.stride.to(device)   # not a buffer — move manually

    if args.mode == "precompute":
        assert args.datadir and args.feat_dir, "--datadir and --feat_dir required"
        precompute_features(model, args.datadir, args.feat_dir, device,
                            args.batch_size, args.num_workers)
        print("Done. Run --mode train --feat_dir ...")
        return

    if args.feat_dir and os.path.isdir(args.feat_dir):
        train_loader, val_loader = get_cached_loaders(args.feat_dir, args.batch_size, args.num_workers)
    else:
        assert args.datadir, "Provide --feat_dir or --datadir."
        train_loader, val_loader = get_coco_loaders(args.datadir, args.batch_size, args.num_workers)

    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Resumed: {args.checkpoint}")

    trainable = list(model.neck.parameters()) + list(model.head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=1e-5)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])
    criterion = _build_criterion(model.head, device)

    train_losses, map50_list = [], []
    best_map50 = 0.0
    # Patience counter only starts after burn_in to avoid early stop before confidence calibrates
    burn_in    = max(args.warmup_epochs, 15)
    no_improve = 0

    for epoch in range(args.epochs):
        t_loss, box_l, cls_l, dfl_l = train_one_epoch(
            model, criterion, train_loader, optimizer, device,
            epoch=epoch, debug_grads=args.debug_grads, debug_every=args.debug_every)
        map50, map50_95, mean_iou = evaluate(model, val_loader, device, datadir=args.datadir)
        val_loss = _compute_val_loss(model, val_loader, criterion, device, n_batches=50)
        scheduler.step()

        train_losses.append(t_loss)
        map50_list.append(map50)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1:02d}/{args.epochs} | LR {current_lr:.2e} | "
              f"Train {t_loss:.3f} (box {box_l:.3f} cls {cls_l:.3f} dfl {dfl_l:.3f}) | "
              f"Val loss {val_loss:.3f} | "
              f"mAP@0.5 {map50:.4f}  mAP@0.5:0.95 {map50_95:.4f} | MeanIoU {mean_iou:.4f}")

        torch.save({"epoch": epoch+1, "model_state_dict": model.state_dict(),
                    "map50": map50, "map50_95": map50_95, "val_loss": val_loss, "mean_iou": mean_iou},
                   os.path.join(args.checkpoint_dir, "dinov8_coco_latest.pth"))

        if map50 > best_map50 + 1e-4:
            best_map50 = map50
            no_improve  = 0
            torch.save({"epoch": epoch+1, "model_state_dict": model.state_dict(),
                        "map50": map50, "map50_95": map50_95, "val_loss": val_loss, "mean_iou": mean_iou},
                       os.path.join(args.checkpoint_dir, "dinov8_coco_best.pth"))
            print(f"  Best mAP@0.5: {best_map50:.4f} saved.")
        elif epoch >= burn_in:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"No improvement for {args.patience} epochs — early stop.")
            break

    os.makedirs("figure", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, len(train_losses) + 1)
    ax1.plot(ep, train_losses); ax1.set_title("Total Train Loss"); ax1.set_xlabel("Epoch")
    ax2.plot(ep, map50_list, color="green"); ax2.set_title("mAP@0.5"); ax2.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig("figure/trainval_curve_coco_v8.jpg")
    print("Curves saved → figure/trainval_curve_coco_v8.jpg")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="AttentionSurgeon — DINOv2 + YOLOv8 anchor-free detection on COCO")
    p.add_argument("--mode",           choices=["precompute", "train"], default="train")
    p.add_argument("--datadir",        default=os.path.join(os.environ.get("BLACKHOLE", "."), "COCO"))
    p.add_argument("--feat_dir",       default=os.path.join(os.environ.get("BLACKHOLE", "."), "COCO_features"))
    p.add_argument("--checkpoint_dir", default="checkpoints/")
    p.add_argument("--checkpoint",     default=None)
    p.add_argument("--epochs",         type=int,   default=60)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--warmup_epochs",  type=int,   default=5,
                   help="Linear LR warmup from lr/10.")
    p.add_argument("--patience",       type=int,   default=30,
                   help="Early-stop patience counted only after 15-epoch burn-in.")
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--device",         default=None)
    p.add_argument("--debug_grads",    action="store_true",
                   help="Print per-module gradient norms every --debug_every steps.")
    p.add_argument("--debug_every",    type=int,   default=100)
    main(p.parse_args())
