# =============================================================================
# yolo_dinov2/dataset.py — AttentionSurgeon: COCO raw-image dataset for
#                          DINOv2 + YOLOv8 single-scale detection (518×518)
# =============================================================================
# Architecture notes:
#   - Images are loaded at native resolution, resized to 518×518 (DINOv2
#     native resolution: 518 / 14 = 37 exactly), and returned as float32
#     tensors in [0, 1].  ImageNet normalisation is applied inside the
#     backbone (DinoPrunableBackbone.normalize), so we do NOT normalise here.
#   - Bounding boxes are stored as normalised xyxy coordinates in [0, 1].
#   - Horizontal flip augmentation is applied at the image level so that
#     the backbone sees genuinely diverse spatial inputs each epoch.
# =============================================================================

from __future__ import annotations

import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch import FloatTensor, LongTensor, Tensor
from torch.utils.data import DataLoader, Dataset

# ── COCO category mapping ─────────────────────────────────────────────────────

IMG_SIZE = 518   # DINOv2 ViT-B/14 native resolution — 518 / 14 = 37 patches
NC       = 80    # COCO classes

COCO_CAT_IDS: list[int] = [
     1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]
CAT_TO_IDX: dict[int, int] = {cat: idx for idx, cat in enumerate(COCO_CAT_IDS)}
IDX_TO_CAT: dict[int, int] = {idx: cat for cat, idx in CAT_TO_IDX.items()}


# =============================================================================
# Dataset
# =============================================================================

class COCODetectionDatasetV2(Dataset):
    """
    COCO 2017 detection dataset that returns raw float32 images [0, 1] so
    that real augmentation can be applied and the backbone receives diverse
    spatial inputs every epoch.

    Directory layout expected::

        <root>/
            train2017/
            val2017/
            annotations/
                instances_train2017.json
                instances_val2017.json

    Each __getitem__ returns:
        img_tensor : FloatTensor (3, 518, 518)  — float32 in [0, 1]
        target     : dict with keys
                         "boxes"    FloatTensor (N, 4)  normalised xyxy
                         "labels"   LongTensor  (N,)    class indices 0–79
                         "image_id" int
    """

    def __init__(self, root: str, split: str = "train", augment: bool = True) -> None:
        assert split in ("train", "val"), f"split must be 'train' or 'val', got {split!r}"

        from pycocotools.coco import COCO  # local import — optional dependency

        self.img_dir = os.path.join(root, f"{split}2017")
        ann_file     = os.path.join(root, "annotations", f"instances_{split}2017.json")
        self.coco    = COCO(ann_file)
        self.img_ids = sorted(self.coco.imgs.keys())
        self.augment = augment

        # Resize to DINOv2 native resolution; ToTensor converts to [0, 1].
        # Both are instantiated once here rather than inside __getitem__ so that
        # no Python object construction overhead occurs per sample.
        self._resize    = T.Resize((IMG_SIZE, IMG_SIZE))
        self._to_tensor = T.ToTensor()

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict]:
        img_id   = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]

        # ── Load image ────────────────────────────────────────────────────────
        img       = Image.open(
            os.path.join(self.img_dir, img_info["file_name"])
        ).convert("RGB")
        orig_w, orig_h = img.size

        # ── Load annotations ──────────────────────────────────────────────────
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns    = self.coco.loadAnns(ann_ids)

        boxes_list:  list[list[float]] = []
        labels_list: list[int]         = []

        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in CAT_TO_IDX:
                continue
            if ann.get("area", 1) == 0:
                continue

            x, y, w, h = ann["bbox"]          # COCO XYWH in original pixel coords

            # Normalise to [0, 1] using ORIGINAL image dimensions
            x1 = x / orig_w
            y1 = y / orig_h
            x2 = (x + w) / orig_w
            y2 = (y + h) / orig_h

            # Clamp and skip degenerate boxes
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes_list.append([x1, y1, x2, y2])
            labels_list.append(CAT_TO_IDX[cat_id])

        if boxes_list:
            boxes: FloatTensor  = torch.tensor(boxes_list,  dtype=torch.float32)
            labels: LongTensor  = torch.tensor(labels_list, dtype=torch.long)
        else:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0,      dtype=torch.long)

        target: dict = {
            "boxes":    boxes,
            "labels":   labels,
            "image_id": img_id,
        }

        # ── Augmentation (training only) ──────────────────────────────────────
        if self.augment:
            # 1. Colour jitter on the PIL image (no bbox changes needed)
            if torch.rand(1).item() < 0.8:
                jitter = T.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05
                )
                img = jitter(img)

            # 2. Random scale jitter: resize to s×IMG_SIZE, then crop or pad
            #    so the final tensor is always IMG_SIZE×IMG_SIZE.
            #    s ∈ [0.6, 1.4] — wide enough to change apparent object size.
            s = 0.6 + 0.8 * torch.rand(1).item()          # uniform [0.6, 1.4]
            new_size = int(IMG_SIZE * s)
            img = img.resize((new_size, new_size), Image.BILINEAR)

            if new_size >= IMG_SIZE:
                # Random crop down to IMG_SIZE
                max_offset = new_size - IMG_SIZE
                x0 = int(torch.randint(0, max_offset + 1, (1,)).item())
                y0 = int(torch.randint(0, max_offset + 1, (1,)).item())
                img = img.crop((x0, y0, x0 + IMG_SIZE, y0 + IMG_SIZE))
                if boxes.numel() > 0:
                    bpx = boxes * new_size          # [0, new_size] xyxy
                    bpx[:, [0, 2]] -= x0
                    bpx[:, [1, 3]] -= y0
                    bpx = bpx.clamp(0.0, float(IMG_SIZE))
                    valid = (bpx[:, 2] > bpx[:, 0] + 1) & (bpx[:, 3] > bpx[:, 1] + 1)
                    boxes  = bpx[valid] / IMG_SIZE
                    labels = labels[valid]
            else:
                # Centre-pad to IMG_SIZE with grey (matches YOLOv8 letterbox convention)
                px = (IMG_SIZE - new_size) // 2
                py = (IMG_SIZE - new_size) // 2
                padded = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (114, 114, 114))
                padded.paste(img, (px, py))
                img = padded
                if boxes.numel() > 0:
                    bpx = boxes * new_size          # [0, new_size] xyxy
                    bpx[:, [0, 2]] += px
                    bpx[:, [1, 3]] += py
                    boxes = bpx / IMG_SIZE

            img_tensor = self._to_tensor(img)

            # 3. Random horizontal flip
            if torch.rand(1).item() < 0.5:
                img_tensor = img_tensor.flip(-1)
                if boxes.numel() > 0:
                    new_x1 = 1.0 - boxes[:, 2]
                    new_x2 = 1.0 - boxes[:, 0]
                    boxes  = torch.stack(
                        [new_x1, boxes[:, 1], new_x2, boxes[:, 3]], dim=1
                    )
        else:
            # Validation: deterministic resize only
            img = self._resize(img)
            img_tensor = self._to_tensor(img)

        target["boxes"]  = boxes
        target["labels"] = labels
        return img_tensor, target


# =============================================================================
# Collate function
# =============================================================================

def collate_fn_v2(batch: list[tuple[Tensor, dict]]) -> tuple[Tensor, list[dict]]:
    """
    Stack images into a single (B, 3, 518, 518) tensor; keep targets as a list
    of dicts so that each image can have a different number of boxes.
    """
    imgs    = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return imgs, targets


# =============================================================================
# Loader factory
# =============================================================================

def get_coco_loaders_v2(
    datadir:     str,
    batch_size:  int = 16,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and val DataLoaders for raw-image COCO detection.

    Args:
        datadir:     Root directory that contains train2017/, val2017/ and
                     annotations/.
        batch_size:  Images per batch.  Default 16 — lower than v8's 32
                     because 518×518 images consume more VRAM.
        num_workers: DataLoader worker processes.

    Returns:
        (train_loader, val_loader)
    """
    train_set = COCODetectionDatasetV2(datadir, split="train", augment=True)
    val_set   = COCODetectionDatasetV2(datadir, split="val",   augment=False)

    # persistent_workers=True keeps worker processes alive between epochs,
    # eliminating the ~5-10 s fork/import overhead at the start of each epoch.
    # prefetch_factor=4 keeps 4 batches queued per worker so the GPU never
    # waits for I/O.  Only set these when workers > 0 (avoids a PyTorch warning
    # when num_workers=0 / debugging).
    extra_kw = (
        dict(persistent_workers=True, prefetch_factor=4)
        if num_workers > 0 else {}
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_v2,
        **extra_kw,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_v2,
        **extra_kw,
    )
    return train_loader, val_loader
