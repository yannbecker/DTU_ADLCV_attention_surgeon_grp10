"""
check_coco_data.py — Visual sanity check for COCO data pipelines.

Draws GT boxes on 6 random val images from each pipeline and saves a figure.
Also prints a statistics report: pixel range, box coordinate range, label range,
NaN/Inf counts, and any degenerate (zero-area) boxes.

Usage:
    python check_coco_data.py --datadir /path/to/COCO
    python check_coco_data.py --datadir $BLACKHOLE/COCO --n 8 --seed 42
"""

import argparse
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# ── Try to import pycocotools ─────────────────────────────────────────────────
try:
    from pycocotools.coco import COCO
except ImportError:
    sys.exit("pycocotools is required: pip install pycocotools")


# ── COCO category mapping (shared by all pipelines) ───────────────────────────

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
CAT_TO_IDX = {cat: idx for idx, cat in enumerate(COCO_CAT_IDS)}
IDX_TO_CAT = {idx: cat for cat, idx in CAT_TO_IDX.items()}

DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD  = [0.229, 0.224, 0.225]

# Unnormalise tensor that was normalised with DINO_MEAN/STD
_un_mean = torch.tensor(DINO_MEAN).view(3, 1, 1)
_un_std  = torch.tensor(DINO_STD ).view(3, 1, 1)

def unnormalise(t: torch.Tensor) -> np.ndarray:
    """(3,H,W) normalised float → (H,W,3) uint8 numpy for imshow."""
    t = t * _un_std + _un_mean
    t = t.clamp(0, 1)
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

def to_uint8(t: torch.Tensor) -> np.ndarray:
    """(3,H,W) [0,1] float → (H,W,3) uint8 numpy for imshow."""
    return (t.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_pipeline_224(datadir: str, img_ids: list[int], coco: COCO):
    """YOLOv3/v8 pipeline — 224×224, ImageNet-normalised, boxes normalised xyxy."""
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(DINO_MEAN, DINO_STD),
    ])
    img_dir = f"{datadir}/val2017"
    samples = []
    for img_id in img_ids:
        info = coco.imgs[img_id]
        img  = Image.open(f"{img_dir}/{info['file_name']}").convert("RGB")
        orig_w, orig_h = img.size
        tensor = transform(img)                      # (3, 224, 224)

        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        boxes, labels = [], []
        for ann in coco.loadAnns(ann_ids):
            if ann["category_id"] not in CAT_TO_IDX:
                continue
            x, y, w, h = ann["bbox"]
            x1, y1 = x / orig_w, y / orig_h
            x2, y2 = (x + w) / orig_w, (y + h) / orig_h
            x1, y1, x2, y2 = (max(0., min(1., v)) for v in (x1, y1, x2, y2))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(CAT_TO_IDX[ann["category_id"]])

        samples.append({
            "tensor": tensor,
            "boxes":  torch.tensor(boxes,  dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(labels, dtype=torch.long)    if labels else torch.zeros(0, dtype=torch.long),
            "img_id": img_id,
            "orig_size": (orig_w, orig_h),
        })
    return samples


def load_pipeline_518(datadir: str, img_ids: list[int], coco: COCO):
    """yolo_dinov2 pipeline — 518×518, [0,1] float (no normalisation), boxes normalised xyxy."""
    resize   = T.Resize((518, 518))
    to_tensor = T.ToTensor()
    img_dir = f"{datadir}/val2017"
    samples = []
    for img_id in img_ids:
        info = coco.imgs[img_id]
        img  = Image.open(f"{img_dir}/{info['file_name']}").convert("RGB")
        orig_w, orig_h = img.size
        tensor = to_tensor(resize(img))              # (3, 518, 518)

        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        boxes, labels = [], []
        for ann in coco.loadAnns(ann_ids):
            if ann["category_id"] not in CAT_TO_IDX:
                continue
            x, y, w, h = ann["bbox"]
            x1, y1 = x / orig_w, y / orig_h
            x2, y2 = (x + w) / orig_w, (y + h) / orig_h
            x1, y1, x2, y2 = (max(0., min(1., v)) for v in (x1, y1, x2, y2))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(CAT_TO_IDX[ann["category_id"]])

        samples.append({
            "tensor": tensor,
            "boxes":  torch.tensor(boxes,  dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(labels, dtype=torch.long)    if labels else torch.zeros(0, dtype=torch.long),
            "img_id": img_id,
            "orig_size": (orig_w, orig_h),
        })
    return samples


# ── Stats printer ─────────────────────────────────────────────────────────────

def print_stats(name: str, samples: list[dict], img_size: int) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}  —  {len(samples)} samples  ({img_size}×{img_size})")
    print(f"{'='*60}")

    all_boxes  = torch.cat([s["boxes"]  for s in samples if s["boxes"].numel()  > 0], dim=0) \
                 if any(s["boxes"].numel() > 0 for s in samples) else torch.zeros((0, 4))
    all_labels = torch.cat([s["labels"] for s in samples if s["labels"].numel() > 0], dim=0) \
                 if any(s["labels"].numel() > 0 for s in samples) else torch.zeros(0, dtype=torch.long)

    # Image pixel stats (unnormalise 224px, keep 518px as-is)
    for s in samples:
        t = s["tensor"]
        nan_count  = t.isnan().sum().item()
        inf_count  = t.isinf().sum().item()
        print(
            f"  img {s['img_id']:>7d} | orig {s['orig_size'][0]:4d}×{s['orig_size'][1]:4d} "
            f"| tensor {tuple(t.shape)} "
            f"| pixel [{t.min():.3f}, {t.max():.3f}] "
            f"| boxes {len(s['boxes']):3d} "
            f"| NaN {nan_count} Inf {inf_count}"
        )

    print(f"\n  --- Aggregate ---")
    if all_boxes.numel() > 0:
        print(f"  Total GT boxes   : {len(all_boxes)}")
        print(f"  Box x1 range     : [{all_boxes[:,0].min():.4f}, {all_boxes[:,0].max():.4f}]  (expected [0,1])")
        print(f"  Box y1 range     : [{all_boxes[:,1].min():.4f}, {all_boxes[:,1].max():.4f}]  (expected [0,1])")
        print(f"  Box x2 range     : [{all_boxes[:,2].min():.4f}, {all_boxes[:,2].max():.4f}]  (expected [0,1])")
        print(f"  Box y2 range     : [{all_boxes[:,3].min():.4f}, {all_boxes[:,3].max():.4f}]  (expected [0,1])")
        degenerate = ((all_boxes[:,2] <= all_boxes[:,0]) | (all_boxes[:,3] <= all_boxes[:,1])).sum()
        print(f"  Degenerate boxes : {degenerate.item()}  (expected 0)")
        oob = ((all_boxes < 0) | (all_boxes > 1)).any(dim=1).sum()
        print(f"  Out-of-range [0,1]: {oob.item()}  (expected 0)")
    else:
        print("  No boxes found in these samples!")

    if all_labels.numel() > 0:
        print(f"  Label range      : [{all_labels.min().item()}, {all_labels.max().item()}]  (expected [0, 79])")
        out_of_range_labels = ((all_labels < 0) | (all_labels > 79)).sum().item()
        print(f"  Out-of-range labels: {out_of_range_labels}  (expected 0)")
    print()


# ── Figure drawer ─────────────────────────────────────────────────────────────

def draw_row(axs, samples: list[dict], img_size: int, normalised: bool, coco: COCO, title: str) -> None:
    cat_names = {info["id"]: info["name"] for info in coco.cats.values()}
    colors    = plt.cm.tab20.colors                     # 20 distinct colours, cycled

    for ax, s in zip(axs, samples):
        t = s["tensor"]
        img_np = unnormalise(t) if normalised else to_uint8(t)
        ax.imshow(img_np)
        ax.set_title(f"id={s['img_id']}  orig={s['orig_size'][0]}×{s['orig_size'][1]}",
                     fontsize=7)
        ax.axis("off")

        for box, lbl in zip(s["boxes"].tolist(), s["labels"].tolist()):
            x1, y1, x2, y2 = box
            # Scale normalised coords → pixel space for this pipeline's img_size
            px1, py1 = x1 * img_size, y1 * img_size
            pw,  ph   = (x2 - x1) * img_size, (y2 - y1) * img_size

            cat_id   = IDX_TO_CAT.get(lbl, -1)
            cat_name = cat_names.get(cat_id, f"idx{lbl}")
            color    = colors[lbl % len(colors)]

            rect = mpatches.FancyBboxPatch(
                (px1, py1), pw, ph,
                linewidth=1.2, edgecolor=color, facecolor="none",
                boxstyle="square,pad=0",
            )
            ax.add_patch(rect)
            ax.text(px1, py1 - 2, cat_name, fontsize=5.5, color=color,
                    ha="left", va="bottom",
                    bbox=dict(facecolor="black", alpha=0.35, pad=1, linewidth=0))

    axs[0].set_ylabel(title, fontsize=9, rotation=90, labelpad=6)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Visual COCO data sanity check")
    p.add_argument("--datadir", required=True, help="COCO root (train2017/, val2017/, annotations/)")
    p.add_argument("--n",    type=int, default=6,  help="Number of images to show per pipeline")
    p.add_argument("--seed", type=int, default=0,  help="Random seed for image selection")
    p.add_argument("--out",  default="figure/coco_check.jpg", help="Output figure path")
    p.add_argument("--split", default="val", choices=["train", "val"])
    args = p.parse_args()

    print(f"Loading COCO {args.split} annotations …")
    ann_file = f"{args.datadir}/annotations/instances_{args.split}2017.json"
    coco = COCO(ann_file)

    # Pick n images that have at least 1 annotation
    rng = random.Random(args.seed)
    ids_with_anns = [
        img_id for img_id in coco.imgs
        if len(coco.getAnnIds(imgIds=img_id, iscrowd=False)) > 0
    ]
    chosen = rng.sample(ids_with_anns, min(args.n, len(ids_with_anns)))
    print(f"Selected {len(chosen)} images: {chosen}")

    print("\nLoading 224px pipeline …")
    samples_224 = load_pipeline_224(args.datadir, chosen, coco)
    print("Loading 518px pipeline …")
    samples_518 = load_pipeline_518(args.datadir, chosen, coco)

    print_stats("Pipeline 1 — YOLOv3/v8  (224×224, ImageNet-normalised)", samples_224, 224)
    print_stats("Pipeline 2 — yolo_dinov2 (518×518, [0,1] float)",          samples_518, 518)

    # ── Figure ────────────────────────────────────────────────────────────────
    n = len(chosen)
    fig, axes = plt.subplots(2, n, figsize=(3.0 * n, 7))
    if n == 1:
        axes = axes.reshape(2, 1)

    draw_row(axes[0], samples_224, 224, normalised=True,  coco=coco,
             title="224px\n(YOLOv3/v8)")
    draw_row(axes[1], samples_518, 518, normalised=False, coco=coco,
             title="518px\n(yolo_dinov2)")

    fig.suptitle(
        "COCO GT box alignment check  —  boxes should tightly wrap objects in both rows",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {args.out}")
    print("Open it and verify: boxes should tightly wrap the actual objects in BOTH rows.")


if __name__ == "__main__":
    main()
