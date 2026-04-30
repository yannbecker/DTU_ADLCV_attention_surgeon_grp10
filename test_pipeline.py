# test_pipeline.py
# Run with: python test_pipeline.py
# Requires: torch, torchvision, Pillow, pycocotools, tqdm
# Does NOT require the full COCO dataset — downloads 3 images from the web.

import torch
import torch.nn as nn
import torchvision.transforms as T
import requests
from PIL import Image
from io import BytesIO
import math
from torchvision.ops import nms

# ── 0. Constants (same as object_detection.py) ──────────────────────────────
NUM_LAYERS, NUM_HEADS, HEAD_DIM = 12, 12, 64
TOTAL_HEADS = NUM_LAYERS * NUM_HEADS  # 144
FEAT_DIM, GRID_SIZE, IMG_SIZE = 768, 16, 224
NUM_CLASSES, NUM_ANCHORS = 80, 3

DINO_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

ANCHORS = torch.tensor([[0.28, 0.22], [0.38, 0.48], [0.90, 0.78]])

# ── 1. Minimal model (paste or import from object_detection.py) ──────────────
# We inline the minimum classes here so the test is self-contained.

def load_dino(device):
    print("  Loading DINOv2 ViT-B/14 from torch.hub …")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg') # With reg token 
    return model.to(device)

class MinimalDetector(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.transformer = load_dino(device)
        for p in self.transformer.parameters():
            p.requires_grad = False
        self.mask = torch.ones(NUM_LAYERS, NUM_HEADS, device=device)
        self.hooks = []
        self._attach_hooks()
        out_ch = NUM_ANCHORS * (5 + NUM_CLASSES)
        self.det_head = nn.Sequential(
            nn.Conv2d(FEAT_DIM, 512, 3, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 3, padding=1),      nn.BatchNorm2d(256), nn.LeakyReLU(0.1),
            nn.Conv2d(256, out_ch, 1),
        )
        self.register_buffer("anchors", ANCHORS.clone())

    def _attach_hooks(self):
        def make_hook(i):
            def fn(mod, inp, out):
                return out * self.mask[i].repeat_interleave(HEAD_DIM).to(out.device)
            return fn
        for h in self.hooks: h.remove()
        self.hooks = []
        for i in range(NUM_LAYERS):
            h = self.transformer.blocks[i].attn.register_forward_hook(make_hook(i))
            self.hooks.append(h)

    def set_mask(self, mask_1d):
        self.mask = mask_1d.view(NUM_LAYERS, NUM_HEADS).to(self.device)

    def _features(self, images):
        x = self.transformer.prepare_tokens_with_masks(images)
        for blk in self.transformer.blocks: x = blk(x)
        x = self.transformer.norm(x)
        patches = x[:, 5:, :]  # drop CLS + 4 register tokens
        B = patches.size(0)
        return patches.permute(0, 2, 1).view(B, FEAT_DIM, GRID_SIZE, GRID_SIZE)

    def forward(self, images):
        feat = self._features(images)
        raw  = self.det_head(feat)
        B, _, H, W = raw.shape
        return raw.view(B, NUM_ANCHORS, 5 + NUM_CLASSES, H, W).permute(0, 1, 3, 4, 2)


def yolo_loss_simple(preds, targets, anchors, device):
    B, A, H, W, _ = preds.shape
    C = preds.shape[-1] - 5
    obj = torch.sigmoid(preds[..., 4])
    tgt_obj  = torch.zeros(B, A, H, W, device=device)
    obj_mask = torch.zeros(B, A, H, W, device=device, dtype=torch.bool)
    tgt_xy   = torch.zeros(B, A, H, W, 2, device=device)
    tgt_wh   = torch.zeros(B, A, H, W, 2, device=device)
    tgt_cls  = torch.zeros(B, A, H, W, C, device=device)
    for b_idx, tgt in enumerate(targets):
        boxes, labels = tgt["boxes"].to(device), tgt["labels"].to(device)
        if boxes.numel() == 0: continue
        cx = (boxes[:,0]+boxes[:,2])/2; cy = (boxes[:,1]+boxes[:,3])/2
        bw =  boxes[:,2]-boxes[:,0];    bh =  boxes[:,3]-boxes[:,1]
        gi = (cx*W).long().clamp(0,W-1); gj = (cy*H).long().clamp(0,H-1)
        wh_gt = torch.stack([bw,bh],1).unsqueeze(1)
        wh_an = anchors.unsqueeze(0)
        inter = torch.min(wh_gt,wh_an).prod(-1)
        union = wh_gt.prod(-1)+wh_an.prod(-1)-inter
        best  = (inter/union.clamp(1e-6)).argmax(1)
        for n in range(len(boxes)):
            a,gi_,gj_ = best[n].item(), gi[n].item(), gj[n].item()
            tgt_obj[b_idx,a,gj_,gi_]=1.; obj_mask[b_idx,a,gj_,gi_]=True
            tgt_xy[b_idx,a,gj_,gi_] = torch.stack([cx[n]*W-gi[n].float(), cy[n]*H-gj[n].float()])
            tgt_wh[b_idx,a,gj_,gi_] = torch.stack([
                torch.log(bw[n]/anchors[a,0].clamp(1e-6)),
                torch.log(bh[n]/anchors[a,1].clamp(1e-6))])
            if labels[n]<C: tgt_cls[b_idx,a,gj_,gi_,labels[n]]=1.
    bce=nn.BCELoss(); mse=nn.MSELoss(); ce=nn.BCEWithLogitsLoss()
    lo = bce(obj, tgt_obj)
    lxy= mse(torch.stack([torch.sigmoid(preds[...,0]),torch.sigmoid(preds[...,1])],-1)[obj_mask],
             tgt_xy[obj_mask]) if obj_mask.any() else torch.tensor(0.,device=device)
    lwh= mse(torch.stack([preds[...,2],preds[...,3]],-1)[obj_mask],
             tgt_wh[obj_mask]) if obj_mask.any() else torch.tensor(0.,device=device)
    lc = ce(preds[...,5:][obj_mask], tgt_cls[obj_mask]) if obj_mask.any() else torch.tensor(0.,device=device)
    return 5*lo + 5*(lxy+lwh) + lc


def decode(raw, anchors, conf=0.05, nms_thr=0.45):
    B,A,H,W,_ = raw.shape
    cw=IMG_SIZE/W; ch=IMG_SIZE/H; results=[]
    for b in range(B):
        boxes,scores,labels=[],[],[]
        for a in range(A):
            for gj in range(H):
                for gi in range(W):
                    p=raw[b,a,gj,gi]
                    ob=torch.sigmoid(p[4]).item()
                    if ob<conf: continue
                    sc,lb=torch.softmax(p[5:],0).max(0)
                    s=(sc*ob).item()
                    if s<conf: continue
                    cx=(gi+torch.sigmoid(p[0]).item())*cw
                    cy=(gj+torch.sigmoid(p[1]).item())*ch
                    bw=anchors[a,0].item()*IMG_SIZE*math.exp(max(-10,min(10,p[2].item())))
                    bh=anchors[a,1].item()*IMG_SIZE*math.exp(max(-10,min(10,p[3].item())))
                    boxes.append([cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2])
                    scores.append(s); labels.append(lb.item())
        if boxes:
            bx=torch.tensor(boxes); sc=torch.tensor(scores); lb=torch.tensor(labels)
            keep=nms(bx,sc,nms_thr)
            results.append({"boxes":bx[keep],"scores":sc[keep],"labels":lb[keep]})
        else:
            results.append({"boxes":torch.zeros(0,4),"scores":torch.zeros(0),"labels":torch.zeros(0,dtype=torch.long)})
    return results


# ── 2. Dummy COCO-style targets (3 fake boxes — no real annotation needed) ──
DUMMY_TARGETS = [
    {"boxes": torch.tensor([[0.1, 0.1, 0.5, 0.5]]), "labels": torch.tensor([0]),  "image_id": 1},
    {"boxes": torch.tensor([[0.3, 0.2, 0.8, 0.7]]), "labels": torch.tensor([2]),  "image_id": 2},
    {"boxes": torch.tensor([[0.0, 0.0, 0.4, 0.3],
                             [0.6, 0.6, 1.0, 1.0]]),
     "labels": torch.tensor([15, 56]), "image_id": 3},
]

# ── 3. Download 3 real COCO val images by their public URLs ─────────────────
COCO_URLS = [
    "http://images.cocodataset.org/val2017/000000000139.jpg",
    "http://images.cocodataset.org/val2017/000000000285.jpg",
    "http://images.cocodataset.org/val2017/000000000632.jpg",
]

def download_images(urls):
    imgs = []
    for url in urls:
        print(f"  Downloading {url.split('/')[-1]} …")
        r = requests.get(url, timeout=30)
        img = Image.open(BytesIO(r.content)).convert("RGB")
        imgs.append(DINO_TRANSFORM(img))
    return torch.stack(imgs)   # (3, 3, 224, 224)


# ── 4. Run all tests ─────────────────────────────────────────────────────────
def run_tests():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  AttentionSurgeon — Detection Pipeline Smoke Test")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # ── Test 1: Download images ──────────────────────────────────────────────
    print("[1/7] Downloading 3 COCO val images …")
    images = download_images(COCO_URLS).to(device)
    print(f"  ✓  images tensor shape: {tuple(images.shape)}")
    assert images.shape == (3, 3, 224, 224), "Image shape mismatch!"

    # ── Test 2: Build model ──────────────────────────────────────────────────
    print("\n[2/7] Building DinoDetector …")
    model = MinimalDetector(device).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    head_params  = sum(p.numel() for p in model.det_head.parameters())
    print(f"  ✓  Total params : {total_params:,}")
    print(f"  ✓  Head params  : {head_params:,}  (only these are trained)")

    # ── Test 3: Feature extraction ───────────────────────────────────────────
    print("\n[3/7] Testing DINOv2 feature extraction …")
    model.eval()
    with torch.no_grad():
        feat = model._features(images)
    print(f"  ✓  Feature map shape: {tuple(feat.shape)}  (expected (3, 768, 16, 16))")
    assert feat.shape == (3, 768, 16, 16)

    # ── Test 4: Forward pass ─────────────────────────────────────────────────
    print("\n[4/7] Testing full forward pass …")
    with torch.no_grad():
        preds = model(images)
    print(f"  ✓  Predictions shape: {tuple(preds.shape)}  (expected (3, 3, 16, 16, 85))")
    assert preds.shape == (3, NUM_ANCHORS, GRID_SIZE, GRID_SIZE, 5 + NUM_CLASSES)

    # ── Test 5: Pruning mask ─────────────────────────────────────────────────
    print("\n[5/7] Testing pruning mask (set_mask) …")
    with torch.no_grad():
        feat_full = model._features(images).clone()
        mask = torch.ones(TOTAL_HEADS, device=device)
        mask[0] = 0.0   # prune head 0 of layer 0
        model.set_mask(mask)
        feat_pruned = model._features(images)
    diff = (feat_full - feat_pruned).abs().max().item()
    print(f"  ✓  Max feature diff after pruning head 0: {diff:.6f}  (should be > 0)")
    assert diff > 0, "Pruning had no effect — hook may be broken!"
    model.set_mask(torch.ones(TOTAL_HEADS, device=device))  # restore

    # ── Test 6: Loss computation ─────────────────────────────────────────────
    print("\n[6/7] Testing YOLO loss …")
    model.train()
    preds = model(images)
    loss  = yolo_loss_simple(preds, DUMMY_TARGETS, model.anchors, device)
    print(f"  ✓  Loss value: {loss.item():.4f}  (should be finite and > 0)")
    assert torch.isfinite(loss) and loss.item() > 0

    # One backward pass
    loss.backward()
    grad_norm = sum(
        p.grad.norm().item() for p in model.det_head.parameters() if p.grad is not None
    )
    print(f"  ✓  Gradient norm (det_head only): {grad_norm:.4f}  (should be > 0)")
    assert grad_norm > 0, "No gradients flowing to detection head!"

    # ── Test 7: Decode + NMS ─────────────────────────────────────────────────
    print("\n[7/7] Testing decode + NMS …")
    model.eval()
    with torch.no_grad():
        preds = model(images)
    dets = decode(preds, model.anchors, conf=0.05)
    for i, d in enumerate(dets):
        n = len(d["scores"])
        print(f"  Image {i+1}: {n} detection(s) after NMS")
        if n > 0:
            print(f"    Best score: {d['scores'].max().item():.4f}  "
                  f"  Label: {d['labels'][d['scores'].argmax()].item()}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  ALL TESTS PASSED ✓")
    print("  The full pipeline is functional:")
    print("  DINOv2 backbone → patch feature map → YOLO head")
    print("  → loss → gradients → decode → NMS")
    print("  Pruning mask works correctly.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_tests()