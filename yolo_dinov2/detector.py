# =============================================================================
# yolo_dinov2/detector.py — AttentionSurgeon: single-scale YOLOv8 detector
# =============================================================================
# Architecture:
#   DinoPrunableBackbone  (frozen DINOv2 ViT-B/14-reg, stride 14, 37×37 grid)
#        │
#        │  (B, 768, 37, 37)
#        ▼
#   ultralytics Detect head  (nc=80, single scale, anchor-free DFL regression)
#        ▼
#   v8DetectionLoss  (TaskAlignedAssigner + CIoU + DFL + BCE)
#
# Design choices:
#   - Single-scale: stride=14, grid=37×37, no FPN needed.
#   - 518×518 input ensures 518/14=37 exactly (no fractional patches).
#   - Taylor importance is computed via the v8 loss signal, mirroring the
#     approach in classification.py but using detection gradients.
#   - forward() accepts EITHER raw images (channel dim == 3) OR pre-computed
#     (B, 768, H, W) feature maps so the RL validator can use cached features.
# =============================================================================

import math
import types
import torch
import torch.nn as nn
from types import SimpleNamespace

from ultralytics.nn.modules.head import Detect
from ultralytics.utils.loss import v8DetectionLoss

from .backbone import DinoPrunableBackbone


# ── Module-level constants ────────────────────────────────────────────────────

STRIDE: float = 14.0   # DINOv2 ViT-B/14 patch stride (pixels)
NC: int = 80            # COCO class count


# =============================================================================
# Helper: convert list-of-dicts targets → ultralytics batch dict
# =============================================================================

def _build_batch_dict(
    targets: list[dict],
    device: torch.device,
) -> dict:
    """
    Convert a list of per-image target dicts into the flat batch dict expected
    by ``v8DetectionLoss``.

    Parameters
    ----------
    targets : list[dict]
        Each dict has:
            ``"boxes"``  : (N, 4) float tensor, normalised xyxy in [0, 1].
            ``"labels"`` : (N,)   long tensor, class indices 0–79.
        Images with no annotations are silently skipped.

    device : torch.device

    Returns
    -------
    dict with keys:
        ``"batch_idx"`` : (total_N,)    float — which image each box belongs to.
        ``"cls"``       : (total_N,)    float — class index for each box.
        ``"bboxes"``    : (total_N, 4)  float — normalised xywh.
    """
    batch_idx_parts: list[torch.Tensor] = []
    cls_parts:       list[torch.Tensor] = []
    bboxes_parts:    list[torch.Tensor] = []

    for img_idx, tgt in enumerate(targets):
        boxes  = tgt["boxes"].to(device)    # (N, 4) normalised xyxy
        labels = tgt["labels"].to(device)   # (N,)   long
        n = labels.numel()
        if n == 0:
            continue

        # xyxy → xywh (centre-format, all coordinates normalised [0, 1])
        cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
        cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
        bw =  boxes[:, 2] - boxes[:, 0]
        bh =  boxes[:, 3] - boxes[:, 1]
        xywh = torch.stack([cx, cy, bw, bh], dim=1)   # (N, 4)

        batch_idx_parts.append(
            torch.full((n,), img_idx, dtype=torch.float32, device=device)
        )
        cls_parts.append(labels.float())
        bboxes_parts.append(xywh)

    if not batch_idx_parts:
        # All images in the batch had empty annotations
        return {
            "batch_idx": torch.zeros(0, device=device),
            "cls":       torch.zeros(0, device=device),
            "bboxes":    torch.zeros((0, 4), device=device),
        }

    return {
        "batch_idx": torch.cat(batch_idx_parts),
        "cls":       torch.cat(cls_parts),
        "bboxes":    torch.cat(bboxes_parts),
    }


# =============================================================================
# Helper: build v8DetectionLoss without a full ultralytics Trainer/model
# =============================================================================

def build_criterion(head: Detect, device: torch.device) -> v8DetectionLoss:
    """
    Instantiate ``v8DetectionLoss`` with a minimal adapter namespace so it can
    be used outside of the ultralytics training framework.

    ``v8DetectionLoss.__init__`` inspects the model object for:
        * ``next(model.parameters()).device``  → device of first parameter
        * ``model.args.{box, cls, dfl}``       → loss weight hyperparameters
        * ``model.model[-1]``                  → the Detect head itself
        * ``model.class_weights``              → optional per-class weights

    Parameters
    ----------
    head : ultralytics.nn.modules.head.Detect
        The fully-configured Detect head (stride and biases already set).
    device : torch.device
        Target compute device.

    Returns
    -------
    v8DetectionLoss
        Ready-to-call loss function.
    """
    _dummy_param = nn.Parameter(torch.empty(0, device=device))

    adapter = types.SimpleNamespace(
        args=SimpleNamespace(box=7.5, cls=0.5, dfl=1.5),
        model=[head],          # model[-1] is the Detect head
        class_weights=None,
    )
    # v8DetectionLoss calls next(model.parameters()) to find the device
    adapter.parameters = lambda: iter([_dummy_param])

    return v8DetectionLoss(adapter)


# =============================================================================
# Main detector class
# =============================================================================

class DinoPrunableDetector(nn.Module):
    """
    Single-scale YOLOv8 detector built on top of ``DinoPrunableBackbone``.

    The backbone (DINOv2 ViT-B/14-reg) is frozen; only the Detect head is
    trainable.  Pruning is managed entirely by the backbone's hook mechanism.

    Parameters
    ----------
    nc : int
        Number of object classes (default 80 for COCO).
    device : str
        Compute device string passed to the backbone constructor.

    Input / Output shapes
    ----------------------
    ``forward(images)``   : (B, 3, 518, 518) float [0,1] → head output
    ``forward(features)`` : (B, 768, H, W)  cached      → head output

    In training mode the Detect head returns a list of raw feature tensors
    suitable for ``v8DetectionLoss``.  In eval mode it returns decoded boxes.
    """

    def __init__(self, nc: int = NC, device: str = "cpu") -> None:
        super().__init__()

        self.backbone = DinoPrunableBackbone(device=device)

        embed: int = self.backbone.EMBED_DIM  # 768

        # Anchor-free detection head: single feature scale at stride 14
        self.head = Detect(nc=nc, ch=(embed,))
        self.head.stride = torch.tensor([STRIDE])
        self.head.bias_init()   # initialise detection biases (uses 640px reference)
        # Correct class bias for our 518px input: log(5/nc/(IMG_SIZE/stride)^2)
        # bias_init() uses (640/stride)^2 = 2090; correct value is (518/14)^2 = 1369
        _correct_cls_bias = math.log(5.0 / nc / (518.0 / STRIDE) ** 2)
        for seq in self.head.cv3:
            seq[-1].bias.data[:nc] = _correct_cls_bias

        self.nc = nc

    # ── Pruning interface (delegates to backbone) ─────────────────────────────

    def set_mask(self, mask_1d: torch.Tensor) -> None:
        """Update the pruning mask; re-registers backbone hooks automatically."""
        self.backbone.set_mask(mask_1d)

    def _register_pruning_hooks(self) -> None:
        """Re-register pruning hooks on the backbone (convenience wrapper)."""
        self.backbone._register_pruning_hooks()

    def get_intra_layer_ranks(self, importance: torch.Tensor) -> torch.Tensor:
        """
        Compute normalised within-layer ranks for Taylor importance scores.

        Parameters
        ----------
        importance : torch.Tensor
            Shape (12, 12).

        Returns
        -------
        torch.Tensor
            Shape (12, 12), values in [0, 1].
        """
        return self.backbone.get_intra_layer_ranks(importance)

    # ── Feature extraction ────────────────────────────────────────────────────

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Run the full backbone forward pass to obtain spatial feature maps.

        Used both during the precompute/caching step and when computing Taylor
        importance (where gradients must flow through the backbone blocks).

        Parameters
        ----------
        images : torch.Tensor
            Shape (B, 3, 518, 518), float in [0, 1].

        Returns
        -------
        torch.Tensor
            Shape (B, 768, 37, 37).
        """
        return self.backbone.forward(images)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        Detection forward pass.

        Accepts two input formats:
            * Raw images (B, 3, H, W)        → backbone → head
            * Cached features (B, 768, H, W) → head only (fast RL path)

        The channel dimension disambiguates: 3 = raw image, else = features.

        Parameters
        ----------
        x : torch.Tensor
            Either (B, 3, 518, 518) images or (B, 768, 37, 37) feature maps.

        Returns
        -------
        list[torch.Tensor] (training mode) or tuple (eval mode)
            Direct output of the ultralytics Detect head.
        """
        if x.shape[1] == 3:
            feat = self.backbone.forward(x)   # image path
        else:
            feat = x                          # pre-cached feature path

        # Detect head expects a list of feature tensors (one per scale)
        return self.head([feat])

    # ── Taylor importance ─────────────────────────────────────────────────────

    def get_taylor_importance(
        self,
        images: torch.Tensor,
        targets: list[dict],
        criterion: v8DetectionLoss,
    ) -> torch.Tensor:
        """
        Compute per-head Taylor importance scores using the v8 detection loss.

        The method:
            1. Attaches temporary forward hooks on each block's ``attn`` module
               to capture attention output activations and their gradients.
            2. Runs a forward + backward pass through the detector.
            3. Computes |activation × gradient| per head and averages over
               batch and token dimensions.
            4. Removes temporary hooks before returning.

        Parameters
        ----------
        images : torch.Tensor
            Shape (B, 3, 518, 518), float in [0, 1].  Must be raw images so
            gradients can flow through the backbone blocks.
        targets : list[dict]
            Per-image annotation dicts with ``"boxes"`` (xyxy, normalised) and
            ``"labels"`` (long) keys.
        criterion : v8DetectionLoss
            Pre-built loss function (see ``build_criterion``).

        Returns
        -------
        torch.Tensor
            Shape (12, 12) — Taylor importance score for each (layer, head).
        """
        num_layers = self.backbone.NUM_LAYERS
        num_heads  = self.backbone.NUM_HEADS
        head_dim   = self.backbone.HEAD_DIM

        scores: torch.Tensor = torch.zeros(
            num_layers, num_heads, device=images.device
        )

        activations: dict[str, torch.Tensor] = {}
        grads:       dict[str, torch.Tensor] = {}

        def _make_save_hook(name: str):
            def hook(
                module: nn.Module,
                inp: tuple,
                output: torch.Tensor,
            ) -> None:
                # Force grad tracking even for frozen-weight layers
                output.requires_grad_(True)
                activations[name] = output.detach()
                output.register_hook(lambda g: grads.update({name: g}))
            return hook

        # Attach temporary hooks on attn modules (NOT proj) to get head outputs
        temp_hooks = []
        for i in range(num_layers):
            h = self.backbone.vit.blocks[i].attn.register_forward_hook(
                _make_save_hook(f"layer_{i}")
            )
            temp_hooks.append(h)

        # Detect head must be in training mode: in eval mode it returns decoded
        # boxes, but v8DetectionLoss needs the raw pre-decode feature tensors.
        was_training = self.training
        self.train()

        with torch.set_grad_enabled(True):
            # Forward pass — must go through raw images for gradient flow
            preds = self.forward(images)

            # Convert targets to ultralytics batch format
            batch_dict = _build_batch_dict(targets, images.device)

            # Compute detection loss and backpropagate
            loss_raw, _ = criterion(preds, batch_dict)
            loss = loss_raw.sum()

            self.zero_grad()   # zero BEFORE backward so no stale .grad accumulation
            loss.backward()

        if not was_training:
            self.eval()

        # Accumulate per-head Taylor scores: |act × grad|
        for i in range(num_layers):
            key = f"layer_{i}"
            if key not in grads:
                continue

            act  = activations[key]   # (B, N, 768)
            grad = grads[key]         # (B, N, 768)

            # Reshape last dim into (num_heads, head_dim) to get per-head scores
            act  = act.reshape(*act.shape[:2],  num_heads, head_dim)
            grad = grad.reshape(*grad.shape[:2], num_heads, head_dim)

            # Sum over head_dim, average over batch and token dimensions
            scores[i] = (act * grad).abs().sum(-1).mean((0, 1))

        # Remove temporary hooks — do not leave them attached
        for h in temp_hooks:
            h.remove()

        return scores
