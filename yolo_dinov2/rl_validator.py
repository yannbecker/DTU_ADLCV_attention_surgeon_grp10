# =============================================================================
# yolo_dinov2/rl_validator.py — AttentionSurgeon: fast RL proxy validator
# =============================================================================
# DinoPrunableValidator caches the pre-backbone token sequences (output of
# prepare_tokens_with_masks) for a fixed 500-image proxy set once at
# construction time.  During each RL step, only the 12 transformer blocks are
# replayed with the current pruning mask active — patch embedding and image
# normalisation are never repeated.
#
# Return signature:  evaluate(mask_1d) -> (avg_loss, proxy_map)
#   avg_loss  : always 0.0 (v8DetectionLoss requires training mode; we use the
#               proxy_map signal instead to avoid corrupting BatchNorm stats)
#   proxy_map : mean max-class sigmoid score across all anchors and all cached
#               batches — a fast surrogate for mAP.
#               values near 0   → model detects nothing (too many heads pruned)
#               values > 0.1    → model is producing meaningful detections
#
# This mirrors FastProxyValidator (rl_utils.py) and DetectionValidator
# (object_detection.py) but works with DinoPrunableDetector from
# yolo_dinov2/detector.py which uses the 518×518 / 37×37 token layout.
# =============================================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class DinoPrunableValidator:
    """
    Fast proxy validator for the RL pruning loop.

    Caches the token sequences that precede the transformer blocks for a fixed
    proxy set of images, then replays only the 12 blocks on each evaluate()
    call with the updated pruning mask.  This avoids repeating the expensive
    patch-embedding / positional-encoding step inside the RL loop.

    Parameters
    ----------
    model : DinoPrunableDetector
        The detector instance (backbone + head).  Must expose:
            model.backbone.extract_tokens(imgs)         -> (B, 1374, 768)
            model.backbone.forward_from_tokens(tokens)  -> (B, 768, 37, 37)
            model.head([feat_map])                      -> (y, x) in eval mode
            model.set_mask(mask_1d)
    proxy_loader : DataLoader
        Loader that yields (imgs, targets) batches.
        imgs    : (B, 3, 518, 518) float in [0, 1]
        targets : list[dict] — arbitrary target dicts (kept for future loss use)
    device : torch.device
        Compute device.
    criterion : optional
        v8DetectionLoss instance.  Reserved for future loss-based reward;
        currently unused because computing the loss requires model.train() which
        would corrupt BatchNorm running statistics during the RL loop.
    """

    def __init__(
        self,
        model,
        proxy_loader: DataLoader,
        device: torch.device,
        criterion=None,
    ) -> None:
        self.model     = model
        self.device    = device
        self.criterion = criterion   # optional: v8DetectionLoss for loss-based reward

        # Each element: (B, 1374, 768) pre-backbone token tensor stored on CPU
        # to preserve VRAM for the transformer replay and head forward passes.
        self.cached_tokens:  list[torch.Tensor] = []
        # Kept for future use with criterion (loss-based reward signal).
        self.cached_targets: list[list[dict]]   = []

        self._build_cache(proxy_loader)

    # ── Cache construction ────────────────────────────────────────────────────

    def _build_cache(self, loader: DataLoader) -> None:
        """
        Run through the proxy loader once and store the pre-block token
        sequences.  Only patch embedding + positional encoding are executed
        here; the 12 transformer blocks are not run.

        Parameters
        ----------
        loader : DataLoader
            Yields (imgs, targets) batches.
            imgs shape: (B, 3, 518, 518), float [0, 1].
        """
        self.model.eval()
        print(f"Building proxy cache for {len(loader.dataset)} samples...")

        with torch.no_grad():
            for imgs, targets in loader:
                imgs = imgs.to(self.device)    # already float [0, 1] from dataset

                # extract_tokens normalises and runs prepare_tokens_with_masks.
                # Result: (B, 1374, 768)  — CLS + 4 registers + 1369 patches.
                tokens = self.model.backbone.extract_tokens(imgs)

                # Store on CPU to save VRAM; moved to GPU inside evaluate().
                self.cached_tokens.append(tokens.cpu())
                self.cached_targets.append(targets)

        n_imgs = sum(t.shape[0] for t in self.cached_tokens)
        print(f"Cache built: {n_imgs} images")

    # ── RL evaluation ─────────────────────────────────────────────────────────

    def evaluate(self, mask_1d: torch.Tensor) -> tuple[float, float]:
        """
        Apply the current pruning mask and evaluate on the cached proxy set.

        Steps per batch
        ---------------
        1. Move cached tokens to device.
        2. Run forward_from_tokens (12 blocks + norm + reshape) — pruning hooks
           are active so pruned heads contribute zero to the output projection.
        3. Feed the (B, 768, 37, 37) feature map through the Detect head.
        4. Extract class scores from the head's eval-mode output tensor y of
           shape (B, 4+nc, N_anchors) — channels 4: are already sigmoid-ed.
        5. Accumulate mean max-class score as proxy_map.

        Parameters
        ----------
        mask_1d : torch.Tensor
            Shape (144,) — flat binary (or soft) mask from the RL agent.

        Returns
        -------
        (avg_loss, proxy_map) : tuple[float, float]
            avg_loss  : 0.0  (loss computation is skipped; see module docstring)
            proxy_map : mean max class confidence score — fast mAP surrogate.
        """
        self.model.eval()
        self.model.set_mask(mask_1d)

        total_proxy = 0.0
        total_loss  = 0.0
        num_batches = len(self.cached_tokens)

        with torch.no_grad():
            for tokens_cpu, targets in zip(self.cached_tokens, self.cached_targets):

                # ── 1. Move cached tokens to GPU ─────────────────────────────
                tokens = tokens_cpu.to(self.device)   # (B, 1374, 768)

                # ── 2. Replay transformer blocks with current pruning mask ────
                # forward_from_tokens runs blocks[0..11] + norm, then reshapes
                # patch tokens to (B, 768, 37, 37).
                feat_map = self.model.backbone.forward_from_tokens(tokens)

                # ── 3. Detect head forward ────────────────────────────────────
                # In eval mode the ultralytics Detect head returns a tuple
                # (y, x) where y is the decoded prediction tensor.
                # y shape: (B, 4 + nc, N_anchors)
                #   channels  0:4  — box coordinates (xyxy, pixel space)
                #   channels  4:   — class probabilities (sigmoid already applied)
                preds_raw = self.model.head([feat_map])

                # Guard: handle both tuple/list output (eval) and plain tensor
                # (training mode or custom wrappers).
                y = preds_raw[0] if isinstance(preds_raw, (tuple, list)) else preds_raw

                # ── 4. Proxy mAP: mean max class score ───────────────────────
                # class_scores: (B, nc, N_anchors) — sigmoid class probabilities
                class_scores = y[:, 4:, :]                # (B, nc, N_anchors)
                # max_scores:   (B, N_anchors) — best class confidence per anchor
                max_scores   = class_scores.max(dim=1).values
                # Scalar: mean confidence across all anchors and all images.
                proxy = max_scores.mean().item()
                total_proxy += proxy

                # ── 5. Loss (reserved / skipped) ─────────────────────────────
                # v8DetectionLoss requires model.train() to obtain the raw
                # feature tensors before Detect's internal decoding.  Switching
                # to train() here would update BatchNorm running statistics and
                # corrupt the backbone's learned normalisation.  We therefore
                # rely solely on proxy_map as the RL reward signal.
                #
                # If self.criterion is provided in a future iteration, the loss
                # can be computed by:
                #   self.model.train()
                #   preds_train = self.model.head([feat_map])
                #   batch_dict  = _build_batch_dict(targets, self.device)
                #   loss_raw, _ = self.criterion(preds_train, batch_dict)
                #   total_loss += loss_raw.sum().item()
                #   self.model.eval()
                # For now we keep total_loss = 0.0.
                total_loss += 0.0

        avg_proxy = total_proxy / max(num_batches, 1)
        avg_loss  = total_loss  / max(num_batches, 1)

        # Return signature matches DetectionValidator.evaluate exactly:
        # (avg_loss, proxy_map)  — proxy_map is the fast mAP surrogate.
        return avg_loss, avg_proxy
