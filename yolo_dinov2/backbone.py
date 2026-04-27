# =============================================================================
# yolo_dinov2/backbone.py — AttentionSurgeon: prunable DINOv2 backbone
# =============================================================================
# DINOv2 ViT-B/14 with register tokens (dinov2_vitb14_reg).
#
# Token layout (dinov2_vitb14_reg, 518×518 input):
#   index 0    : CLS token
#   index 1–4  : 4 register tokens
#   index 5–1373: 37×37 = 1369 patch tokens  ← used as spatial features
#   total      : 1 + 4 + 1369 = 1374 tokens
#
# Pruning mechanism:
#   Pre-hooks on blocks[i].attn.proj zero out entire head columns (64 dims
#   each) BEFORE the output projection mixes them — faithful simulation of
#   removing the head entirely without touching model weights.
# =============================================================================

import torch
import torch.nn as nn


class DinoPrunableBackbone(nn.Module):
    """
    Frozen DINOv2 ViT-B/14-reg backbone with per-head pruning hooks.

    The backbone is loaded once and all parameters are frozen.
    Pruning is applied via forward pre-hooks on each block's attn.proj layer:
    the 768-dim concatenated head output is zero-masked before projection,
    effectively removing the contribution of pruned heads.

    Attributes
    ----------
    IMG_SIZE  : int   Native DINOv2 input resolution (518 = 37 × 14 px).
    GRID_SIZE : int   Spatial grid of patch tokens (37 × 37).
    EMBED_DIM : int   Token embedding dimension (768).
    NUM_LAYERS: int   Number of transformer blocks (12).
    NUM_HEADS : int   Attention heads per block (12).
    HEAD_DIM  : int   Dimension per head (64 = 768 / 12).
    """

    # ── Class-level constants ─────────────────────────────────────────────────
    IMG_SIZE   = 518   # 518 / 14 = 37 exactly (DINOv2 native resolution)
    GRID_SIZE  = 37    # 37 × 37 patch grid
    EMBED_DIM  = 768
    NUM_LAYERS = 12
    NUM_HEADS  = 12
    HEAD_DIM   = 64    # 768 / 12

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(self, device: str = "cpu") -> None:
        super().__init__()

        # Load pretrained DINOv2 ViT-B/14 with register tokens
        self.vit = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14_reg",
            pretrained=True,
            force_reload=False,
        )

        # Freeze all backbone parameters — only pruning hooks affect output
        for p in self.vit.parameters():
            p.requires_grad = False

        # ImageNet normalisation statistics (input images arrive as float [0,1])
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

        # Pruning mask: 1 = keep head, 0 = prune head
        self.mask: torch.Tensor = torch.ones(
            self.NUM_LAYERS, self.NUM_HEADS, device=device
        )

        # Hook handles — stored so they can be removed and re-registered
        self.hooks: list = []
        self._register_pruning_hooks()

    # ── Pruning hooks ─────────────────────────────────────────────────────────

    def _register_pruning_hooks(self) -> None:
        """
        Attach forward pre-hooks on each block's attn.proj layer.

        The pre-hook receives inp[0] of shape (B, N, 768) — the concatenated
        head outputs before the linear projection mixes them.  It zeros out
        the 64-dimensional slice belonging to each pruned head, then returns
        the masked tensor so the projection sees a zeroed contribution.

        Existing hooks are removed before re-registering to avoid duplicates.
        """
        for h in self.hooks:
            h.remove()
        self.hooks = []

        def _make_hook(layer_idx: int):
            def pre_hook(module: nn.Module, inp: tuple) -> tuple:
                # inp[0]: (B, N, 768) — concatenated head outputs
                mask_row: torch.Tensor = self.mask[layer_idx]       # (12,)
                full_mask = mask_row.repeat_interleave(self.HEAD_DIM)  # (768,)
                return (inp[0] * full_mask.to(inp[0].device),)
            return pre_hook

        for i in range(self.NUM_LAYERS):
            proj_layer = self.vit.blocks[i].attn.proj
            handle = proj_layer.register_forward_pre_hook(_make_hook(i))
            self.hooks.append(handle)

    def set_mask(self, mask_1d: torch.Tensor) -> None:
        """
        Update the pruning mask and re-register hooks with the new values.

        Parameters
        ----------
        mask_1d : torch.Tensor
            Shape (144,) — flat binary (or soft) mask provided by the RL agent.
            Reshaped internally to (NUM_LAYERS, NUM_HEADS) = (12, 12).
        """
        self.mask = mask_1d.view(self.NUM_LAYERS, self.NUM_HEADS)
        self._register_pruning_hooks()

    # ── Image normalisation ───────────────────────────────────────────────────

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ImageNet mean/std normalisation to float [0, 1] images.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, 3, H, W), values in [0, 1].

        Returns
        -------
        torch.Tensor
            Normalised tensor of the same shape.
        """
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    # ── Token extraction (called ONCE per image for caching) ─────────────────

    def extract_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """
        Normalise images and produce the full token sequence via
        prepare_tokens_with_masks (patch embedding + positional encoding +
        CLS / register tokens).  No transformer blocks are run here.

        This is the caching entry-point for the RL validator: call this once
        per image, store the result, then replay blocks with set_mask() +
        forward_from_tokens() for each candidate mask.

        Parameters
        ----------
        images : torch.Tensor
            Shape (B, 3, 518, 518), float in [0, 1].

        Returns
        -------
        torch.Tensor
            Shape (B, 1374, 768).
            Layout: [CLS, reg0, reg1, reg2, reg3, patch_0, …, patch_1368].
        """
        x = self.normalize(images)
        return self.vit.prepare_tokens_with_masks(x)

    # ── Masked block replay (fast path in RL loop) ────────────────────────────

    def forward_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Replay all 12 transformer blocks on pre-cached tokens, with the
        current pruning mask active via the registered pre-hooks.

        Parameters
        ----------
        tokens : torch.Tensor
            Shape (B, 1374, 768) — output of extract_tokens().

        Returns
        -------
        torch.Tensor
            Spatial feature map of shape (B, 768, 37, 37).
        """
        x = tokens
        for blk in self.vit.blocks:
            x = blk(x)
        x = self.vit.norm(x)

        # Discard CLS + 4 register tokens; keep the 1369 patch tokens
        patches = x[:, 5:, :]                                   # (B, 1369, 768)
        B = patches.size(0)
        return patches.permute(0, 2, 1).reshape(
            B, self.EMBED_DIM, self.GRID_SIZE, self.GRID_SIZE
        )                                                        # (B, 768, 37, 37)

    # ── Standard forward ──────────────────────────────────────────────────────

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        End-to-end forward: normalise → token embedding → 12 blocks → reshape.

        Parameters
        ----------
        images : torch.Tensor
            Shape (B, 3, 518, 518), float in [0, 1].

        Returns
        -------
        torch.Tensor
            Shape (B, 768, 37, 37).
        """
        return self.forward_from_tokens(self.extract_tokens(images))

    # ── Taylor importance utilities ───────────────────────────────────────────

    def get_intra_layer_ranks(self, importance: torch.Tensor) -> torch.Tensor:
        """
        Compute the normalised within-layer rank of each head's Taylor score.

        A rank of 0.0 means the head is the least important in its layer;
        1.0 means most important.

        Parameters
        ----------
        importance : torch.Tensor
            Shape (12, 12) — Taylor importance scores, one per (layer, head).

        Returns
        -------
        torch.Tensor
            Shape (12, 12) — normalised ranks in [0, 1].
        """
        ranks = torch.argsort(
            torch.argsort(importance, dim=1), dim=1
        ).float()
        return ranks / max(importance.shape[1] - 1, 1)
