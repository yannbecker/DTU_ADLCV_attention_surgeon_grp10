# yolo_dinov2 — DINOv2 + YOLOv8 single-scale detection with RL pruning support
# Architecture: DINOv2 ViT-B/14 (frozen, prunable) → 37×37 patch grid → Detect head (stride=14)
# Reference: itsprakhar/Yolo-DinoV2 (DINOv2 backbone + Ultralytics Detect head)

from .backbone import DinoPrunableBackbone
from .detector import DinoPrunableDetector, build_criterion, _build_batch_dict
from .dataset import COCODetectionDatasetV2, get_coco_loaders_v2, CAT_TO_IDX, IDX_TO_CAT
from .rl_validator import DinoPrunableValidator

__all__ = [
    "DinoPrunableBackbone",
    "DinoPrunableDetector",
    "build_criterion",
    "_build_batch_dict",
    "COCODetectionDatasetV2",
    "get_coco_loaders_v2",
    "CAT_TO_IDX",
    "IDX_TO_CAT",
    "DinoPrunableValidator",
]
