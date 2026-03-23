"""
================================================================================
类别顺序 — 须与 ``datasets/data.yaml`` 中 ``names`` 一致
================================================================================

**YOLO 标签**（检测/分割的 ``class`` 索引，不含背景）::

    0 = crack, 1 = decay, 2 = defect, 3 = knot

**语义分割 mask 像素 id**（含背景；``pipeline``、``train.py`` 的 ``--num-classes=5``）::

    0 = background, 1 = crack, 2 = decay, 3 = defect, 4 = knot

映射关系：``YOLO_class == mask_pixel_id - 1``（当 mask 前景为 1..4 时）。

若 Roboflow 导出顺序与此不一致，应修改 ``data.yaml`` 的 ``names`` 或重排标签，
并同步修改本文件与 ``pipeline`` 中的 id 常量。
================================================================================
"""

from __future__ import annotations

# 与 data.yaml 中 names 顺序一致（YOLO 0..nc-1）
YOLO_CLASS_NAMES: tuple[str, ...] = ("crack", "decay", "defect", "knot")

MASK_BACKGROUND = 0
MASK_CRACK = 1
MASK_DECAY = 2
MASK_DEFECT = 3
MASK_KNOT = 4

# train_detect_distill 中 cls_ids 顺序：决定 mask id → YOLO id 的语义（须与上行一一对应）
MASK_FOREGROUND_IDS_IN_YOLO_ORDER: tuple[int, ...] = (
    MASK_CRACK,
    MASK_DECAY,
    MASK_DEFECT,
    MASK_KNOT,
)
