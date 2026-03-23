# -*- coding: utf-8 -*-
"""
【数据集】语义掩膜 → Ultralytics YOLO「实例分割」训练格式

深度学习里常见两种分割：
  - **语义分割**：每个像素只有一个类别号（0=背景，1=裂缝…），本仓库 pipeline 输出的 mask 就是这种。
  - **实例分割**：每个「物体实例」单独一条标注（框 + 类 + 该实例的掩膜）。YOLO-seg 的损失函数按实例算。

本文件做的事：
  把一张语义 mask 里「同一类别、彼此不连通的区域」当成多个实例（连通域），
  为每个实例生成：类别 id、包围框 (cx,cy,w,h 归一化)、以及在一张「合并图」上用 1,2,3… 标记实例编号
  （与 Ultralytics v8SegmentationLoss 在 overlap_mask=True 时的约定一致）。

依赖：OpenCV (cv2)，一般随 ultralytics 一起装好。
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

try:
    import cv2
except ImportError as e:
    cv2 = None  # type: ignore
    _CV2_ERR = e
else:
    _CV2_ERR = None

from torch.utils.data import Dataset


def collate_seg_yolo(batch: List[dict]) -> dict:
    """
    把多个样本拼成一个 batch（PyTorch DataLoader 的 collate_fn）。

    返回字典里：
      img: [B, 3, H, W]，数值约 0~1（与 Ultralytics 训练图像一致）
      batch_idx: 每条目标属于 batch 里第几张图
      cls, bboxes: 所有图的目标拼在一起
      masks: [B, H, W]，浮点张量但存的是「实例 id」（0=背景，1,2,…=第几个实例）
    """
    imgs = torch.stack([b["img"] for b in batch], dim=0)
    mask_stack = torch.stack([b["merged_instance_mask"] for b in batch], dim=0).float()

    batch_idx_parts: List[torch.Tensor] = []
    cls_parts: List[torch.Tensor] = []
    bbox_parts: List[torch.Tensor] = []

    for i, b in enumerate(batch):
        n = b["cls"].shape[0]
        if n == 0:
            continue
        batch_idx_parts.append(torch.full((n, 1), float(i), dtype=torch.float32))
        cls_parts.append(b["cls"])
        bbox_parts.append(b["bboxes"])

    if batch_idx_parts:
        batch_idx = torch.cat(batch_idx_parts, dim=0)
        cls = torch.cat(cls_parts, dim=0)
        bboxes = torch.cat(bbox_parts, dim=0)
    else:
        batch_idx = torch.zeros(0, 1, dtype=torch.float32)
        cls = torch.zeros(0, 1, dtype=torch.float32)
        bboxes = torch.zeros(0, 4, dtype=torch.float32)

    return {
        "img": imgs,
        "batch_idx": batch_idx,
        "cls": cls,
        "bboxes": bboxes,
        "masks": mask_stack,
    }


class SemanticMaskYoloSegDataset(Dataset):
    """
    PyTorch Dataset：按「文件名 stem」配对 images/*.png 与 masks/*.png。

    参数 num_classes：语义类别总数 **含背景**。例如 5 表示像素值合法范围 0~4。
    YOLO 的前景类数 nc = num_classes - 1（背景不参与检测类索引）。

    当 ``num_classes==5`` 时，语义 id 1..4 依次对应 YOLO class 0..3，与 ``datasets/data.yaml`` 中
    ``names`` 顺序一致：crack, decay, defect, knot（见 ``class_order.py``）。
    """

    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        image_list: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        num_classes: int = 5,
        min_component_area: int = 4,
    ):
        if _CV2_ERR is not None:
            raise ImportError("需要 opencv-python: pip install opencv-python") from _CV2_ERR

        super().__init__()
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.num_classes = num_classes
        self.transform = transform
        self.min_component_area = min_component_area

        if image_list is not None:
            stems = [s.strip() for s in image_list if s.strip()]
        else:
            stems = sorted({p.stem for p in self.images_dir.glob("*.png")})

        self.samples: List[Tuple[Path, Path]] = []
        for stem in stems:
            ip = self.images_dir / f"{stem}.png"
            mp = self.masks_dir / f"{stem}.png"
            if ip.is_file() and mp.is_file():
                self.samples.append((ip, mp))

        if not self.samples:
            raise FileNotFoundError(f"No matched image/mask pairs under {self.images_dir} / {self.masks_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def _semantic_to_instances(
        self, mask_np: np.ndarray
    ) -> Tuple[np.ndarray, List[int], List[Tuple[float, float, float, float]]]:
        """
        核心：语义图 → 实例列表 + 合并 id 图。

        对每个前景语义 id（1..num_classes-1）做 connectedComponents，
        每个连通域 = 一个实例。合并图 merged 中该域像素值 = 实例序号（从 1 开始），
        与 YOLO 里「第 k 个 gt 对应 merged==k+1」的约定一致（由损失里 target_gt_idx 对齐）。
        """
        h, w = mask_np.shape
        merged = np.zeros((h, w), dtype=np.int32)
        classes: List[int] = []
        bboxes: List[Tuple[float, float, float, float]] = []

        blobs: List[Tuple[int, np.ndarray]] = []

        for sem_id in range(1, self.num_classes):
            binary = (mask_np == sem_id).astype(np.uint8)
            if binary.sum() == 0:
                continue
            n_lab, labels = cv2.connectedComponents(binary)
            for lab in range(1, n_lab):
                comp = (labels == lab).astype(np.uint8)
                area = int(comp.sum())
                if area < self.min_component_area:
                    continue
                blobs.append((sem_id - 1, comp))

        areas = [int(b[1].sum()) for b in blobs]
        order = sorted(range(len(blobs)), key=lambda i: -areas[i])

        for k, j in enumerate(order):
            yolo_cls, comp = blobs[j]
            ys, xs = np.where(comp > 0)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            merged[comp > 0] = k + 1
            classes.append(yolo_cls)
            bw, bh = x2 - x1 + 1, y2 - y1 + 1
            cx = (x1 + x2 + 1) / 2.0 / w
            cy = (y1 + y2 + 1) / 2.0 / h
            bboxes.append((cx, cy, bw / w, bh / h))

        return merged, classes, bboxes

    def __getitem__(self, idx: int) -> dict:
        ip, mp = self.samples[idx]
        img = Image.open(ip).convert("RGB")
        mask = Image.open(mask)

        if self.transform is not None:
            img, mask = self.transform(img, mask)

        w, h = img.size
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_chw = np.transpose(img_np, (2, 0, 1))
        mask_np = np.array(mask, dtype=np.int64)
        if mask_np.shape[:2] != (h, w):
            raise ValueError(f"mask spatial {mask_np.shape} != image ({h},{w})")
        if mask_np.max() >= self.num_classes or mask_np.min() < 0:
            mask_np = np.clip(mask_np, 0, self.num_classes - 1)

        merged, classes, bboxes = self._semantic_to_instances(mask_np)

        if classes:
            cls_t = torch.tensor(classes, dtype=torch.float32).view(-1, 1)
            bbox_t = torch.tensor(bboxes, dtype=torch.float32).view(-1, 4)
        else:
            cls_t = torch.zeros(0, 1, dtype=torch.float32)
            bbox_t = torch.zeros(0, 4, dtype=torch.float32)

        return {
            "img": torch.from_numpy(img_chw),
            "cls": cls_t,
            "bboxes": bbox_t,
            "merged_instance_mask": torch.from_numpy(merged.astype(np.int64)),
            "stem": ip.stem,
        }
