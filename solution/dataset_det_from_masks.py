"""
【数据集】检测用：从「语义分割 mask」生成「目标检测框」。

为什么需要它：
  Ultralytics 的检测头要的是「每张图若干框 + 类别」，而 pipeline 给的是「每像素一个类」。
  本文件对每个前景类别做连通域，取外接矩形，再转成 YOLO 常用的 cx,cy,w,h（相对整图 0~1）。

目录约定：
  images/*.png 与 masks/*.png 同名；mask 像素 id：0 背景，1~4 为 crack/decay/defect/knot
  （与 ``datasets/data.yaml`` 的 ``names`` 顺序一致，见 ``class_order.py``；``cls_offset=1`` 时 YOLO class 0..3）。

配合：train_detect_distill.py + collate_det。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from class_order import MASK_FOREGROUND_IDS_IN_YOLO_ORDER


def _mask_components(mask: np.ndarray, target_ids: Iterable[int]) -> List[Tuple[int, int, int, int, int]]:
    """
    连通域外接矩形提取（BFS）。

    返回：
      (x_min, y_min, x_max, y_max, cls_id)
    其中 x_max/y_max 为开区间（即裁剪时直接用切片 [y_min:y_max, x_min:x_max]）。
    """
    target_set = set(target_ids)
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: List[Tuple[int, int, int, int, int]] = []

    for y in range(h):
        for x in range(w):
            cls = int(mask[y, x])
            if cls not in target_set or visited[y, x]:
                continue
            q = [(x, y)]
            visited[y, x] = True
            x_min = x_max = x
            y_min = y_max = y
            idx = 0
            while idx < len(q):
                cx, cy = q[idx]
                idx += 1
                x_min = min(x_min, cx)
                x_max = max(x_max, cx)
                y_min = min(y_min, cy)
                y_max = max(y_max, cy)
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if nx < 0 or ny < 0 or nx >= w or ny >= h:
                        continue
                    if visited[ny, nx]:
                        continue
                    if int(mask[ny, nx]) != cls:
                        continue
                    visited[ny, nx] = True
                    q.append((nx, ny))
            comps.append((x_min, y_min, x_max + 1, y_max + 1, cls))

    return comps


def _xyxy_to_xywh_norm(
    x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """将像素坐标 xyxy(开区间) 转为归一化 cxcywh。"""
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return (cx / img_w, cy / img_h, bw / img_w, bh / img_h)


class MaskToYoloDetDataset(Dataset):
    """
    数据集：读取 patch 图像与 mask，在线/预计算连通域框，返回 detection labels。
    """

    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        stems: Sequence[str],
        num_classes: int = 5,
        background_id: int = 0,
        cls_ids: Sequence[int] = MASK_FOREGROUND_IDS_IN_YOLO_ORDER,
        cls_offset: int = 1,
        min_box_area_px: int = 1,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.stems = list(stems)
        self.num_classes = num_classes
        self.background_id = background_id
        self.cls_ids = list(cls_ids)
        self.cls_offset = int(cls_offset)
        self.min_box_area_px = int(min_box_area_px)

        self._labels_cache: Dict[str, torch.Tensor] = {}

        for stem in self.stems:
            img_path = self.images_dir / f"{stem}.png"
            msk_path = self.masks_dir / f"{stem}.png"
            if not img_path.is_file() or not msk_path.is_file():
                continue
            img = np.array(Image.open(img_path).convert("RGB"))
            mask = np.array(Image.open(msk_path))
            labels = self._mask_to_labels(mask, img.shape[1], img.shape[0])
            self._labels_cache[stem] = labels

    def _mask_to_labels(self, mask: np.ndarray, img_w: int, img_h: int) -> torch.Tensor:
        """
        mask: H×W int
        return: [N,5] (cls,cx,cy,w,h) float32
        """
        comps = _mask_components(mask, self.cls_ids)
        out: List[List[float]] = []
        for x1, y1, x2, y2, cls_id in comps:
            bw = x2 - x1
            bh = y2 - y1
            if bw * bh < self.min_box_area_px:
                continue
            cx, cy, w, h = _xyxy_to_xywh_norm(x1, y1, x2, y2, img_w, img_h)
            # Convert mask class id -> YOLO class id (0-based)
            yolo_cls = int(cls_id) - self.cls_offset
            if 0 <= yolo_cls < (self.num_classes - 1):
                out.append([float(yolo_cls), cx, cy, w, h])
        if not out:
            return torch.zeros((0, 5), dtype=torch.float32)
        return torch.tensor(out, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]
        img_path = self.images_dir / f"{stem}.png"
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)  # HWC
        img = torch.from_numpy(img).permute(2, 0, 1)  # CHW

        # YOLO detection trainer expects images scaled to [0,1]
        img = img / 255.0

        labels = self._labels_cache.get(stem, torch.zeros((0, 5), dtype=torch.float32))
        return {"img": img, "labels": labels, "stem": stem}


def collate_det(batch: List[dict]) -> dict:
    """
    将 dataset 输出拼成 Ultralytics v8DetectionLoss 所需格式：
      batch = {
        "img": [B,3,H,W] float32
        "batch_idx": [M] float32
        "cls": [M] float32
        "bboxes": [M,4] float32, xywh normalized
      }
    """
    imgs = torch.stack([b["img"] for b in batch], dim=0)

    batch_idx_list: List[torch.Tensor] = []
    cls_list: List[torch.Tensor] = []
    bboxes_list: List[torch.Tensor] = []

    for i, b in enumerate(batch):
        labels = b["labels"]
        if labels.numel() == 0:
            continue
        cls = labels[:, 0]
        bboxes = labels[:, 1:5]
        n = labels.shape[0]
        batch_idx_list.append(torch.full((n,), float(i), dtype=torch.float32))
        cls_list.append(cls.to(torch.float32))
        bboxes_list.append(bboxes.to(torch.float32))

    if batch_idx_list:
        batch_idx = torch.cat(batch_idx_list, dim=0)
        cls = torch.cat(cls_list, dim=0)
        bboxes = torch.cat(bboxes_list, dim=0)
    else:
        batch_idx = torch.zeros((0,), dtype=torch.float32)
        cls = torch.zeros((0,), dtype=torch.float32)
        bboxes = torch.zeros((0, 4), dtype=torch.float32)

    return {"img": imgs, "batch_idx": batch_idx, "cls": cls, "bboxes": bboxes}


