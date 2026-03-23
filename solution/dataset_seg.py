"""
【数据集】语义分割用：读 pipeline 产出的 images/ + masks/（像素级类别 id）。

返回张量约定：
  - image: float32，形状 [3,H,W]，已做 ImageNet 均值方差归一化（与常见 ViT 输入一致）；
  - mask: int64，形状 [H,W]，取值 0 .. num_classes-1。

说明：主训练脚本 train.py 若走 YOLO-seg，则用 dataset_seg_yolo；本模块仍提供
  split_stems()、default_train_transforms() 供 YOLO 数据管线复用。
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def default_train_transforms() -> Callable:
    """轻量数据增强：随机水平翻转（同步作用于图与掩膜）。"""
    def _apply(img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask

    return _apply


class SegmentationPatchDataset(Dataset):
    """
    从目录加载 (image, mask) 对；假定文件名一一对应（同 stem）。

    Args:
        images_dir: pipeline 输出的 images 目录
        masks_dir: pipeline 输出的 masks 目录
        image_list: 可选，仅使用这些 stem（不含扩展名）；为 None 则使用 images_dir 下全部 png
        transform: 可选，接收 (PIL.Image, PIL.Image) -> (PIL, PIL)
        num_classes: 分割类别数（含背景），用于校验掩膜取值范围 [0, num_classes-1]
    """

    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        image_list: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        num_classes: int = 5,
    ):
        super().__init__()
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.num_classes = num_classes
        self.transform = transform

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
            raise FileNotFoundError(
                f"No matched image/mask pairs under {self.images_dir} / {self.masks_dir}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        ip, mp = self.samples[idx]
        img = Image.open(ip).convert("RGB")
        mask = Image.open(mp)

        if self.transform is not None:
            img, mask = self.transform(img, mask)

        img_np = np.array(img, dtype=np.float32) / 255.0
        # ImageNet 归一化（与 ViT 教师一致时常用）
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std
        img_chw = np.transpose(img_np, (2, 0, 1))

        mask_np = np.array(mask, dtype=np.int64)
        if mask_np.max() >= self.num_classes or mask_np.min() < 0:
            # 不中断训练，仅截断到合法范围（生产环境应修复标注）
            mask_np = np.clip(mask_np, 0, self.num_classes - 1)

        return {
            "image": torch.from_numpy(img_chw),
            "mask": torch.from_numpy(mask_np),
            "stem": ip.stem,
        }


def split_stems(
    images_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """按文件名 stem 随机划分训练/验证列表。"""
    stems = sorted({p.stem for p in Path(images_dir).glob("*.png")})
    rng = random.Random(seed)
    rng.shuffle(stems)
    n_val = max(1, int(len(stems) * val_ratio)) if len(stems) > 1 else 0
    val_stems = stems[:n_val]
    train_stems = stems[n_val:]
    if not train_stems:
        train_stems = val_stems
        val_stems = []
    return train_stems, val_stems
