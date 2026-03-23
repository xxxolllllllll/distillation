"""
================================================================================
【数据流水线】原图 + 语义掩膜 → 训练用 patch（图像与 mask 严格对齐）
================================================================================

流程：
  1) 原图过长边缩小（max_long_edge），减轻显存与 IO；
  2) 仅用**滑动窗口**切 patch_size×patch_size（步长 window_stride），越界处对称 pad；
  3) 对几乎全背景的窗口按 keep_background_ratio 抽样保留，避免训练被空 patch 淹没；
  4) 写出 images/*.png、masks/*.png 及 meta/*.json。

约定：mask 像素值 = 类别 id（0 背景，1~4 为裂缝/腐朽/缺损/节子等）。
================================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@dataclass
class PipelineConfig:
    """
    全流程配置：
    - 输入/输出路径
    - 背景类别 id（用于 mask 补边）
    - 原图长边上限、滑窗尺寸与步长、空背景 patch 保留比例
    """
    image_dir: Path
    mask_dir: Path
    output_dir: Path
    background_id: int = 0
    max_long_edge: int = 3000
    patch_size: int = 1024
    window_stride: int = 800
    keep_background_ratio: float = 0.2


def read_image(path: Path) -> np.ndarray:
    """读取 RGB 图像，返回 H×W×3 的 uint8 数组。"""
    return np.array(Image.open(path).convert("RGB"))


def read_mask(path: Path) -> np.ndarray:
    """
    读取语义分割掩膜，按原始像素值返回 H×W 的整数数组（类别 ID）。
    不做颜色到 ID 的额外映射，假定文件本身已用类别索引编码。
    """
    return np.array(Image.open(path))


def save_image(path: Path, array: np.ndarray) -> None:
    """将 H×W×3 数组保存为图像文件。"""
    Image.fromarray(array).save(path)


def save_mask(path: Path, array: np.ndarray) -> None:
    """将 H×W 掩膜数组保存为单通道 PNG（uint8）。"""
    Image.fromarray(array.astype(np.uint8)).save(path)


def resize_keep_ratio(image: np.ndarray, mask: np.ndarray, max_long_edge: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    对原图和掩膜做等比例缩放，使长边不超过 max_long_edge，保持长宽比不变。
    - image: H×W×3
    - mask: H×W
    返回缩放后的 image, mask。
    """
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return image, mask

    scale = max_long_edge / float(long_edge)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img_rs = np.array(Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR))
    msk_rs = np.array(Image.fromarray(mask).resize((new_w, new_h), Image.NEAREST))
    return img_rs, msk_rs


def sliding_starts(length: int, patch: int, stride: int) -> List[int]:
    """
    计算按给定窗口大小 patch 和步长 stride 在 1D 方向上的起始坐标列表，
    保证覆盖到最后一个位置（即最后一个窗口右/下边界对齐 length）。
    """
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def pad_to_size(image: np.ndarray, mask: np.ndarray, out_h: int, out_w: int, background_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    将任意大小的图像和掩膜对称 padding 到指定尺寸 (out_h, out_w)。
    - 图像使用边缘复制模式填充
    - 掩膜使用 background_id 常数填充
    """
    h, w = image.shape[:2]
    pad_h = max(out_h - h, 0)
    pad_w = max(out_w - w, 0)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    img_pad = np.pad(image, ((top, bottom), (left, right), (0, 0)), mode="edge")
    msk_pad = np.pad(mask, ((top, bottom), (left, right)), mode="constant", constant_values=background_id)
    return img_pad, msk_pad


def window_crop(image: np.ndarray, mask: np.ndarray, x: int, y: int, size: int, background_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    从给定左上角 (x,y) 裁剪 size×size 的窗口，若越界则先裁后 pad。
    返回对应的图像 patch 和掩膜 patch。
    """
    h, w = image.shape[:2]
    x2 = min(x + size, w)
    y2 = min(y + size, h)
    img = image[y:y2, x:x2]
    msk = mask[y:y2, x:x2]
    return pad_to_size(img, msk, size, size, background_id)


def run_pipeline(cfg: PipelineConfig) -> None:
    """
    执行数据预处理：原图级缩放 → 仅滑窗切片 → 写出 patch 与 meta。
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_img = cfg.output_dir / "images"
    out_msk = cfg.output_dir / "masks"
    out_meta = cfg.output_dir / "meta"
    out_img.mkdir(exist_ok=True)
    out_msk.mkdir(exist_ok=True)
    out_meta.mkdir(exist_ok=True)

    image_files = sorted([p for p in cfg.image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}])
    kept_bg = 0
    total_bg = 0
    n_processed = 0
    n_skipped_no_mask = 0

    for img_path in image_files:
        stem = img_path.stem
        mask_path = cfg.mask_dir / f"{stem}.png"
        if not mask_path.exists():
            n_skipped_no_mask += 1
            continue
        image = read_image(img_path)
        mask = read_mask(mask_path)
        image, mask = resize_keep_ratio(image, mask, cfg.max_long_edge)
        h, w = image.shape[:2]

        ys = sliding_starts(h, cfg.patch_size, cfg.window_stride)
        xs = sliding_starts(w, cfg.patch_size, cfg.window_stride)
        patch_index = 0
        for y in ys:
            for x in xs:
                patch_img, patch_mask = window_crop(image, mask, x, y, cfg.patch_size, cfg.background_id)
                fg_ratio = float(np.mean(patch_mask != cfg.background_id))
                keep = True
                if fg_ratio < 1e-5:
                    total_bg += 1
                    if kept_bg / float(max(total_bg, 1)) > cfg.keep_background_ratio:
                        keep = False
                    else:
                        kept_bg += 1
                if keep:
                    name = f"{stem}_win_{patch_index:05d}"
                    save_image(out_img / f"{name}.png", patch_img)
                    save_mask(out_msk / f"{name}.png", patch_mask)
                    (out_meta / f"{name}.json").write_text(
                        json.dumps({"type": "window", "x": x, "y": y, "w": cfg.patch_size, "h": cfg.patch_size}, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    patch_index += 1

        n_processed += 1

    if n_processed == 0:
        print(
            "[pipeline] 未生成任何 patch：mask-dir 下需要与图像**同名**的 **.png** 语义掩膜（每像素=类别 id）。\n"
            f"  扫描图像目录: {cfg.image_dir.resolve()}（共 {len(image_files)} 张）\n"
            f"  掩膜目录: {cfg.mask_dir.resolve()}\n"
            "  若只有 YOLO 的 .txt，不能直接作 mask-dir；请先执行:\n"
            "    python scripts/yolo_labels_to_semantic_masks.py "
            f"--images-dir {cfg.image_dir} --labels-dir <你的labels> --output-dir <生成png的目录>\n"
            "  再将 pipeline 的 --mask-dir 指向该目录。",
            file=sys.stderr,
        )
    else:
        print(
            f"[pipeline] 完成：处理原图 {n_processed} 张，跳过（无同名 mask.png）{n_skipped_no_mask} 张；输出 {cfg.output_dir.resolve()}",
            file=sys.stderr,
        )


def parse_args() -> PipelineConfig:
    """命令行参数 → PipelineConfig（仅滑窗相关）。"""
    parser = argparse.ArgumentParser(description="Sliding-window preprocessing: image + semantic mask → aligned patches.")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-long-edge", type=int, default=3000)
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=800)
    args = parser.parse_args()

    return PipelineConfig(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        max_long_edge=args.max_long_edge,
        patch_size=args.patch_size,
        window_stride=args.window_stride,
    )


if __name__ == "__main__":
    run_pipeline(parse_args())
