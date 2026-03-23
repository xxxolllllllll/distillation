"""
================================================================================
【数据流水线】原图 + 语义掩膜 → 训练用 patch（图像与 mask 严格对齐）
================================================================================

整体在做什么（可对照 实验方案.md 第 3.1 节）：
  1) 原图过长边缩小（max_long_edge），减轻显存与 IO；
  2) 大场景用滑窗切 1024×1024（可改 patch_size / window_stride）；
  3) 对小目标按「细长 / 紧凑极小」等规则裁 ROI，再缩放到 target_input_size；
  4) 写出 images/*.png、masks/*.png 及 meta/*.json 便于追溯。

约定：mask 像素值 = 类别 id（0 背景，1~4 为裂缝/腐朽/缺损/节子等，与 PipelineConfig 一致）。
================================================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from class_order import MASK_CRACK, MASK_DECAY, MASK_DEFECT, MASK_KNOT


@dataclass
class PipelineConfig:
    """
    全流程配置参数：
    - 输入/输出路径
    - 类别 ID 映射
    - 原图级缩放上限
    - 大场景滑窗 patch 尺寸与步长
    - ROI 判定与构造相关阈值（细长/紧凑/极小、最小边长、固定方形边长）
    - 统一网络输入尺寸
    """
    # Input/output
    image_dir: Path
    mask_dir: Path
    output_dir: Path
    # Labels（与 datasets/data.yaml 中 names 顺序一致，见 class_order.py）
    background_id: int = 0
    crack_id: int = MASK_CRACK
    decay_id: int = MASK_DECAY
    defect_id: int = MASK_DEFECT
    knot_id: int = MASK_KNOT
    # Global preprocess
    max_long_edge: int = 3000
    # Sliding window
    patch_size: int = 1024
    window_stride: int = 800
    keep_background_ratio: float = 0.2
    # ROI for small targets
    r_max: float = 5.0
    r_line: float = 5.0
    r_compact: float = 2.5
    l_tiny: int = 24
    s_min: int = 128
    s_hole: int = 192
    target_input_size: int = 1024


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


def mask_components(mask: np.ndarray, target_ids: Iterable[int]) -> List[Tuple[int, int, int, int, int]]:
    """
    在掩膜上对给定类别集合 target_ids 做连通域分割。
    返回每个连通块的外接矩形和类别 ID，形式为：
    (x_min, y_min, x_max, y_max, cls_id)，右边界为开区间。
    采用简单的 BFS flood fill，不依赖额外图像库。
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
            # BFS component
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


def adjust_roi(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    cls_id: int,
    img_w: int,
    img_h: int,
    cfg: PipelineConfig,
) -> Tuple[int, int, int, int, str]:
    """
    根据实验方案中的 ROI 规则，为单个目标外接框选择并构造 ROI：
    - Rule A: 自适应矩形 ROI（默认，适用于裂缝或细长目标）
    - Rule B: 固定边长正方形 ROI（仅适用于紧凑且极小目标）
    优先级：若同时满足“极小”和“细长”，优先按 Rule A 处理，以避免裂缝被误归为点状目标。
    返回裁剪到图像边界内的 ROI 坐标 (rx1,ry1,rx2,ry2) 以及使用的规则名 roi_rule。
    """
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # ROI rule decision:
    # Rule A (adaptive rectangle, default): crack class OR elongated shape.
    # Rule B (fixed square): compact and tiny targets only.
    # Priority: if tiny and elongated, still use Rule A.
    short_edge = max(1, min(w, h))
    long_edge = max(w, h)
    ratio = long_edge / float(short_edge)
    is_crack = cls_id == cfg.crack_id
    use_rule_a = is_crack or (ratio > cfg.r_line)
    use_rule_b = (not use_rule_a) and (long_edge <= cfg.l_tiny) and (ratio <= cfg.r_compact)

    if use_rule_b:
        roi_rule = "rule_b_fixed_square"
        half = cfg.s_hole / 2.0
        rx1 = int(round(cx - half))
        ry1 = int(round(cy - half))
        rx2 = int(round(cx + half))
        ry2 = int(round(cy + half))
    else:
        roi_rule = "rule_a_adaptive_rect"
        rw, rh = float(w), float(h)
        long_edge = max(rw, rh)
        short_edge = min(rw, rh)
        if short_edge <= 0:
            short_edge = 1.0
        if long_edge / short_edge > cfg.r_max:
            short_edge = long_edge / cfg.r_max
            if rw >= rh:
                rh = short_edge
            else:
                rw = short_edge
        rw = max(rw, float(cfg.s_min))
        rh = max(rh, float(cfg.s_min))
        rx1 = int(round(cx - rw / 2.0))
        ry1 = int(round(cy - rh / 2.0))
        rx2 = int(round(cx + rw / 2.0))
        ry2 = int(round(cy + rh / 2.0))

    rx1 = max(0, rx1)
    ry1 = max(0, ry1)
    rx2 = min(img_w, rx2)
    ry2 = min(img_h, ry2)
    if rx2 <= rx1:
        rx2 = min(img_w, rx1 + 1)
    if ry2 <= ry1:
        ry2 = min(img_h, ry1 + 1)
    return rx1, ry1, rx2, ry2, roi_rule


def resize_and_pad_roi(
    roi_img: np.ndarray,
    roi_mask: np.ndarray,
    target_size: int,
    background_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将任意大小的 ROI 图像和掩膜按保持长宽比的方式缩放，
    再 padding 到 target_size×target_size：
    - 缩放因子 = min(target_size/H, target_size/W)
    - 缩放后不足部分对称 padding
    """
    h, w = roi_img.shape[:2]
    scale = min(target_size / float(h), target_size / float(w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    img_rs = np.array(Image.fromarray(roi_img).resize((new_w, new_h), Image.BILINEAR))
    msk_rs = np.array(Image.fromarray(roi_mask).resize((new_w, new_h), Image.NEAREST))
    return pad_to_size(img_rs, msk_rs, target_size, target_size, background_id)


def run_pipeline(cfg: PipelineConfig) -> None:
    """
    执行完整数据预处理与切片流程：
    1) 原图级等比例缩放
    2) 大场景滑窗切片并抽样保留背景 patch
    3) 按 ROI 规则为 crack/defect/knot 等小目标生成 ROI patch
    4) 将所有 patch 统一到 target_input_size，并保存图像/掩膜/元信息
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

    for img_path in image_files:
        stem = img_path.stem
        mask_path = cfg.mask_dir / f"{stem}.png"
        if not mask_path.exists():
            continue
        image = read_image(img_path)
        mask = read_mask(mask_path)
        image, mask = resize_keep_ratio(image, mask, cfg.max_long_edge)
        h, w = image.shape[:2]

        # 1) Large-scene sliding patches
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
                    # keep only a subset of empty patches
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

        # 2) ROI patches for small targets
        comp_ids = [cfg.crack_id, cfg.defect_id, cfg.knot_id]
        comps = mask_components(mask, comp_ids)
        for idx, (x1, y1, x2, y2, cls_id) in enumerate(comps):
            rx1, ry1, rx2, ry2, roi_rule = adjust_roi(x1, y1, x2, y2, cls_id, w, h, cfg)
            roi_img = image[ry1:ry2, rx1:rx2]
            roi_mask = mask[ry1:ry2, rx1:rx2]
            roi_img, roi_mask = resize_and_pad_roi(roi_img, roi_mask, cfg.target_input_size, cfg.background_id)
            name = f"{stem}_roi_{idx:05d}"
            save_image(out_img / f"{name}.png", roi_img)
            save_mask(out_msk / f"{name}.png", roi_mask)
            (out_meta / f"{name}.json").write_text(
                json.dumps(
                    {
                        "type": "roi",
                        "class_id": cls_id,
                        "roi_rule": roi_rule,
                        "box": [x1, y1, x2, y2],
                        "roi": [rx1, ry1, rx2, ry2],
                        "target_size": cfg.target_input_size,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )


def parse_args() -> PipelineConfig:
    """
    从命令行解析配置参数并构造 PipelineConfig。
    只暴露关键阈值和路径，其他使用默认值即可支持论文中的实验方案。
    """
    parser = argparse.ArgumentParser(description="Data preprocessing and slicing pipeline for high-res wood-defect segmentation.")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-long-edge", type=int, default=3000)
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--window-stride", type=int, default=800)
    parser.add_argument("--target-input-size", type=int, default=1024)
    parser.add_argument("--r-max", type=float, default=5.0)
    parser.add_argument("--r-line", type=float, default=5.0)
    parser.add_argument("--r-compact", type=float, default=2.5)
    parser.add_argument("--l-tiny", type=int, default=24)
    parser.add_argument("--s-min", type=int, default=128)
    parser.add_argument("--s-hole", type=int, default=192)
    args = parser.parse_args()

    return PipelineConfig(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        max_long_edge=args.max_long_edge,
        patch_size=args.patch_size,
        window_stride=args.window_stride,
        target_input_size=args.target_input_size,
        r_max=args.r_max,
        r_line=args.r_line,
        r_compact=args.r_compact,
        l_tiny=args.l_tiny,
        s_min=args.s_min,
        s_hole=args.s_hole,
    )


if __name__ == "__main__":
    run_pipeline(parse_args())
