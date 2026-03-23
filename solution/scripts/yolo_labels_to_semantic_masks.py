"""
根据 YOLO 格式 ``.txt`` 标注生成 ``pipeline.py`` 可用的语义 mask PNG（``--mask-dir``）。

支持两种每行格式（与 Ultralytics / Roboflow 常见导出一致）：

1. **检测**：``class cx cy w h``（均为相对整图宽高的 0~1 归一化）
2. **实例/语义分割（多边形）**：``class x1 y1 x2 y2 ...``（坐标同样归一化，至少 3 个点）

**类别映射**（须与 ``datasets/data.yaml``、``class_order.py`` 一致）::

    YOLO class 0,1,2,3  ->  mask 像素 1,2,3,4  (crack, decay, defect, knot)
    背景                ->  0

重叠区域：按文件中**行顺序**，后绘制的多边形/框会覆盖先绘制的像素。

在 ``solution`` 目录执行示例::

    python scripts/yolo_labels_to_semantic_masks.py \\
        --images-dir datasets/train/images \\
        --labels-dir datasets/train/labels \\
        --output-dir data/masks_from_yolo

再与 ``images-dir`` 一起交给 ``pipeline.py``（原图目录可用同一份 ``images`` 或复制一份指向原图的路径）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import cv2
except ImportError as e:
    print("需要 opencv-python: pip install opencv-python", file=sys.stderr)
    raise SystemExit(1) from e

from class_order import YOLO_CLASS_NAMES

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def yolo_class_to_mask_pixel(yolo_cls: int, num_yolo_classes: int) -> int | None:
    """YOLO 前景类 0..nc-1 -> 语义 mask 1..nc；非法则返回 None。"""
    if yolo_cls < 0 or yolo_cls >= num_yolo_classes:
        return None
    return yolo_cls + 1


def rasterize_line(
    mask: np.ndarray,
    parts: list[str],
    num_yolo_classes: int,
) -> list[str]:
    """在 mask 上绘制一行标注，返回警告信息列表。"""
    warns: list[str] = []
    if len(parts) < 2:
        return ["空行或缺少类别"]
    try:
        yolo_cls = int(float(parts[0]))
    except ValueError:
        return [f"无法解析类别: {parts[0]}"]

    mid = yolo_class_to_mask_pixel(yolo_cls, num_yolo_classes)
    if mid is None:
        return [f"类别 {yolo_cls} 超出 0..{num_yolo_classes - 1}"]

    try:
        rest = [float(x) for x in parts[1:]]
    except ValueError as e:
        return [f"坐标解析失败: {e}"]

    H, W = mask.shape[:2]

    if len(rest) == 4:
        # 检测框 cx cy w h
        cx, cy, bw, bh = rest
        x1 = int(round((cx - bw / 2.0) * W))
        y1 = int(round((cy - bh / 2.0) * H))
        x2 = int(round((cx + bw / 2.0) * W))
        y2 = int(round((cy + bh / 2.0) * H))
        x1 = max(0, min(W - 1, x1))
        x2 = max(0, min(W, x2))
        y1 = max(0, min(H - 1, y1))
        y2 = max(0, min(H, y2))
        if x2 <= x1 or y2 <= y1:
            return ["跳过退化框"]
        mask[y1:y2, x1:x2] = mid
        return []

    if len(rest) >= 6 and len(rest) % 2 == 0:
        pts = np.array(
            [[int(round(rest[i] * W)), int(round(rest[i + 1] * H))] for i in range(0, len(rest), 2)],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [pts], int(mid))
        return []

    return [f"无法识别格式（坐标个数 {len(rest)}），需 4 个(bbox)或 ≥6 偶数个(polygon)"]


def main() -> None:
    p = argparse.ArgumentParser(description="YOLO txt -> semantic mask PNG for pipeline.py mask-dir")
    p.add_argument("--images-dir", type=Path, required=True, help="与 labels 同名的图像目录")
    p.add_argument("--labels-dir", type=Path, required=True, help="YOLO 标签 *.txt 目录")
    p.add_argument("--output-dir", type=Path, required=True, help="输出语义 mask PNG 目录（将创建）")
    p.add_argument(
        "--nc",
        type=int,
        default=len(YOLO_CLASS_NAMES),
        help=f"YOLO 前景类数（默认 {len(YOLO_CLASS_NAMES)}，与 data.yaml 一致）",
    )
    p.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    args = p.parse_args()

    img_dir = args.images_dir.resolve()
    lbl_dir = args.labels_dir.resolve()
    out_dir = args.output_dir.resolve()

    if not img_dir.is_dir():
        print(f"图像目录不存在: {img_dir}", file=sys.stderr)
        raise SystemExit(1)
    if not lbl_dir.is_dir():
        print(f"标签目录不存在: {lbl_dir}", file=sys.stderr)
        raise SystemExit(1)

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        fp for fp in img_dir.iterdir() if fp.is_file() and fp.suffix.lower() in IMAGE_EXTS
    )
    if not image_files:
        print(f"{img_dir} 下未找到支持的图像", file=sys.stderr)
        raise SystemExit(1)

    n_ok = 0
    n_empty_label = 0
    n_warn = 0

    for img_path in image_files:
        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"

        with Image.open(img_path) as im:
            w_img, h_img = im.size

        mask = np.zeros((h_img, w_img), dtype=np.uint8)

        if lbl_path.is_file():
            text = lbl_path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                for line_no, line in enumerate(text.splitlines(), start=1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    ws = rasterize_line(mask, parts, args.nc)
                    for wmsg in ws:
                        print(f"[{stem}.txt:{line_no}] {wmsg}", file=sys.stderr)
                        n_warn += 1
            else:
                n_empty_label += 1
        else:
            n_empty_label += 1
            print(f"[提示] 无标签文件，输出全背景: {stem}", file=sys.stderr)

        if not args.dry_run:
            out_path = out_dir / f"{stem}.png"
            Image.fromarray(mask, mode="L").save(out_path)
        n_ok += 1

    print(
        f"完成: 处理 {n_ok} 张图，输出目录 {out_dir}；"
        f"无标注或空标注计 {n_empty_label}；警告行 {n_warn}"
    )
    print("映射: YOLO class " + ", ".join(f"{i}->{YOLO_CLASS_NAMES[i]}" for i in range(min(args.nc, len(YOLO_CLASS_NAMES)))) + f" -> mask 1..{args.nc}")


if __name__ == "__main__":
    main()
