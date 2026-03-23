"""
从 datasets/train 中按比例划分出 valid/ 与 test/（移动图片与同 stem 的 labels/*.txt）。

在 solution 目录执行::

    python scripts/split_train_val_test.py

默认：15% -> test，再 15% -> valid，剩余留在 train。使用固定 seed 可复现。
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Split YOLO train into train/valid/test by moving files.")
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=None,
        help="含 train/valid/test 的数据集根目录，默认: solution/datasets（相对本脚本上级）",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15, help="验证集占比（相对原 train 样本数）")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="测试集占比（相对原 train 样本数）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="只打印将移动的文件，不执行")
    args = parser.parse_args()

    solution_root = Path(__file__).resolve().parent.parent
    root = args.datasets_root if args.datasets_root is not None else solution_root / "datasets"

    train_img = root / "train" / "images"
    train_lbl = root / "train" / "labels"
    if not train_img.is_dir() or not train_lbl.is_dir():
        print(f"需要存在目录: {train_img} 与 {train_lbl}", file=sys.stderr)
        raise SystemExit(1)

    pairs: list[tuple[Path, Path]] = []
    for img_path in sorted(train_img.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl_path = train_lbl / f"{img_path.stem}.txt"
        if not lbl_path.is_file():
            print(f"[跳过] 无标签: {img_path.name}", file=sys.stderr)
            continue
        pairs.append((img_path, lbl_path))

    n = len(pairs)
    if n == 0:
        print("train/images 下没有可配对的图片+标签。", file=sys.stderr)
        raise SystemExit(1)

    n_test = round(n * args.test_ratio)
    n_val = round(n * args.val_ratio)
    n_test = max(0, min(n_test, n))
    n_val = max(0, min(n_val, n))
    # train 至少保留 1 张
    while n_test + n_val >= n:
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break
    if n_test + n_val >= n or n - n_test - n_val < 1:
        print(f"样本过少 (n={n})，无法同时划分 val 与 test。", file=sys.stderr)
        raise SystemExit(1)

    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    test_pairs = pairs[:n_test]
    val_pairs = pairs[n_test : n_test + n_val]
    keep_pairs = pairs[n_test + n_val :]

    for sub in ("valid", "test"):
        (root / sub / "images").mkdir(parents=True, exist_ok=True)
        (root / sub / "labels").mkdir(parents=True, exist_ok=True)

    def move_batch(batch: list[tuple[Path, Path]], split_name: str) -> None:
        dst_img_dir = root / split_name / "images"
        dst_lbl_dir = root / split_name / "labels"
        for img_path, lbl_path in batch:
            dst_i = dst_img_dir / img_path.name
            dst_l = dst_lbl_dir / lbl_path.name
            if args.dry_run:
                print(f"  {split_name}: {img_path.name}")
            else:
                shutil.move(str(img_path), str(dst_i))
                shutil.move(str(lbl_path), str(dst_l))

    print(f"数据集根: {root.resolve()}")
    print(f"可配对样本数: {n}  (seed={args.seed})")
    print(f"-> test: {len(test_pairs)}, valid: {len(val_pairs)}, 留在 train: {len(keep_pairs)}")
    if args.dry_run:
        print("[dry-run] 将移动到 test:")
        move_batch(test_pairs, "test")
        print("[dry-run] 将移动到 valid:")
        move_batch(val_pairs, "valid")
        return

    move_batch(test_pairs, "test")
    move_batch(val_pairs, "valid")
    print("完成。")


if __name__ == "__main__":
    main()
