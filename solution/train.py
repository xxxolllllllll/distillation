# -*- coding: utf-8 -*-
"""
================================================================================
【脚本】分割训练入口 —— Ultralytics YOLO-seg + 可选 ViT 特征蒸馏
================================================================================

你在实验方案里做的「图像 + 语义 mask」数据，先由 pipeline.py 切成 patch。
本脚本负责：

  1) 用 dataset_seg_yolo 把「语义 mask」转成 YOLO 能训练的「实例标注」；
  2) 加载官方 YOLO 分割模型（如 yolo11m-seg.pt），用 Ultralytics 自带的 v8SegmentationLoss
     同时优化：框回归、实例掩膜、分类、DFL；
  3) 若加 --distill：再算「学生 neck 特征 vs 冻结 ViT 教师」的蒸馏损失（与检测脚本思路一致）。

【你需要提前准备】
  - 已运行 pipeline，得到 data_dir/images 与 data_dir/masks（mask 像素 id 与 ``datasets/data.yaml`` 一致，见 ``class_order.py``）；
  - 安装 requirements.txt（含 ultralytics、torch 等）；
  - yolo11m-seg.pt 等权重文件路径正确；若 COCO 80 类与数据 4 类不一致，脚本会按 yaml 重建头再部分加载。

【运行示例】
  cd solution
  python train.py --data-dir ../data/processed --student-weights yolo11m-seg.pt --imgsz 1024 --distill
================================================================================
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 把「本文件所在文件夹」加入 Python 路径，这样无论从哪启动都能找到同目录模块
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_seg import default_train_transforms, split_stems
from dataset_seg_yolo import SemanticMaskYoloSegDataset, collate_seg_yolo
from distill_modules import AdaptiveTeacherFusion, FeatureDistillLoss, StudentChannelAlign
from models.teacher_vit import build_teacher


def set_seed(seed: int) -> None:
    """统一随机种子：让「数据集划分、Dropout、部分 CUDA 算子」尽量可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yolo_seg_student(weights: str, nc: int, device: torch.device):
    """
    加载 Ultralytics 的 SegmentationModel。

    nc：YOLO 前景类数（不含背景）。若 checkpoint 里类别数与 nc 不同，
    会用「同名 yaml」重建网络再 load 权重（形状对得上的层会拷贝，检测/分割头可能对不齐而随机初始化）。
    """
    from ultralytics import YOLO
    from ultralytics.nn.tasks import SegmentationModel, torch_safe_load
    from ultralytics.utils import DEFAULT_CFG

    wpath = Path(weights)
    yo = YOLO(str(wpath))
    student = yo.model
    head = student.model[-1]
    if getattr(head, "nc", nc) != nc:
        # 与 COCO 等预训练类数不一致时：用 yolo11m-seg.pt → yolo11m-seg.yaml（会解析到 yolo11-seg + scale m）
        cfg_name = wpath.stem + ".yaml"
        student = SegmentationModel(cfg_name, ch=3, nc=nc, verbose=False)
        # 新版 ultralytics：BaseModel.load 只接受「checkpoint 字典（含 model）」或 nn.Module，不能传路径字符串
        ckpt, _ = torch_safe_load(str(wpath))
        student.load(ckpt)

    # v8SegmentationLoss 会读 model.args 里的 box/cls/dfl、overlap_mask 等
    student.args = deepcopy(DEFAULT_CFG)
    student.args.overlap_mask = True
    student.task = getattr(student, "task", None) or "segment"
    student.to(device)
    student.train()
    return student


def infer_neck_out_channels(student: torch.nn.Module, device: torch.device, imgsz: int) -> Tuple[int, int, int]:
    """
    通过一次「假输入」前向，在 Segment 模块的 hook 里读取 neck 输给头的三个尺度通道数。
    StudentChannelAlign 的 in_channels 必须与此一致。
    """
    captured: List[int] = []

    def _hook(_module, inputs, _output):
        if not inputs:
            return
        x0 = inputs[0]
        if isinstance(x0, (list, tuple)) and len(x0) == 3:
            captured.clear()
            captured.extend(int(t.shape[1]) for t in x0)

    seg = student.model[-1]
    h = seg.register_forward_hook(_hook)
    try:
        student.train()
        with torch.no_grad():
            student(torch.zeros(1, 3, imgsz, imgsz, device=device))
    finally:
        h.remove()
    if len(captured) != 3:
        raise RuntimeError(f"无法推断 neck 三尺度通道: {captured}")
    return captured[0], captured[1], captured[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO-seg + optional ViT distillation")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--student-weights", type=str, default="yolo11m-seg.pt")
    p.add_argument("--output-dir", type=Path, default=Path("runs/seg_yolo_train"))
    p.add_argument("--num-classes", type=int, default=5, help="语义类别数含背景；YOLO nc = 此值-1")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--distill", action="store_true")
    p.add_argument("--lambda-feat", type=float, default=0.5)
    p.add_argument("--lambda-cls", type=float, default=1.0, help="YOLO 总损失向量的系数（内部已含 box/seg/cls/dfl 增益）")
    p.add_argument("--lambda-l1", type=float, default=0.5)
    p.add_argument("--lambda-l2", type=float, default=0.3)
    p.add_argument("--lambda-l3", type=float, default=0.2)
    p.add_argument(
        "--teacher-weights",
        type=str,
        default="",
        help="教师 HF 快照目录（须含 config.json）；留空则用 solution/weights/dinov3-vitb16-pretrain-lvd1689m",
    )
    p.add_argument("--teacher-img-size", type=int, default=1024)
    p.add_argument("--teacher-no-pretrained", action="store_true")
    p.add_argument("--student-feat-channels", type=str, default="", help="不填则自动推断 P3,P4,P5 通道")
    return p.parse_args()


def _resize_batch_img(batch: dict, imgsz: int) -> None:
    """把 batch 里的图像和实例 id 掩膜统一缩放到 imgsz×imgsz（与训练输入一致）。"""
    x = batch["img"]
    if x.shape[-1] != imgsz or x.shape[-2] != imgsz:
        batch["img"] = F.interpolate(x, size=(imgsz, imgsz), mode="bilinear", align_corners=False)
    m = batch["masks"]
    if m.shape[-1] != imgsz or m.shape[-2] != imgsz:
        batch["masks"] = F.interpolate(m.unsqueeze(1), size=(imgsz, imgsz), mode="nearest").squeeze(1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    img_dir = args.data_dir / "images"
    msk_dir = args.data_dir / "masks"
    if not img_dir.is_dir() or not msk_dir.is_dir():
        raise FileNotFoundError(f"需要 {img_dir} 与 {msk_dir}")

    nc_yolo = args.num_classes - 1
    if nc_yolo < 1:
        raise ValueError("--num-classes 至少为 2")

    train_stems, val_stems = split_stems(img_dir, val_ratio=args.val_ratio, seed=args.seed)
    train_ds = SemanticMaskYoloSegDataset(
        img_dir, msk_dir, image_list=train_stems, transform=default_train_transforms(), num_classes=args.num_classes
    )
    val_list = val_stems if val_stems else train_stems[: max(1, len(train_stems) // 5)]
    val_ds = SemanticMaskYoloSegDataset(img_dir, msk_dir, image_list=val_list, transform=None, num_classes=args.num_classes)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_seg_yolo, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_seg_yolo, pin_memory=device.type == "cuda",
    )

    student = load_yolo_seg_student(args.student_weights, nc_yolo, device)

    from ultralytics.utils.loss import v8SegmentationLoss

    criterion = v8SegmentationLoss(student)

    if args.student_feat_channels.strip():
        c3, c4, c5 = [int(x.strip()) for x in args.student_feat_channels.split(",")]
    else:
        c3, c4, c5 = infer_neck_out_channels(student, device, args.imgsz)

    # 768 = ViT-B 类模型的 hidden_size；把学生通道投影到与教师同一维度再算 L1
    align = StudentChannelAlign(in_channels=(c3, c4, c5), out_channels=768).to(device)
    feat_loss_fn = FeatureDistillLoss(lambdas=(args.lambda_l1, args.lambda_l2, args.lambda_l3)).to(device)

    teacher = None
    fusion = None
    _student_feats: Optional[List[torch.Tensor]] = None

    def _seg_hook(_module, inputs, _output):
        nonlocal _student_feats
        if not inputs:
            return
        x0 = inputs[0]
        if isinstance(x0, (list, tuple)) and len(x0) == 3:
            _student_feats = list(x0)

    hook_handle = student.model[-1].register_forward_hook(_seg_hook)

    if args.distill:
        tw = args.teacher_weights.strip() or None
        teacher = build_teacher(
            img_size=args.teacher_img_size,
            pretrained=not args.teacher_no_pretrained,
            weights_dir=tw,
        ).to(device)
        fusion = AdaptiveTeacherFusion(channels=teacher.embed_dim).to(device)

    params = list(student.parameters()) + list(align.parameters())
    if fusion is not None:
        params += list(fusion.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    try:
        for epoch in range(1, args.epochs + 1):
            student.train()
            align.train()
            if fusion is not None:
                fusion.train()

            running_yolo, running_feat, n_batches = 0.0, 0.0, 0

            for batch in train_loader:
                for k in ("img", "masks", "batch_idx", "cls", "bboxes"):
                    batch[k] = batch[k].to(device, non_blocking=True)
                _resize_batch_img(batch, args.imgsz)

                optimizer.zero_grad(set_to_none=True)

                # 训练态 Segment 返回 (各尺度特征图列表, mask 系数, proto)，供 criterion 解算损失
                preds = student(batch["img"])
                loss_vec, _ = criterion(preds, batch)
                loss_yolo = loss_vec.sum()

                loss_feat = torch.tensor(0.0, device=device)
                if args.distill and teacher is not None and fusion is not None and _student_feats is not None:
                    p3, p4, p5 = _student_feats
                    s3, s4, s5 = align(p3, p4, p5)
                    target_sizes = [(s3.shape[2], s3.shape[3]), (s4.shape[2], s4.shape[3]), (s5.shape[2], s5.shape[3])]
                    x_t = F.interpolate(
                        batch["img"], size=(args.teacher_img_size, args.teacher_img_size),
                        mode="bilinear", align_corners=False,
                    )
                    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=x_t.dtype).view(1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=x_t.dtype).view(1, 3, 1, 1)
                    x_t = (x_t - mean) / std
                    with torch.no_grad():
                        t_feats = teacher(x_t)
                    t1, t2, t3 = fusion(t_feats, target_sizes)
                    loss_feat = feat_loss_fn([s3, s4, s5], [t1, t2, t3])

                loss = args.lambda_cls * loss_yolo + args.lambda_feat * loss_feat
                loss.backward()
                optimizer.step()

                running_yolo += float(loss_yolo.detach())
                running_feat += float(loss_feat.detach())
                n_batches += 1

            # 验证：仍用 train 模式跑前向，否则 Segment 推理分支与 loss 不兼容
            align.eval()
            if fusion is not None:
                fusion.eval()
            val_losses = []
            student.train()
            with torch.no_grad():
                for batch in val_loader:
                    for k in ("img", "masks", "batch_idx", "cls", "bboxes"):
                        batch[k] = batch[k].to(device, non_blocking=True)
                    _resize_batch_img(batch, args.imgsz)
                    preds = student(batch["img"])
                    lv, _ = criterion(preds, batch)
                    val_losses.append(float(lv.sum().cpu()))

            mean_val = float(np.mean(val_losses)) if val_losses else 0.0
            print(
                f"Epoch {epoch}/{args.epochs}  train_yolo={running_yolo/max(n_batches,1):.4f}  "
                f"train_feat={running_feat/max(n_batches,1):.4f}  val_yolo={mean_val:.4f}"
            )

            ckpt = {
                "epoch": epoch,
                "student": student.state_dict(),
                "align": align.state_dict(),
                "args": vars(args),
                "neck_channels": (c3, c4, c5),
            }
            if fusion is not None:
                ckpt["fusion"] = fusion.state_dict()
            torch.save(ckpt, args.output_dir / "last.pt")
            if mean_val <= best_val:
                best_val = mean_val
                torch.save(ckpt, args.output_dir / "best.pt")

    finally:
        hook_handle.remove()

    (args.output_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    print(f"训练结束。最佳 val_yolo_loss≈{best_val:.4f}，权重目录: {args.output_dir}")


if __name__ == "__main__":
    main()
