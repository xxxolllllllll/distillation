from __future__ import annotations

"""
YOLOv11-m 检测端：**分类（BCE，与 Ultralytics TaskAligned 分配一致）+ NWD 定位损失 + DFL +
特征蒸馏**（教师 ViT + 学生 YOLO）。

总损失：λ_cls·L_cls + λ_loc·L_NWD + λ_feat·L_feat + λ_dfl·L_dfl（权重见 ``--lambda-*`` 与 ``DistillLossConfig``）。

数据输入：
  使用 pipeline.py 生成的 patch 数据：
    <data_dir>/images/*.png
    <data_dir>/masks/*.png

掩膜类别 id 默认（与 ``datasets/data.yaml`` 的 names 顺序一致，详见 ``class_order.py``）：
  0=background, 1=crack, 2=decay, 3=defect(孔洞/钉眼), 4=knot → YOLO class 0..3

本训练脚本会从 mask 自动提取连通域并生成 YOLO 检测框标签（cxcywh 归一化）。

与实验方案一致：
  - **学生 YOLO 输入**固定为 ``1024×1024``（由 ``--imgsz`` 控制，默认 1024），与 pipeline 输出的 patch 一致；
  若个别图像尺寸不一致，会在训练时双线性缩放到该尺寸，标注为归一化坐标故仍有效。
  - **教师 ViT 输入**为 ``--teacher-img-size``（默认 1024），仅用于教师支路特征提取。
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys

# 从项目根目录或其它工作目录直接 `python solution/train_detect_distill.py` 时，
# 需把本文件所在目录加入 sys.path，才能 import 同目录下的 detect_* / dataset_* 等模块。
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from class_order import MASK_CRACK, MASK_DECAY, MASK_DEFECT, MASK_KNOT
from detect_nwd_distill_loss import DistillLossConfig, NWDDetectionDistillLoss
from dataset_det_from_masks import MaskToYoloDetDataset, collate_det
from distill_modules import AdaptiveTeacherFusion, FeatureDistillLoss, StudentChannelAlign
from models.teacher_vit import build_teacher


def set_seed(seed: int) -> None:
    """固定 Python / NumPy / PyTorch 随机种子，便于划分与训练可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """解析命令行；检测损失权重与 `detect_nwd_distill_loss.DistillLossConfig` 一一对应。"""
    p = argparse.ArgumentParser(
        description="YOLOv11-m: cls(BCE) + NWD + DFL + feature distill (ViT teacher + YOLO student)."
    )
    p.add_argument("--data-dir", type=Path, required=True, help="pipeline 输出根目录，包含 images/ 与 masks/")
    p.add_argument("--student-weights", type=str, default="yolo11m.pt", help="需要与 nc(类别数)匹配的检测权重")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--imgsz", type=int, default=1024, help="patch 输入尺寸，应与 pipeline 输出一致")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)

    # 掩膜语义类别数（含背景 0）；与 pipeline 导出的 mask 一致
    p.add_argument("--num-mask-classes", type=int, default=5, help="mask 包含背景：0..4 共5类")
    p.add_argument("--crack-id", type=int, default=MASK_CRACK, help="语义 mask 中裂缝 id，默认与 data.yaml 第 0 类对齐")
    p.add_argument("--decay-id", type=int, default=MASK_DECAY, help="腐朽 id，对应 data.yaml 第 1 类")
    p.add_argument("--defect-id", type=int, default=MASK_DEFECT, help="缺损 id，对应 data.yaml 第 2 类")
    p.add_argument("--knot-id", type=int, default=MASK_KNOT, help="节子 id，对应 data.yaml 第 3 类")
    # YOLO 检测头类别数（不含背景）；须与 student 权重里的 det.nc 一致
    p.add_argument("--num-yolo-classes", type=int, default=4)

    # 本地 HF 快照目录；与 DistillLossConfig 中教师 resize 的 img_size 一致
    p.add_argument(
        "--teacher-weights",
        type=str,
        default="",
        help="教师 HF 快照目录（须含 config.json）；留空则用 solution/weights/dinov3-vitb16-pretrain-lvd1689m",
    )
    p.add_argument("--teacher-img-size", type=int, default=1024)

    # 总损失：λ_cls·L_cls + λ_loc·L_NWD + λ_feat·L_feat + λ_dfl·L_dfl（见 NWDDetectionDistillLoss）
    p.add_argument("--lambda-cls", type=float, default=1.0)
    p.add_argument("--lambda-loc", type=float, default=1.0)  # NWD
    p.add_argument("--lambda-feat", type=float, default=0.5)
    p.add_argument("--lambda-dfl", type=float, default=1.0)
    # L_feat 内对 P3/P4/P5 三层对齐项的相对权重（FeatureDistillLoss）
    p.add_argument("--lambda-l1", type=float, default=0.5)
    p.add_argument("--lambda-l2", type=float, default=0.3)
    p.add_argument("--lambda-l3", type=float, default=0.2)

    # NWD 损失超参 C（NWDLoss）
    p.add_argument("--nwd-c", type=float, default=0.7)

    # 须与 YOLO neck 输出到 Detect 前的通道一致（yolo11m 常见为 192,384,512）
    p.add_argument("--student-feat-channels", type=str, default="192,384,512", help="格式: c3,c4,c5")

    # TaskAlignedAssigner：每个 gt 最多匹配的 anchor 数
    p.add_argument("--tal-topk", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)

    # pipeline 标准输出：images 为 RGB patch，masks 为同名的整型语义掩膜
    images_dir = args.data_dir / "images"
    masks_dir = args.data_dir / "masks"
    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError(f"Expect {images_dir} and {masks_dir} to exist.")

    # 以 images 下的文件名 stem 为准，与 masks 按同名配对
    stems = sorted({p.stem for p in images_dir.glob("*.png")})
    if not stems:
        raise FileNotFoundError(f"No images found under {images_dir}.")

    # 简易随机划分：≥5 张时约 8:2；样本过少时不划验证集（val_stems 可能为空，当前脚本未跑验证）
    rng = random.Random(args.seed)
    rng.shuffle(stems)
    n_val = max(1, int(0.2 * len(stems))) if len(stems) >= 5 else 0
    train_stems = stems[n_val:]
    val_stems = stems[:n_val]

    # 参与「转检测框」的 mask 前景类：连通域 -> YOLO 框；顺序决定 yolo class id 0..3
    cls_ids = [args.crack_id, args.decay_id, args.defect_id, args.knot_id]

    # __getitem__ 返回 dict：img, batch_idx, cls, bboxes（归一化 cxcywh），供 Ultralytics 式损失使用
    train_ds = MaskToYoloDetDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        stems=train_stems,
        num_classes=args.num_mask_classes,
        cls_ids=cls_ids,
        cls_offset=1,  # mask id 1..4 -> yolo class id 0..3
    )
    # collate_det：把变长目标拼成单 batch 张量
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_det,
        pin_memory=(device.type == "cuda"),
    )

    # Ultralytics YOLO：加载预训练检测权重；student 即 nn.Module，forward 输入 [B,3,H,W]
    from ultralytics import YOLO

    yolo = YOLO(args.student_weights)
    student = yolo.model
    student.to(device)
    student.train()

    # 最后一层一般为 Detect：内含 nc、stride、reg_max；与我们的标签类别数必须一致
    det = student.model[-1]
    if getattr(det, "nc", None) != args.num_yolo_classes:
        raise ValueError(
            f"Student model nc={det.nc} != args.num_yolo_classes={args.num_yolo_classes}.\n"
            f"Please use a checkpoint trained for {args.num_yolo_classes} classes (no background), "
            f"or reconfigure the model head."
        )

    # 教师：仅前向取多层特征，参数在 NWDDetectionDistillLoss 内 requires_grad=False
    tw = args.teacher_weights.strip() or None
    teacher = build_teacher(
        img_size=args.teacher_img_size,
        pretrained=True,
        weights_dir=tw,
    ).to(device)

    # 学生 P3/P4/P5 -> 768 维，与 ViT hidden 对齐；fusion 融合教师相邻 block 再与学生对齐尺寸算 L_feat
    c3, c4, c5 = [int(x.strip()) for x in args.student_feat_channels.split(",")]
    align = StudentChannelAlign(in_channels=(c3, c4, c5), out_channels=768).to(device)
    fusion = AdaptiveTeacherFusion(channels=768).to(device)
    feat_loss_fn = FeatureDistillLoss(lambdas=(args.lambda_l1, args.lambda_l2, args.lambda_l3)).to(device)

    # 封装：TaskAligned 分配 + 分类 BCE + NWD 框损失 + DFL + 特征蒸馏（hook 抓 Detect 输入 P3/P4/P5）
    loss_cfg = DistillLossConfig(
        lambda_cls=args.lambda_cls,
        lambda_loc=args.lambda_loc,
        lambda_feat=args.lambda_feat,
        lambda_dfl=args.lambda_dfl,
        nwd_c=args.nwd_c,
        tal_topk=args.tal_topk,
        teacher_img_size=args.teacher_img_size,
    )

    criterion = NWDDetectionDistillLoss(
        student_model=student,
        teacher_model=teacher,
        student_align=align,
        teacher_fusion=fusion,
        feat_loss_fn=feat_loss_fn,
        cfg=loss_cfg,
    )

    # 只更新学生与对齐/融合模块；教师不参与反传
    optim_params = list(student.parameters()) + list(align.parameters()) + list(fusion.parameters())
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)

    # 混合精度：CUDA 上启用可省显存；CPU 上 GradScaler 自动不生效
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    print(f"学生网络输入尺寸: {args.imgsz}×{args.imgsz}；教师支路输入: {args.teacher_img_size}×{args.teacher_img_size}")

    for epoch in range(1, args.epochs + 1):
        running = {"loc": 0.0, "cls": 0.0, "dfl": 0.0, "total": 0.0, "feat": 0.0}
        n_batches = 0

        student.train()
        align.train()
        fusion.train()

        for batch in train_loader:
            # 与 DataLoader 约定一致：tensor 搬到目标设备
            for k, v in batch.items():
                batch[k] = v.to(device, non_blocking=True)

            # 实验方案：学生模型输入统一为 imgsz×imgsz（默认 1024×1024）
            # 标签为相对整图的归一化坐标，与图像同比例缩放后仍对应同一几何位置
            h, w = batch["img"].shape[-2], batch["img"].shape[-1]
            if h != args.imgsz or w != args.imgsz:
                batch["img"] = F.interpolate(
                    batch["img"],
                    size=(args.imgsz, args.imgsz),
                    mode="bilinear",
                    align_corners=False,
                )

            optimizer.zero_grad(set_to_none=True)

            # preds：Ultralytics 习惯为 (loss_items, feats) 或依赖子模块；criterion 内按与 v8DetectionLoss 相同方式解析
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                preds = student(batch["img"])
                loss_total, loss_items = criterion(preds, batch)

            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()

            # loss_items 顺序：[L_NWD, L_cls, L_dfl]；L_feat 已加在 loss_total 中，此处未单独打印
            loc, cls, dfl = loss_items.detach().cpu().tolist()
            running["loc"] += loc
            running["cls"] += cls
            running["dfl"] += dfl
            running["total"] += float(loss_total.detach().cpu())
            n_batches += 1

        msg = (
            f"Epoch {epoch}/{args.epochs}: "
            f"total={running['total']/max(n_batches,1):.4f}, "
            f"loc(NWD)={running['loc']/max(n_batches,1):.4f}, "
            f"cls={running['cls']/max(n_batches,1):.4f}, "
            f"dfl={running['dfl']/max(n_batches,1):.4f}"
        )
        print(msg)


if __name__ == "__main__":
    main()

