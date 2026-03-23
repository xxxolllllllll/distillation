from __future__ import annotations

"""
================================================================================
【检测损失封装】在 Ultralytics YOLO 检测模型上：NWD 换 CIoU + 特征蒸馏
================================================================================

初学者可先只看类 NWDDetectionDistillLoss.__call__ 末尾「总损失怎么加」，再回头看分配与各项。

YOLOv11-m 检测端：**分类（BCE）+ NWD 定位损失 + DFL + 多尺度特征蒸馏**（与实验方案一致）。

核心思路：
1) 复用 Ultralytics 检测流程：TaskAlignedAssigner 分配目标、**分类 BCEWithLogits**、DFL；
2) 将 bbox 回归的 CIoU 项替换为归一化 Wasserstein Distance (NWD) loss；
3) 同步计算特征蒸馏 L_feat（空间对齐 + 通道对齐 + 相邻层自适应加权融合 + SmoothL1）。

总损失：L_total = λ_cls·L_cls + λ_loc·L_NWD + λ_feat·L_feat + λ_dfl·L_dfl。

输入 batch 需要包含：
  - batch['img']: [B,3,H,W] float（trainer 习惯在 [0,1]）
  - batch['batch_idx']: [M] float 或 long（目标属于第几个样本）
  - batch['cls']: [M] float 或 long（类别 id）
  - batch['bboxes']: [M,4] float，xywh 归一化到 [0,1]

criterion(preds, batch) 返回：
  - loss_total: 标量
  - loss_items: tensor[3]，按 [box_nwd, cls, dfl] 排列（便于日志/统计）
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import DFLoss
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import TaskAlignedAssigner, bbox2dist, dist2bbox, make_anchors

from distill_modules import AdaptiveTeacherFusion, FeatureDistillLoss, NWDLoss, StudentChannelAlign


@dataclass
class DistillLossConfig:
    # Experiment weights
    lambda_cls: float = 1.0
    lambda_loc: float = 1.0  # NWD
    lambda_feat: float = 0.5
    lambda_dfl: float = 1.0

    # NWD hyperparameter C
    nwd_c: float = 0.7
    # assignment
    tal_topk: int = 10

    # teacher input
    teacher_img_size: int = 1024  # teacher input size (aligned with experiment setting)
    # ImageNet normalization for ViT teacher
    imagenet_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


class NWDDetectionDistillLoss:
    """
    为 YOLO detection 训练提供完整检测损失 + 蒸馏：
      L_total = λ_cls·L_cls(BCE) + λ_loc·L_NWD + λ_feat·L_feat + λ_dfl·L_dfl
    """

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        student_align: StudentChannelAlign,
        teacher_fusion: AdaptiveTeacherFusion,
        feat_loss_fn: FeatureDistillLoss,
        cfg: DistillLossConfig,
    ):
        self.student = student_model
        self.teacher = teacher_model
        self.align = student_align
        self.fusion = teacher_fusion
        self.feat_loss_fn = feat_loss_fn
        self.cfg = cfg

        device = next(self.student.parameters()).device
        self.device = device

        # Detect head / task hyperparams from Ultralytics model
        m = self.student.model[-1]  # Detect() module
        self.stride = m.stride
        self.nc = m.nc
        self.reg_max = m.reg_max
        self.no = m.nc + m.reg_max * 4

        # Standard YOLO components
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.assigner = TaskAlignedAssigner(topk=cfg.tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.use_dfl = self.reg_max > 1
        self.dfl_loss = DFLoss(self.reg_max) if self.use_dfl else None

        # For DFL -> decode
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=device)

        # NWD loss core (returns scalar, supports mask-weighting)
        self.nwd_loss = NWDLoss(c=cfg.nwd_c).to(device)

        # Capture student P3/P4/P5 features from Detect module inputs
        self._student_feats: Optional[Sequence[torch.Tensor]] = None
        self._register_detect_input_hook(m)

        # Ensure teacher is frozen
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    def _register_detect_input_hook(self, detect_module: nn.Module) -> None:
        """
        forward hook:
          Detect.forward(x_list) where x_list is feature maps [P3,P4,P5]
        """

        def _hook(_module, inputs, _output):
            # inputs[0] is typically a list of feature maps.
            x_list = None
            if len(inputs) >= 1:
                if isinstance(inputs[0], (list, tuple)):
                    x_list = inputs[0]
                else:
                    # some variants pass directly
                    x_list = None
            if x_list is None:
                # fallback: search all list/tuple inputs
                for it in inputs:
                    if isinstance(it, (list, tuple)) and len(it) == 3 and all(torch.is_tensor(t) for t in it):
                        x_list = it
                        break
            if x_list is not None:
                self._student_feats = x_list

        detect_module.register_forward_hook(_hook)

    def _preprocess_targets(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """
        Copy from ultralytics v8DetectionLoss.preprocess.

        targets: [nl, 6] with columns:
            batch_idx, cls, cx, cy, w, h  (cxcywh normalized)
        output: [B, max_n, 5] with columns:
            cls, x1, y1, x2, y2 (xyxy absolute pixels)
        """
        nl, ne = targets.shape
        if nl == 0:
            return torch.zeros(batch_size, 0, ne - 1, device=self.device)

        i = targets[:, 0]  # image index
        _, counts = i.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)
        out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
        for j in range(batch_size):
            matches = i == j
            if n := matches.sum():
                out[j, :n] = targets[matches, 1:]
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def _bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """
        Copy from ultralytics v8DetectionLoss.bbox_decode.
        anchor_points: [A,2] in grid units
        pred_dist: [B,A,4*reg_max]
        return pred_bboxes_xyxy_grid: [B,A,4]
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def _nwd_loc_loss(
        self,
        pred_bboxes_grid: torch.Tensor,  # [B,A,4]
        target_bboxes_pix: torch.Tensor,  # [B,A,4]
        anchor_fg_mask: torch.Tensor,  # [B,A] bool
        stride_tensor: torch.Tensor,  # [A] float
        target_scores: torch.Tensor,  # [B,A,nc] (or [B,A,1] depending)
        target_scores_sum: torch.Tensor,  # scalar tensor
        imgsz: torch.Tensor,  # [h,w] pixels
    ) -> torch.Tensor:
        """
        Compute NWD loss on positive anchors only.
        """
        if anchor_fg_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        # weight per positive anchor
        weight = target_scores.sum(-1)[anchor_fg_mask].unsqueeze(-1)  # [M,1]
        # flatten strides for positive anchors
        stride_b = stride_tensor.unsqueeze(0).expand_as(anchor_fg_mask)  # [B,A]
        stride_pos = stride_b[anchor_fg_mask]  # [M]

        pred_xyxy_pix = pred_bboxes_grid[anchor_fg_mask] * stride_pos.unsqueeze(-1)  # [M,4]
        gt_xyxy_pix = target_bboxes_pix[anchor_fg_mask]  # [M,4]

        # xyxy -> cxcywh (pixels)
        px1, py1, px2, py2 = pred_xyxy_pix.unbind(-1)
        gx1, gy1, gx2, gy2 = gt_xyxy_pix.unbind(-1)
        pcx = (px1 + px2) / 2.0
        pcy = (py1 + py2) / 2.0
        pw = (px2 - px1).clamp(min=0)
        ph = (py2 - py1).clamp(min=0)
        gcx = (gx1 + gx2) / 2.0
        gcy = (gy1 + gy2) / 2.0
        gw = (gx2 - gx1).clamp(min=0)
        gh = (gy2 - gy1).clamp(min=0)

        h_img, w_img = imgsz[0].clamp(min=1), imgsz[1].clamp(min=1)
        pred_norm = torch.stack([pcx / w_img, pcy / h_img, pw / w_img, ph / h_img], dim=-1)
        gt_norm = torch.stack([gcx / w_img, gcy / h_img, gw / w_img, gh / h_img], dim=-1)

        # NWDLoss supports weighting by mask
        loss = self.nwd_loss(pred_norm, gt_norm, mask=weight.squeeze(-1))
        return loss

    def _dfl_loss(
        self,
        pred_distri: torch.Tensor,  # [B,A,4*reg_max]
        anchor_points: torch.Tensor,  # [A,2] grid units
        target_bboxes_pix: torch.Tensor,  # [B,A,4] pixels
        stride_tensor: torch.Tensor,  # [A]
        fg_mask: torch.Tensor,  # [B,A]
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Distribution Focal Loss for positive anchors only (same as ultralytics).
        """
        if not self.use_dfl or self.dfl_loss is None:
            return torch.tensor(0.0, device=self.device)
        if fg_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)  # [M,1]
        # convert target boxes from pixels -> grid units for bbox2dist
        stride_b = stride_tensor.unsqueeze(0).expand_as(fg_mask)  # [B,A]
        target_bboxes_grid = target_bboxes_pix / stride_b.unsqueeze(-1)  # [B,A,4]
        target_ltrb = bbox2dist(anchor_points, target_bboxes_grid, self.dfl_loss.reg_max - 1)  # [B,A,4]

        pred_dist_fg = pred_distri[fg_mask].view(-1, self.dfl_loss.reg_max)  # [M*4, reg_max]
        target_ltrb_fg = target_ltrb[fg_mask]  # [M,4]

        loss_dfl = self.dfl_loss(pred_dist_fg, target_ltrb_fg) * weight  # [M,1]*[M,1]
        loss_dfl = loss_dfl.sum() / target_scores_sum
        return loss_dfl

    def _compute_feat_distill_loss(self, batch_img: torch.Tensor) -> torch.Tensor:
        """
        Compute L_feat using:
          - student features captured by detect input hook (P3/P4/P5)
          - teacher ViT features
          - align + adaptive fusion + SmoothL1
        """
        if self._student_feats is None:
            # Hook failed; return zero rather than crashing.
            return torch.tensor(0.0, device=self.device)

        # Student P3/P4/P5
        if len(self._student_feats) != 3:
            raise RuntimeError(f"Expected 3 student feature maps, got {len(self._student_feats)}")
        p3, p4, p5 = self._student_feats

        s3, s4, s5 = self.align(p3, p4, p5)
        target_sizes = [(s3.shape[2], s3.shape[3]), (s4.shape[2], s4.shape[3]), (s5.shape[2], s5.shape[3])]

        # Teacher input: resize + ImageNet normalize
        x_t = F.interpolate(batch_img, size=(self.cfg.teacher_img_size, self.cfg.teacher_img_size), mode="bilinear", align_corners=False)
        mean = torch.tensor(self.cfg.imagenet_mean, device=x_t.device, dtype=x_t.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.cfg.imagenet_std, device=x_t.device, dtype=x_t.dtype).view(1, 3, 1, 1)
        x_t = (x_t - mean) / std

        with torch.no_grad():
            t_feats = self.teacher(x_t)  # [F3,F4,F7,F8,F11,F12]

        t1, t2, t3 = self.fusion(t_feats, target_sizes)
        loss_feat = self.feat_loss_fn([s3, s4, s5], [t1, t2, t3])
        return loss_feat

    @torch.no_grad()
    def _empty_float_loss_items(self) -> torch.Tensor:
        return torch.zeros(3, device=self.device)

    def __call__(self, preds, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        preds: model(batch_img) output
        batch: dict with img, batch_idx, cls, bboxes

        Returns:
          loss_total: scalar tensor for backward
          loss_items: tensor[3] = [box_nwd, cls, dfl] (detached)
        """
        # Extract feats list (same as ultralytics v8DetectionLoss)
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # [B,A,nc]
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # [B,A,4*reg_max]

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]

        # image size (pixels)
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)  # anchor_points [A,2], stride_tensor [A]

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self._preprocess_targets(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # [B,max,1], [B,max,4]
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Decode predicted bboxes to xyxy grid coords
        pred_bboxes_grid = self._bbox_decode(anchor_points, pred_distri)  # [B,A,4] xyxy grid

        # Assign targets
        _, target_bboxes_pix, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes_grid.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = target_scores.sum().clamp(min=1.0)

        # cls loss
        loss_cls = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # box loss (NWD) + DFL
        loss_loc = torch.tensor(0.0, device=self.device)
        loss_dfl = torch.tensor(0.0, device=self.device)
        if fg_mask.sum():
            loss_loc = self._nwd_loc_loss(
                pred_bboxes_grid=pred_bboxes_grid,
                target_bboxes_pix=target_bboxes_pix,
                anchor_fg_mask=fg_mask,
                stride_tensor=stride_tensor,
                target_scores=target_scores,
                target_scores_sum=target_scores_sum,
                imgsz=imgsz,
            )
            loss_dfl = self._dfl_loss(
                pred_distri=pred_distri,
                anchor_points=anchor_points,
                target_bboxes_pix=target_bboxes_pix,
                stride_tensor=stride_tensor,
                fg_mask=fg_mask,
                target_scores=target_scores,
                target_scores_sum=target_scores_sum,
            )

        # Feature distillation loss
        loss_feat = self._compute_feat_distill_loss(batch["img"])

        # Total loss per experiment:
        # L_total = λ1*L_cls + λ2*L_loc + λ3*L_feat + λ4*L_dfl
        loss_total = (
            self.cfg.lambda_cls * loss_cls
            + self.cfg.lambda_loc * loss_loc
            + self.cfg.lambda_feat * loss_feat
            + self.cfg.lambda_dfl * loss_dfl
        )

        # loss_items for logging (detach)
        loss_items = torch.stack([loss_loc.detach(), loss_cls.detach(), loss_dfl.detach()], dim=0)
        return loss_total, loss_items

