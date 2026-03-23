# 实验方案代码实现说明

> **每个文件干什么、建议阅读顺序**：请先看 **[文件说明.md](./文件说明.md)**（带深度学习术语对照）。  
> 仓库总入口：**[../README.md](../README.md)**

本目录代码对应 `实验方案.md` 中的数据流程与蒸馏方案。

### 教师权重（DINOv3 门控模型）

在 **`solution`** 目录执行：

```bash
python scripts/hf_login.py
python scripts/download_dinov3_teacher.py
```

下载目录：`weights/dinov3-vitb16-pretrain-lvd1689m/`（训练时 `build_teacher` **仅**从此类本地目录加载）。

## 1) 数据预处理与切片

文件：`pipeline.py`

实现内容（**仅滑动窗口**，已无 ROI 分支）：
- 原图级等比例缩放（长边约束）
- 固定窗口切片（默认 `1024×1024`，可改 `patch_size` / `window_stride`）
- 越界窗口对称 padding（图像 `edge`、mask 背景常数）
- 几乎全背景的窗口按 `keep_background_ratio` 抽样保留
- 图像双线性、掩膜最近邻（仅出现在原图缩放一步）

### 运行示例

```bash
python pipeline.py ^
  --image-dir "data/raw/images" ^
  --mask-dir "data/raw/masks" ^
  --output-dir "data/processed" ^
  --max-long-edge 3000 ^
  --patch-size 1024 ^
  --window-stride 800
```

输出结构：
- `data/processed/images/*.png`
- `data/processed/masks/*.png`
- `data/processed/meta/*.json`

---

## 2) 特征对齐与蒸馏模块

文件：`distill_modules.py`

实现内容：
- `StudentChannelAlign`：学生 `P3/P4/P5` 通过 `1x1 Conv` 映射到 768 通道
- `AdaptiveTeacherFusion`：教师相邻层（3/4、7/8、11/12）先做空间对齐，再自适应加权融合
- `AdaptiveTeacherFusion` 内部空间对齐逻辑（`F.interpolate`）：
  - 教师 `3/4` 层 -> 对齐到学生 `P3` 的空间尺寸（如 `128x128`）
  - 教师 `7/8` 层 -> 对齐到学生 `P4` 的空间尺寸（如 `64x64`，即你提到的中间层对齐）
  - 教师 `11/12` 层 -> 对齐到学生 `P5` 的空间尺寸（如 `32x32`）
  - 代码入口：`AdaptiveTeacherFusion.forward(teacher_feats, target_sizes)`，其中 `target_sizes` 应传入 `[(H3,W3), (H4,W4), (H5,W5)]`
- `FeatureDistillLoss`：分层 SmoothL1 蒸馏损失（默认 `0.5/0.3/0.2`）
- `NWDLoss`：归一化 Wasserstein 定位损失
- `total_loss`：总损失聚合（默认 `1.0, 1.0, 0.5, 1.0`）

可将这些模块接入你现有 YOLOv11-m 训练流程中（Ultralytics 自定义 trainer 或独立训练脚本）。

---

## 3) 模型蒸馏训练（YOLOv11-m 检测端：分类 + NWD + DFL + 特征蒸馏）

新增训练入口文件：`train_detect_distill.py`、以及检测数据集/损失实现：

- `dataset_det_from_masks.py`：从 `pipeline` 生成的 `images/` + `masks/` 在线提取连通域框，生成 YOLO 检测标签。
- `detect_nwd_distill_loss.py`：在保留 **分类 BCE** 与 **DFL** 的前提下，将框回归 **CIoU 换为 NWD**，并加入多尺度 **特征蒸馏**；总损失为 ``λ_cls·L_cls + λ_loc·L_NWD + λ_feat·L_feat + λ_dfl·L_dfl``。

**与实验方案输入尺寸一致**：学生 YOLO 输入固定为 **1024×1024**（`--imgsz`，默认 1024），与 pipeline 默认 `patch-size` 一致；若磁盘上 patch 尺寸不同会双线性缩放到该边长（框为归一化坐标）。**教师 ViT 支路**默认也使用 **1024×1024**（`--teacher-img-size` 默认 1024），用于教师特征提取。

训练命令示例（需要你的检测权重与 `--num-yolo-classes` 匹配）：

```bash
python train_detect_distill.py --data-dir ../data/processed --student-weights yolo11m.pt --epochs 50 --batch-size 2 --imgsz 1024
```

说明：脚本会在检测损失中自动冻结 ViT 教师，并对 `StudentChannelAlign` 与 `AdaptiveTeacherFusion` 的参数进行优化。

---

## 3) 模型训练（YOLO-seg + 可选特征蒸馏）

文件：`train.py`、`dataset_seg_yolo.py`、`dataset_seg.py`（增强与划分）、`models/teacher_vit.py`

- 读取 `pipeline` 的 `images/`、`masks/`（语义 id 图，默认 1024×1024 patch）。
- `dataset_seg_yolo.py`：按连通域把语义 mask 转为 Ultralytics **实例分割** 监督（`batch_idx` / `cls` / `bboxes` / `overlap_mask` 合并 id 图）。
- 学生：**Ultralytics `SegmentationModel`**（默认权重 `yolo11m-seg.pt`）；损失为 **`v8SegmentationLoss`**（box + seg + cls + dfl）。`--num-classes` 为语义类别数（含背景），YOLO 的 `nc = num_classes - 1`。
- neck 通道可由脚本按输入尺寸 **自动推断**（或 `--student-feat-channels`）；蒸馏仍用 `StudentChannelAlign` → 768 维对齐教师。
- `--distill`：冻结 ViT 教师（见 ``build_teacher``，仅本地 HF 快照），`AdaptiveTeacherFusion` + `FeatureDistillLoss`。

### 依赖

蒸馏教师使用 **transformers** 从**本地目录**加载 DINOv3 ViT-B/16 快照（与 ``requirements.txt`` 中 ``transformers`` / ``huggingface_hub`` 一致）。**首次获取权重**：须在 Hub 同意条款后 ``huggingface-cli login`` 或设置 ``HF_TOKEN``，再在 ``solution`` 下执行::

    python scripts/download_dinov3_teacher.py

权重保存至 ``solution/weights/dinov3-vitb16-pretrain-lvd1689m/``；``build_teacher`` **只**读取该目录（或 ``--teacher-weights`` 指定的其它含 ``config.json`` 的目录）。大文件目录已写入仓库根 ``.gitignore``。

```bash
pip install -r requirements.txt
```

### 运行示例

```bash
# YOLO-seg（需本机存在 yolo11m-seg.pt 或改 --student-weights）
python train.py --data-dir data/processed --student-weights yolo11m-seg.pt --imgsz 1024 --epochs 50 --batch-size 2 --output-dir runs/yolo_seg

# + 特征蒸馏
python train.py --data-dir data/processed --student-weights yolo11m-seg.pt --imgsz 1024 --epochs 50 --batch-size 2 --distill --lambda-feat 0.5 --output-dir runs/yolo_seg_distill
```

输出：`output-dir/last.pt`、`best.pt`（含 Ultralytics 学生 `state_dict`、`align`，蒸馏时含 `fusion`；并保存 `neck_channels`）。

**说明**：若 COCO 预训练权重的 `nc` 与数据不一致，脚本会用同名 `yolo11m-seg.yaml` 以目标 `nc` 重建模型再 `load`（骨干尽量加载，分割头可能重新初始化）。专利中的 **YOLO 检测 + NWD** 见 `train_detect_distill.py`。
