# 古建木构件 · 蒸馏与检测实验代码

本目录为**实验方案**对应的可运行代码；**方案级说明**见根目录 `实验方案.md`。

## 快速导航

| 你想做的事 | 去哪里 |
|------------|--------|
| 弄清每个文件干什么、先读什么 | **[solution/文件说明.md](solution/文件说明.md)** ⭐ |
| 数据切片与 ROI | `solution/pipeline.py` |
| YOLO 检测 + NWD + 蒸馏 | `solution/train_detect_distill.py` |
| YOLO 分割 + 可选蒸馏 | `solution/train.py` |
| 下载 DINOv3 教师到本地 | `solution/scripts/` 下两个脚本 |

## 环境

```bash
cd solution
pip install -r requirements.txt
```

## 目录结构（精简）

```
distillation/
├── 实验方案.md
├── README.md                 ← 本文件
└── solution/
    ├── 文件说明.md           ← 每个源码文件的详细说明 + 阅读顺序
    ├── README_实现说明.md    ← 命令行示例与流程
    ├── requirements.txt
    ├── pipeline.py           # 数据
    ├── train.py              # 分割训练
    ├── train_detect_distill.py
    ├── dataset_*.py
    ├── distill_modules.py
    ├── detect_nwd_distill_loss.py
    ├── models/teacher_vit.py
    ├── scripts/              # 仅工具：HF 登录、下载教师
    └── weights/              # 本地大文件（默认不提交 Git）
```
