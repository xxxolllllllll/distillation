# scripts — 辅助脚本（非训练核心）

在 **`solution`** 目录下执行：

```bash
python scripts/hf_login.py
python scripts/download_dinov3_teacher.py
# 从 datasets/train 划出约 15% 到 valid/、15% 到 test/（图片+标签一起移动）
python scripts/split_train_val_test.py
# YOLO txt -> 语义 mask PNG（供 pipeline --mask-dir）
python scripts/yolo_labels_to_semantic_masks.py --images-dir datasets/train/images --labels-dir datasets/train/labels --output-dir data/masks_gen
# 仅预览：python scripts/split_train_val_test.py --dry-run
```

详见 **`../文件说明.md`** 第四节、**`../datasets/README.md`**。
