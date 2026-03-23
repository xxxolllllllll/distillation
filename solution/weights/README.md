# weights — 本地大文件（教师等）

- **DINOv3 ViT-B/16（HF）**：目录名 `dinov3-vitb16-pretrain-lvd1689m/`  
  由上级目录执行：

  ```bash
  cd ..
  python scripts/download_dinov3_teacher.py
  ```

- 下载成功后应包含 **`config.json`** 与权重（如 `model.safetensors`）。  
- `models/teacher_vit.build_teacher` **默认**使用该目录；也可用 `--teacher-weights` 指向其它本地快照。
