"""
【工具】下载门控模型 facebook/dinov3-vitb16-pretrain-lvd1689m 到项目内 ``solution/weights/``。

在 **solution** 目录下执行（需已登录 Hub，见 scripts/hf_login.py）::

    python scripts/download_dinov3_teacher.py

权重目录与 ``models/teacher_vit.default_teacher_weights_dir()`` / ``build_teacher`` 的默认路径一致。
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
_LOCAL_DIRNAME = "dinov3-vitb16-pretrain-lvd1689m"


def main() -> None:
    # 本文件在 solution/scripts/，上一级是 solution/
    solution_root = Path(__file__).resolve().parent.parent
    out_dir = solution_root / "weights" / _LOCAL_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import get_token, snapshot_download
    except ImportError:
        print("请先安装: pip install huggingface_hub", file=sys.stderr)
        raise SystemExit(1) from None

    if not get_token():
        print(
            "未检测到登录。请先执行:\n  python scripts/hf_login.py\n"
            "或设置环境变量 HF_TOKEN 后再运行本脚本。",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"正在下载 {_REPO_ID} -> {out_dir}")
    snapshot_download(repo_id=_REPO_ID, local_dir=str(out_dir), token=True)
    cfg = out_dir / "config.json"
    if not cfg.is_file():
        print(f"下载可能不完整，缺少 {cfg}", file=sys.stderr)
        raise SystemExit(2)
    print(f"完成: {out_dir.resolve()}")
    print("训练时 build_teacher 默认从该目录加载（须含 config.json）。")


if __name__ == "__main__":
    main()
