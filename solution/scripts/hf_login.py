"""
【工具】Hugging Face 登录（不依赖 huggingface-cli 在系统 PATH 里）

在 **solution** 目录下执行::

    python scripts/hf_login.py

详见仓库 ``solution/文件说明.md``。
"""
from __future__ import annotations

import getpass
import os
import sys


def _normalize_token(raw: str) -> str:
    """去掉首尾空白、引号；去掉 token 内所有空白字符，避免复制进不可见空格。"""
    t = raw.strip().strip('"').strip("'")
    return "".join(t.split())


def main() -> None:
    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("请先执行: pip install -U huggingface_hub", file=sys.stderr)
        raise SystemExit(1) from None

    if os.environ.get("HF_ENDPOINT"):
        print(
            f"提示：已设置 HF_ENDPOINT={os.environ['HF_ENDPOINT']!r}。\n"
            "若未使用官方镜像，可能导致 whoami 400；可尝试清除该环境变量。\n",
            file=sys.stderr,
        )

    print(
        "从 https://huggingface.co/settings/tokens 复制 Token（Read 即可）。\n"
        "建议新建一条，粘贴到记事本检查后再复制进此处。\n"
    )
    raw = getpass.getpass("粘贴 Token（输入不显示）: ")
    token = _normalize_token(raw)
    if not token:
        print("Token 为空。", file=sys.stderr)
        raise SystemExit(1)

    try:
        HfApi(token=token).whoami()
    except Exception as e:
        print(f"\n校验失败: {e}\n", file=sys.stderr)
        print(
            "可尝试: pip install -U huggingface_hub httpx；或 $env:HF_TOKEN='...' 后运行下载脚本。\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    login(token=token, add_to_git_credential=False)
    print("\n已保存。下一步: python scripts/download_dinov3_teacher.py")


if __name__ == "__main__":
    main()
