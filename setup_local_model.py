"""一键下载并配置本地小模型（GGUF），用于「本地小模型+内存」预测与对比

运行:
  python setup_local_model.py --recommend        # 根据本机配置推荐模型（不下载）
  python setup_local_model.py --auto             # 按推荐自动选择并下载
  python setup_local_model.py                    # 默认：TinyLlama 1.1B
  python setup_local_model.py --model reasoning  # 推理优化：Llama-3.2-1B-Reasoning
  python setup_local_model.py --model smol2     # 质量更好：SmolLM2-1.7B（约 1.1GB）
  python setup_local_model.py --model llama32_3b   # Llama-3.2 3B（约 2GB，4GB 显存可跑）
  python setup_local_model.py --model llama32_8b   # Llama-3.2 8B（约 5GB，Colab/8GB+ 显存）
  python setup_local_model.py --model llama31_70b  # Llama-3.1 70B（约 40GB，Colab Pro A100 推荐）

模型对比:
  - tiny（默认）: TinyLlama 1.1B，约 700MB，CPU 友好
  - reasoning:    Llama-3.2-1B 推理版
  - smol2:        SmolLM2-1.7B，约 1.1GB，4GB+ 显存
  - llama32_3b:   Llama-3.2 3B，约 2GB，4GB+ 显存
  - llama32_8b:   Llama-3.2 8B，约 5GB，8GB+ 显存
  - llama31_70b:  Llama-3.1 70B，约 40GB，Colab Pro A100 / 40GB+ 显存，效果最佳
"""

import os
import sys
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")

# 可选模型：(repo, filename, 保存到本地的文件名)
MODEL_OPTIONS = {
    "tiny": (
        "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "tinyllama-1.1b-chat.q4_k_m.gguf",
    ),
    "reasoning": (
        "mradermacher/Reasoning-Llama-3.2-1B-Instruct-v1.3-GGUF",
        "Reasoning-Llama-3.2-1B-Instruct-v1.3.Q4_K_M.gguf",
        "reasoning-llama-3.2-1b.q4_k_m.gguf",
    ),
    "smol2": (
        "unsloth/SmolLM2-1.7B-Instruct-GGUF",
        "SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
        "smollm2-1.7b-instruct.q4_k_m.gguf",
    ),
    "llama32_3b": (
        "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "llama-3.2-3b-instruct.q4_k_m.gguf",
    ),
    "llama32_8b": (
        "bartowski/Llama-3.2-8B-Instruct-GGUF",
        "Llama-3.2-8B-Instruct-Q4_K_M.gguf",
        "llama-3.2-8b-instruct.q4_k_m.gguf",
    ),
    "llama31_70b": (
        "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
        "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf",
        "llama-3.1-70b-instruct.q4_k_m.gguf",
    ),
}


def _get_gpu_vram_gb():
    """返回 NVIDIA 显卡显存（GB），无或非 NVIDIA 返回 0。"""
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return int(out.stdout.strip().split("\n")[0].strip()) / 1024.0
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            ctypes.windll.dxva2  # 触发可用性
        except Exception:
            pass
        try:
            import subprocess
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'NVIDIA' } | Select-Object -First 1).AdapterRAM"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if r.returncode == 0 and r.stdout.strip():
                b = int(r.stdout.strip())
                if b > 0:
                    return b / (1024 ** 3)
        except Exception:
            pass
    return 0.0


def _get_system_ram_gb():
    """返回系统内存约数（GB）。"""
    try:
        import ctypes
        if sys.platform == "win32":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                ]
            m = MEMORYSTATUSEX()
            m.dwLength = ctypes.sizeof(m)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return m.ullTotalPhys / (1024 ** 3)
        else:
            import resource
            # 仅部分 Unix 可用
            return 0.0
    except Exception:
        pass
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def recommend_model():
    """
    根据本机 CPU/内存/显卡 推荐模型。
    显存>=40GB（Colab Pro A100）推荐 70B；>=8GB 推荐 8B；4GB+ 推荐 smol2；否则 reasoning/tiny。
    """
    vram = _get_gpu_vram_gb()
    ram = _get_system_ram_gb()
    info = {"gpu_vram_gb": round(vram, 1), "ram_gb": round(ram, 1)}
    # 显存 >= 40GB（Colab Pro A100）：可跑 70B Q4，推荐 Llama-3.1 70B
    if vram >= 38.0:
        return "llama31_70b", {**info, "reason": "显存>=38GB（如 Colab Pro A100），推荐 Llama-3.1 70B（效果最佳）"}
    # 显存 >= 8GB（如 Colab T4 15GB）：推荐 Llama-3.2 8B
    if vram >= 8.0:
        return "llama32_8b", {**info, "reason": "显存>=8GB（如 Colab），推荐 Llama-3.2 8B（效果更好）"}
    if vram >= 4.0:
        return "smol2", {**info, "reason": "显存>=4GB，推荐 SmolLM2-1.7B（质量较好）"}
    if vram >= 2.0 or ram >= 16.0:
        return "reasoning", {**info, "reason": "推荐 Llama-3.2-1B 推理版（适合作业/状态预测）"}
    return "tiny", {**info, "reason": "资源有限，推荐 TinyLlama 1.1B（体积小、速度快）"}


def main():
    parser = argparse.ArgumentParser(description="下载本地小模型（GGUF）")
    parser.add_argument(
        "--model",
        choices=list(MODEL_OPTIONS.keys()),
        default=None,
        help="tiny / reasoning / smol2 / llama32_3b / llama32_8b / llama31_70b；与 --auto 二选一",
    )
    parser.add_argument("--recommend", action="store_true", help="仅根据本机配置推荐模型，不下载")
    parser.add_argument("--auto", action="store_true", help="根据本机配置自动选择模型并下载")
    args = parser.parse_args()

    if args.recommend:
        name, info = recommend_model()
        print("本机配置与推荐:")
        print(f"  显存: {info.get('gpu_vram_gb', '?')} GB")
        print(f"  内存: {info.get('ram_gb', '?')} GB")
        print(f"  推荐: {name} — {info.get('reason', '')}")
        print(f"\n下载命令: python setup_local_model.py --model {name}")
        return

    if args.auto:
        name, info = recommend_model()
        print(f"根据本机配置（显存 {info.get('gpu_vram_gb')}GB / 内存 {info.get('ram_gb')}GB）推荐: {name}")
        args.model = name

    if args.model is None:
        args.model = "tiny"

    HF_REPO, HF_FILENAME, LOCAL_FILENAME = MODEL_OPTIONS[args.model]
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, LOCAL_FILENAME)
    if os.path.exists(path):
        print(f"本地模型已存在: {path}")
        print(f"直接运行对比: python compare_world_model_modes.py --max_cases 15 --local_model_path \"{path}\"")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("请先安装 huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    size_hint = {"tiny": "约 700MB", "reasoning": "约 700MB", "smol2": "约 1.1GB", "llama32_3b": "约 2GB", "llama32_8b": "约 5GB", "llama31_70b": "约 40GB"}.get(args.model, "约 700MB–40GB")
    print(f"正在从 Hugging Face 下载 [{args.model}]: {HF_REPO}")
    print(f"文件: {HF_FILENAME} ({size_hint}，首次较慢，请耐心等待)")
    import shutil
    import tempfile
    final_path = os.path.join(MODELS_DIR, LOCAL_FILENAME)
    try:
        # 先下到临时目录再复制，避免 Windows 下文件占用导致 WinError 32
        with tempfile.TemporaryDirectory() as tmpdir:
            downloaded = hf_hub_download(
                repo_id=HF_REPO,
                filename=HF_FILENAME,
                local_dir=tmpdir,
            )
            if os.path.isfile(downloaded):
                shutil.copy2(downloaded, final_path)
        print(f"下载完成: {final_path}")
        print(f"\n之后运行对比（可省略 --local_model_path）:")
        print(f"  python compare_world_model_modes.py --max_cases 15 --local_model_path \"{final_path}\"")
    except Exception as e:
        print(f"下载失败: {e}")
        print(f"\n可手动打开 https://huggingface.co/{HF_REPO} 下载 {HF_FILENAME} 到 {MODELS_DIR}/")
        sys.exit(1)


if __name__ == "__main__":
    main()
