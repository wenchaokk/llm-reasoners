"""从 api_key.txt 加载 OPENAI_API_KEY（若环境变量未设置）"""
import os
import shutil

_DIR = os.path.dirname(os.path.abspath(__file__))
_API_KEY_FILE = os.path.join(_DIR, "api_key.txt")
_API_KEY_EXAMPLE = os.path.join(_DIR, "api_key.txt.example")


def load_api_key() -> bool:
    """从文件加载 API Key 到环境变量。若已设置或加载成功返回 True，否则 False。"""
    if os.getenv("OPENAI_API_KEY"):
        return True
    if not os.path.exists(_API_KEY_FILE) and os.path.exists(_API_KEY_EXAMPLE):
        shutil.copy(_API_KEY_EXAMPLE, _API_KEY_FILE)
    if os.path.exists(_API_KEY_FILE):
        with open(_API_KEY_FILE, encoding="utf-8") as f:
            key = f.read().strip().split("#")[0].strip()
        if key and not key.startswith("sk-your-"):
            os.environ["OPENAI_API_KEY"] = key
            return True
    return False
