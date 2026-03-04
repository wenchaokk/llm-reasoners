"""验证经验脚本

使用方法：
python validate_experiences.py --memory_file world_model_memory.json
python validate_experiences.py --memory_file world_model_memory.json --verbose
"""

import argparse
from reasoners.world_model.experience_validator import validate_experiences


def main():
    parser = argparse.ArgumentParser(description="验证 WorldModel 经验的准确性")
    parser.add_argument(
        "--memory_file",
        type=str,
        default="world_model_memory.json",
        help="内存文件路径"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细信息"
    )
    
    args = parser.parse_args()
    
    validate_experiences(memory_file=args.memory_file, verbose=args.verbose)


if __name__ == "__main__":
    main()
