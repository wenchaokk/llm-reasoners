"""循环生成经验直到达到 10000 条（可后台运行）

用法: python run_until_10000.py
"""
import os
import sys

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
os.chdir(_root)

from reasoners.world_model import MemoryStore
from reasoners.world_model.experience_generator import generate_experiences

def main():
    memory_file = os.path.join(_root, "world_model_memory.json")
    memory_store = MemoryStore(memory_file=memory_file, max_memory_size=20000, save_every=200)
    target = 10000
    batch = 2000
    round = 0
    while len(memory_store.memories) < target:
        round += 1
        n_before = len(memory_store.memories)
        need = target - n_before
        this_batch = min(batch, need)
        print(f"\n[Round {round}] {n_before} / {target}, generating {this_batch} (rule)...")
        generate_experiences(
            memory_store=memory_store,
            method="rule",
            num_experiences=this_batch,
            validate=True,
        )
        n_after = len(memory_store.memories)
        print(f"[Round {round}] done. Total: {n_after}")
        if n_after >= target:
            break
        if n_after == n_before:
            print("No new experiences this round, stopping.")
            break
    print("\nFinal total:", len(memory_store.memories))
    print("File:", memory_file)

if __name__ == "__main__":
    main()
