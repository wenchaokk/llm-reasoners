"""实时监控脚本 - 自动刷新显示"""

import os
import sys
import time

_root = os.path.dirname(__file__)
sys.path.insert(0, _root)

from reasoners.world_model import MemoryStore

def live_monitor():
    """实时监控，自动刷新"""
    memory_file = os.path.join(_root, "world_model_memory.json")
    target = 10000
    refresh_interval = 3
    
    print("="*60)
    print("实时监控经验生成进度")
    print("="*60)
    print(f"目标: {target:,} 条经验")
    print(f"刷新间隔: {refresh_interval} 秒")
    print("按 Ctrl+C 停止监控")
    print("="*60 + "\n")
    
    last_count = 0
    start_time = time.time()
    
    try:
        while True:
            if os.path.exists(memory_file):
                try:
                    memory_store = MemoryStore(memory_file=memory_file)
                    current_count = len(memory_store.memories)
                    
                    # 计算进度
                    progress = current_count / target * 100
                    remaining = target - current_count
                    
                    # 计算速度和预计时间
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        speed = current_count / elapsed
                        if remaining > 0 and speed > 0:
                            eta_seconds = remaining / speed
                            eta_min = int(eta_seconds // 60)
                            eta_sec = int(eta_seconds % 60)
                            eta_str = f"{eta_min}分{eta_sec}秒"
                        else:
                            eta_str = "计算中..."
                    else:
                        speed = 0
                        eta_str = "计算中..."
                    
                    # 计算增量
                    delta = current_count - last_count
                    
                    # 清屏（Windows）
                    os.system('cls')
                    
                    # 显示信息
                    print("="*60)
                    print("实时监控经验生成进度")
                    print("="*60)
                    print(f"目标: {target:,} 条经验")
                    print(f"刷新间隔: {refresh_interval} 秒")
                    print("按 Ctrl+C 停止监控")
                    print("="*60 + "\n")
                    
                    print(f"当前经验数: {current_count:,} / {target:,}")
                    print(f"进度: {progress:.1f}%")
                    print(f"剩余: {remaining:,} 条")
                    print(f"平均速度: {speed:.1f} 条/秒")
                    print(f"预计剩余时间: {eta_str}")
                    
                    if delta > 0:
                        print(f"本次刷新新增: +{delta} 条")
                    
                    # 进度条
                    bar_width = 50
                    filled = int(bar_width * progress / 100)
                    bar = "=" * filled + "-" * (bar_width - filled)
                    print(f"\n进度条: [{bar}] {progress:.1f}%")
                    
                    # 如果达到目标
                    if current_count >= target:
                        print("\n" + "="*60)
                        print("[OK] 已完成！达到目标 10000 条经验")
                        print("="*60)
                        break
                    
                    last_count = current_count
                    
                except Exception as e:
                    print(f"读取错误: {e}")
                    time.sleep(refresh_interval)
                    continue
            else:
                print("等待内存文件创建...")
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\n监控已停止")
        if os.path.exists(memory_file):
            try:
                memory_store = MemoryStore(memory_file=memory_file)
                print(f"\n最终经验数: {len(memory_store.memories):,} / {target:,}")
            except:
                pass

if __name__ == "__main__":
    live_monitor()
