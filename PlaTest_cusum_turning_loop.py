"""
主程序入口 - 支持循環執行分析和重新調整參數
此文件包裝原始的 PlaTest_cusum_turning.py，允許用戶在分析完成後重新調整參數並再次執行分析。
"""

import subprocess
import sys
import os

def run_analysis():
    """執行分析程式"""
    script_path = os.path.join(os.path.dirname(__file__), "PlaTest_cusum_turning.py")
    try:
        result = subprocess.run([sys.executable, script_path], check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"執行分析程式時出錯: {e}")
        return False

def main():
    """主循環 - 允許用戶重複執行分析"""
    print("\n" + "="*62)
    print("  CUSUM 壓縮分析 - 循環執行模式")
    print("="*62)
    
    first_run = True
    while True:
        if not first_run:
            print("\n" + "="*62)
            choice = input("分析已完成。是否要調整參數後重新分析？(y/n): ").strip().lower()
            if choice not in ("y", "yes", "1", "是", ""):
                print("\n程式已結束。")
                break
            print("="*62)
        
        first_run = False
        
        # 執行分析程式
        if not run_analysis():
            print("\n分析程式執行失敗，請檢查是否有錯誤。")
            choice = input("是否重新嘗試？(y/n): ").strip().lower()
            if choice not in ("y", "yes", "1", "是", ""):
                break

if __name__ == "__main__":
    main()
