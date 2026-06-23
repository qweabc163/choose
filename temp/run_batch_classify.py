# run_batch.py
import subprocess
import sys
import argparse
from datetime import datetime, timedelta

def main():
    parser = argparse.ArgumentParser(description="批量执行 pool03.py --classify-only")
    parser.add_argument("--start", required=True, help="起始日期，格式 YYYY-MM-DD")
    parser.add_argument("--days", type=int, required=True, help="连续天数（包含起始日）")
    args = parser.parse_args()

    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    except ValueError:
        print("日期格式错误，请使用 YYYY-MM-DD")
        sys.exit(1)

    for i in range(args.days):
        current_date = start_date + timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
        cmd = ["python", "pool03.py", "--date", date_str, "--classify-only"]
        print(f"\n{'='*40}")
        print(f"执行: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"命令执行失败，返回码 {e.returncode}")
        except FileNotFoundError:
            print("未找到 python 或 pool03.py，请检查环境")
            sys.exit(1)

if __name__ == "__main__":
    main()