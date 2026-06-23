#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A股历史日线数据下载与缓存模块

功能：
- 单独运行：下载指定股票或全部主板股票的历史行情，并缓存到本地。
- 作为模块：调用 fetch_stock_history() 获取单只股票数据。
- 支持通过参数控制缓存策略：
    auto    : 按缓存时间判断是否重新下载
    local   : 强制使用本地缓存，不联网下载行情
    refresh : 强制重新下载，忽略缓存时间

用法示例：
    python stock_data02.py --code 600000,000001
    python stock_data02.py --code 600000 --cache-mode local
    python stock_data02.py --code 600000 --cache-mode refresh
    python stock_data02.py --all --days 1095 --workers 8 --cache-mode refresh
"""

import os
import re
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import akshare as ak
from tqdm import tqdm


# ==============================
# 工具函数
# ==============================

def normalize_code(code) -> str:
    """统一股票代码为 6 位纯数字"""
    s = str(code).strip()
    m = re.search(r"(\d{6})", s)
    if m:
        return m.group(1)
    return s.zfill(6)


def to_market_symbol(code: str) -> str:
    """仅主板：600000 -> sh600000, 000001 -> sz000001"""
    code = normalize_code(code)
    if code.startswith("60"):
        return "sh" + code
    if code.startswith("00"):
        return "sz" + code
    return code


def today_str() -> str:
    return datetime.now().strftime("%Y%m%d")


def start_date_str(history_days: int = 30) -> str:
    """根据回溯天数计算起始日期，默认30天"""
    d = datetime.now() - timedelta(days=history_days)
    return d.strftime("%Y%m%d")


def ensure_data_dir(data_dir: str = "data"):
    Path(data_dir).mkdir(parents=True, exist_ok=True)


def is_cache_valid(file_path: str, cache_valid_hours: int = 12) -> bool:
    """
    判断缓存文件是否在有效期内。

    说明：
        cache_valid_hours <= 0 表示缓存永久有效。
        仅 cache_mode='auto' 时会使用这个判断。
    """
    if not os.path.exists(file_path):
        return False

    if cache_valid_hours <= 0:
        return True

    mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
    now = datetime.now()
    return (now - mtime) < timedelta(hours=cache_valid_hours)


def read_cache(cache_path: str, min_bars: int = 1) -> Optional[pd.DataFrame]:
    """
    读取本地缓存文件。

    参数：
        cache_path: CSV缓存路径
        min_bars:   最少需要的数据条数

    返回：
        DataFrame 或 None
    """
    if not os.path.exists(cache_path):
        return None

    try:
        df = pd.read_csv(cache_path, parse_dates=["date"])
        if df is not None and not df.empty and len(df) >= min_bars:
            return df
    except Exception:
        return None

    return None


# ==============================
# 数据标准化
# ==============================

def standardize_history_df(raw_df: pd.DataFrame, code: str) -> Optional[pd.DataFrame]:
    """
    将 ak.stock_zh_a_daily 返回的数据标准化为统一格式。

    目标字段：
        date, open, close, high, low, volume, amount,
        pct_chg, change, turnover, amplitude
    """
    if raw_df is None or raw_df.empty:
        return None

    df = raw_df.copy()

    # 如果索引为日期，重置并重命名
    if "date" not in df.columns and "日期" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            first_col = df.columns[0]
            df = df.rename(columns={first_col: "date"})

    rename_map = {
        "日期": "date",
        "时间": "date",

        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",

        "成交量": "volume",
        "成交额": "amount",
        "成交金额": "amount",

        "振幅": "amplitude",
        "涨跌幅": "pct_chg",
        "涨跌额": "change",
        "换手率": "turnover",

        "date": "date",
        "open": "open",
        "close": "close",
        "high": "high",
        "low": "low",
        "volume": "volume",
        "amount": "amount",
        "turnover": "turnover",
    }

    df = df.rename(columns=rename_map)

    required_cols = ["date", "open", "close", "high", "low", "volume"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"{code} 缺少必要字段 {missing}，实际字段：{df.columns.tolist()}")
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_cols = [
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "amplitude",
        "pct_chg",
        "change",
        "turnover",
        "outstanding_share",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "open", "close", "high", "low", "volume"])
    df = df.sort_values("date").reset_index(drop=True)

    if df.empty:
        return None

    # 补充缺失的衍生字段
    if "amount" not in df.columns or df["amount"].isna().all():
        df["amount"] = df["volume"] * df["close"]

    if "change" not in df.columns or df["change"].isna().all():
        df["change"] = df["close"].diff()

    if "pct_chg" not in df.columns or df["pct_chg"].isna().all():
        df["pct_chg"] = df["close"].pct_change() * 100

    if "amplitude" not in df.columns or df["amplitude"].isna().all():
        pre_close = df["close"].shift(1)
        df["amplitude"] = np.where(
            pre_close > 0,
            (df["high"] - df["low"]) / pre_close * 100,
            np.nan,
        )

    if "turnover" not in df.columns:
        df["turnover"] = np.nan

    # 固定常用字段顺序，其他字段保留在后面
    preferred_cols = [
        "date",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "pct_chg",
        "change",
        "turnover",
        "amplitude",
    ]
    existing_preferred = [c for c in preferred_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_preferred]
    df = df[existing_preferred + other_cols]

    return df


# ==============================
# 带缓存策略的历史行情获取
# ==============================

def fetch_stock_history(
    code: str,
    history_days: int = 30,
    data_dir: str = "data",
    cache_valid_hours: int = 12,
    adjust: str = "qfq",
    min_bars: int = 130,
    sleep_on_error: float = 0.5,
    cache_mode: str = "auto",
) -> Optional[pd.DataFrame]:
    """
    拉取单只股票历史日线数据。

    参数：
        code:               6位股票代码，如 '600000'
        history_days:       回溯天数，默认30天
        data_dir:           缓存目录
        cache_valid_hours:  缓存有效时间，仅 cache_mode='auto' 时生效
                            <=0 表示缓存永久有效
        adjust:             复权方式：'qfq'前复权，'hfq'后复权，''不复权
        min_bars:           最少需要的交易日数量，不足返回None
        sleep_on_error:     网络出错后等待秒数
        cache_mode:         缓存策略
                            'auto'    : 按 cache_valid_hours 判断是否使用缓存
                            'local'   : 强制使用本地缓存，不联网下载行情
                            'refresh' : 强制重新下载，忽略缓存时间

    返回：
        标准化 DataFrame，或 None
    """
    code = normalize_code(code)
    ensure_data_dir(data_dir)
    cache_path = os.path.join(data_dir, f"{code}.csv")

    if cache_mode not in ("auto", "local", "refresh"):
        raise ValueError("cache_mode 只能是 'auto', 'local', 'refresh'")

    # ==============================
    # 1. local 模式：强制使用本地缓存
    # ==============================
    if cache_mode == "local":
        df = read_cache(cache_path, min_bars=min_bars)
        if df is None:
            print(f"{code} 本地缓存不存在或数据不足")
        return df

    # ==============================
    # 2. auto 模式：缓存有效则直接读取
    # ==============================
    if cache_mode == "auto":
        if is_cache_valid(cache_path, cache_valid_hours):
            df = read_cache(cache_path, min_bars=min_bars)
            if df is not None:
                return df
            # 缓存存在但损坏或数据不足，则继续下载

    # ==============================
    # 3. refresh 模式：跳过缓存，直接下载
    #    或 auto 模式缓存失效后下载
    # ==============================
    try:
        symbol = to_market_symbol(code)

        raw_df = ak.stock_zh_a_daily(
            symbol=symbol,
            start_date=start_date_str(history_days),
            end_date=today_str(),
            adjust=adjust,
        )

        df = standardize_history_df(raw_df, code)

        if df is not None and not df.empty:
            df.to_csv(cache_path, index=False)

            if len(df) < min_bars:
                return None

            return df

        # 下载成功但数据为空，尝试使用旧缓存
        old_df = read_cache(cache_path, min_bars=min_bars)
        if old_df is not None:
            print(f"{code} 下载为空，使用旧缓存")
            return old_df

        return None

    except Exception as e:
        print(f"{code} 下载失败：{e}")

        # 下载失败时，如果有旧缓存则使用
        old_df = read_cache(cache_path, min_bars=min_bars)
        if old_df is not None:
            print(f"{code} 使用本地缓存数据")
            return old_df

        time.sleep(sleep_on_error)
        return None


# ==============================
# 获取主板股票列表
# ==============================

def get_mainboard_stocks() -> pd.DataFrame:
    """
    获取沪深主板股票列表，剔除ST和退市股。

    返回：
        DataFrame(code, name)
    """
    print("正在获取A股股票列表...")
    stock_info = None

    try:
        stock_info = ak.stock_info_a_code_name()
    except Exception:
        try:
            stock_info = ak.stock_zh_a_spot()
        except Exception as e:
            raise RuntimeError(f"无法获取股票列表：{e}")

    if stock_info is None or stock_info.empty:
        raise RuntimeError("股票列表为空")

    # 统一列名
    if "code" in stock_info.columns and "name" in stock_info.columns:
        df = stock_info[["code", "name"]].copy()

    elif "代码" in stock_info.columns and "名称" in stock_info.columns:
        df = stock_info[["代码", "名称"]].copy()
        df.columns = ["code", "name"]

    elif "symbol" in stock_info.columns and "name" in stock_info.columns:
        df = stock_info[["symbol", "name"]].copy()
        df.columns = ["code", "name"]

    else:
        raise ValueError(f"股票列表字段异常：{stock_info.columns.tolist()}")

    df["code"] = df["code"].apply(normalize_code)
    df["name"] = df["name"].astype(str)

    # 剔除 ST、退市
    df = df[~df["name"].str.contains(r"ST|退", case=False, regex=True, na=False)]

    # 仅主板：沪市60开头，深市00开头
    df = df[df["code"].str.startswith(("60", "00"))]

    df = df.drop_duplicates(subset=["code"]).reset_index(drop=True)

    print(f"主板股票数量：{len(df)}")
    return df


def get_codes_from_local_cache(data_dir: str = "data") -> list:
    """
    从本地缓存目录读取股票代码列表。

    用途：
        当 --all --cache-mode local 时，可以不请求股票列表接口，
        直接扫描本地 data 目录下的 CSV 文件。

    返回：
        股票代码列表
    """
    if not os.path.exists(data_dir):
        return []

    codes = []
    for file in os.listdir(data_dir):
        if not file.lower().endswith(".csv"):
            continue

        code = os.path.splitext(file)[0]
        code = normalize_code(code)

        if code.startswith(("60", "00")):
            codes.append(code)

    return sorted(list(set(codes)))


def download_all_stocks(
    history_days: int = 30,
    data_dir: str = "data",
    cache_valid_hours: int = 12,
    adjust: str = "qfq",
    max_workers: int = 8,
    sleep_on_error: float = 0.5,
    cache_mode: str = "auto",
):
    """
    下载所有主板股票的历史行情并缓存。

    参数：
        cache_mode:
            auto    : 按缓存时间判断
            local   : 只读取本地缓存
            refresh : 强制重新下载
    """
    if cache_mode not in ("auto", "local", "refresh"):
        raise ValueError("cache_mode 只能是 'auto', 'local', 'refresh'")

    # local 模式下优先直接扫描本地缓存目录，避免联网获取股票列表
    if cache_mode == "local":
        codes = get_codes_from_local_cache(data_dir)
        if not codes:
            print(f"本地缓存目录 {data_dir} 中没有可用的股票CSV文件")
            return
        print(f"从本地缓存目录读取到 {len(codes)} 只股票")
    else:
        universe = get_mainboard_stocks()
        codes = universe["code"].tolist()

    success_count = 0
    fail_count = 0

    print(f"开始处理 {len(codes)} 只股票，使用 {max_workers} 个线程，cache_mode={cache_mode}...")

    def _download(code):
        try:
            df = fetch_stock_history(
                code=code,
                history_days=history_days,
                data_dir=data_dir,
                cache_valid_hours=cache_valid_hours,
                adjust=adjust,
                min_bars=1,  # 批量下载时不强制最小bar数，由调用方决定
                sleep_on_error=sleep_on_error,
                cache_mode=cache_mode,
            )
            if df is not None and not df.empty:
                return True
            return False
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download, code): code for code in codes}

        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                ok = future.result()
                if ok:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception:
                fail_count += 1

    print(f"处理完成：成功 {success_count}，失败 {fail_count}")


# ==============================
# 命令行入口
# ==============================

def main():
    parser = argparse.ArgumentParser(
        description="A股历史日线数据下载器（主板）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  python stock_data02.py --code 600000,000001
  python stock_data02.py --code 600000 --cache-mode local
  python stock_data02.py --code 600000 --cache-mode refresh
  python stock_data02.py --all --days 30 --workers 8
  python stock_data02.py --all --days 1095 --workers 8 --cache-mode refresh
  python stock_data02.py --all --cache-mode local
        """
    )

    parser.add_argument(
        "--code",
        type=str,
        help="单个或多个股票代码，用逗号分隔，如 600000,000001"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="下载或处理所有主板股票"
    )

    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="回溯天数，默认1天"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="缓存目录，默认 data"
    )

    parser.add_argument(
        "--cache-hours",
        type=int,
        default=3000,
        help="缓存有效小时数，默认3000；仅 --cache-mode auto 时生效；<=0 表示永久有效"
    )

    parser.add_argument(
        "--cache-mode",
        type=str,
        default="auto",
        choices=["auto", "local", "refresh"],
        help="缓存策略：auto按缓存时间判断，local强制使用本地缓存，refresh强制重新下载"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并发线程数，默认8"
    )

    parser.add_argument(
        "--adjust",
        type=str,
        default="qfq",
        choices=["qfq", "hfq", ""],
        help="复权方式：qfq前复权，hfq后复权，''不复权"
    )

    args = parser.parse_args()

    if not args.code and not args.all:
        parser.print_help()
        return

    if args.code:
        codes = [normalize_code(c.strip()) for c in args.code.split(",") if c.strip()]

        for code in codes:
            if args.cache_mode == "local":
                print(f"正在读取本地缓存 {code} ...")
            elif args.cache_mode == "refresh":
                print(f"正在强制重新下载 {code} ...")
            else:
                print(f"正在处理 {code} ...")

            df = fetch_stock_history(
                code=code,
                history_days=args.days,
                data_dir=args.data_dir,
                cache_valid_hours=args.cache_hours,
                adjust=args.adjust,
                min_bars=1,
                sleep_on_error=0.5,
                cache_mode=args.cache_mode,
            )

            if df is not None and not df.empty:
                print(
                    f"  {code} 成功，共 {len(df)} 条数据，"
                    f"缓存文件：{os.path.join(args.data_dir, code + '.csv')}"
                )
            else:
                print(f"  {code} 失败或无数据")

    elif args.all:
        download_all_stocks(
            history_days=args.days,
            data_dir=args.data_dir,
            cache_valid_hours=args.cache_hours,
            adjust=args.adjust,
            max_workers=args.workers,
            sleep_on_error=0.5,
            cache_mode=args.cache_mode,
        )

    print("任务完成。")


if __name__ == "__main__":
    main()