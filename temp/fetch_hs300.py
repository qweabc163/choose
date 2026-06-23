#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
沪深300指数数据获取与收益率计算（独立版）

功能：
- 通过 AkShare 获取 sh000300 日线数据
- 计算最近5个交易日和10个交易日的收益率
- 可选保存原始数据到 CSV 文件
- 打印结果到控制台
"""

import os
import warnings
import pandas as pd
import akshare as ak
from datetime import datetime

warnings.filterwarnings("ignore")


def get_hs300_daily() -> pd.DataFrame:
    """
    获取沪深300指数日线数据（sh000300）
    返回 DataFrame，包含 date, close 等字段，按日期升序排列
    """
    try:
        # AkShare 获取指数日线
        df = ak.stock_zh_index_daily(symbol="sh000300")
        if df is None or df.empty:
            raise ValueError("返回数据为空")
    except Exception as e:
        raise RuntimeError(f"从 AkShare 获取沪深300数据失败: {e}")

    # 统一列名（兼容不同版本）
    rename_map = {}
    if "日期" in df.columns:
        rename_map["日期"] = "date"
    if "收盘" in df.columns:
        rename_map["收盘"] = "close"
    if "date" not in df.columns and "日期" not in df.columns:
        raise ValueError("数据缺少日期列")
    if "close" not in df.columns and "收盘" not in df.columns:
        raise ValueError("数据缺少收盘价列")
    df = df.rename(columns=rename_map)

    # 确保日期和收盘价是正确类型
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


def calc_returns(df: pd.DataFrame, periods: list = [5, 10]) -> dict:
    """
    计算指定周期（交易日数）的收益率
    返回字典 {period: return}
    """
    if len(df) < max(periods) + 1:
        raise ValueError(f"数据不足：当前仅有 {len(df)} 条记录，至少需要 {max(periods)+1} 条")

    close = df["close"]
    latest = close.iloc[-1]
    result = {}
    for p in periods:
        if len(close) >= p + 1:
            ret = latest / close.iloc[-p-1] - 1
            result[p] = ret
        else:
            result[p] = None
    return result


def save_hs300_data(df: pd.DataFrame, output_dir: str = ".") -> str:
    """保存数据到 CSV 文件，返回文件路径"""
    os.makedirs(output_dir, exist_ok=True)
    fname = f"hs300_{datetime.now().strftime('%Y%m%d')}.csv"
    path = os.path.join(output_dir, fname)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main():
    print("开始获取沪深300指数数据...")
    try:
        df = get_hs300_daily()
    except Exception as e:
        print(f"错误：{e}")
        return

    if df.empty:
        print("未获取到有效数据")
        return

    print(f"数据时间范围：{df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"总记录数：{len(df)}")

    try:
        rets = calc_returns(df, periods=[5, 10])
    except Exception as e:
        print(f"计算收益率失败：{e}")
        return

    print("\n=== 最新收益率 ===")
    for p, r in rets.items():
        if r is not None:
            print(f"近{p}日收益率：{r:.2%}")
        else:
            print(f"近{p}日收益率：数据不足")

    # 可选保存
    save_choice = input("\n是否保存原始数据到 CSV？(y/n): ").strip().lower()
    if save_choice == 'y':
        path = save_hs300_data(df)
        print(f"数据已保存至：{path}")


if __name__ == "__main__":
    main()