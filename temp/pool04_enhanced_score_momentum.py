#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A股趋势策略增强补丁版：
在 pool04_enhanced_score_trend.py 基础上增加“评分动量排序”。

使用方式：

1. 将本文件保存为：
   pool04_enhanced_score_momentum.py

2. 保证原文件仍在同目录：
   pool04_enhanced_score_trend.py

3. 运行：

仅分类：
python pool04_enhanced_score_momentum.py --date 2026-06-18 --classify-only

非交易日自动前移：
python pool04_enhanced_score_momentum.py --date 2026-06-19 --use-prev-trading-day --classify-only

回测：
python pool04_enhanced_score_momentum.py --date 2026-06-18 --days 30

调整评分趋势回看天数：
python pool04_enhanced_score_momentum.py --date 2026-06-23 --classify-only --score-trend-lookback 10
"""

import os
import numpy as np
import pandas as pd

# 导入你原来的完整策略文件
import pool04_enhanced_score_trend as base


# =========================================================
# 1. 评分趋势增强：增加 1日/3日/5日变化、上涨天数、评分动量
# =========================================================

def classify_score_trend(g: pd.DataFrame) -> pd.Series:
    """
    增强版评分趋势分类。

    新增字段：
    - score_1d_ago
    - score_change_1d
    - score_rise_days_5
    - score_momentum

    score_momentum 用于排序：
    最近5日变化权重最高，其次3日，再其次1日，并奖励连续上涨天数。
    """

    g = g.sort_values("signal_date").copy()

    scores = g["score"].dropna().astype(float).tolist()
    statuses = g["status"].astype(str).tolist()

    if len(scores) < 4:
        return pd.Series({
            "score_trend_status": "样本不足",

            "score_now": np.nan,
            "score_1d_ago": np.nan,
            "score_3d_ago": np.nan,
            "score_5d_ago": np.nan,

            "score_change_1d": np.nan,
            "score_change_3d": np.nan,
            "score_change_5d": np.nan,

            "score_ma3": np.nan,
            "score_ma5": np.nan,
            "score_min5": np.nan,
            "score_max5": np.nan,

            "score_rise_days_5": np.nan,
            "score_momentum": np.nan,

            "status_5d_ago": "",
            "status_changed": False,
        })

    window = getattr(base, "SCORE_TREND_WINDOW", 5)
    last_scores = scores[-window:] if len(scores) >= window else scores

    score_now = scores[-1]

    score_1d_ago = scores[-2] if len(scores) >= 2 else np.nan
    score_3d_ago = scores[-4] if len(scores) >= 4 else np.nan
    score_5d_ago = scores[-6] if len(scores) >= 6 else last_scores[0]

    score_change_1d = score_now - score_1d_ago if not pd.isna(score_1d_ago) else np.nan
    score_change_3d = score_now - score_3d_ago if not pd.isna(score_3d_ago) else np.nan
    score_change_5d = score_now - score_5d_ago if not pd.isna(score_5d_ago) else np.nan

    score_ma3 = np.mean(last_scores[-3:]) if len(last_scores) >= 3 else np.mean(last_scores)
    score_ma5 = np.mean(last_scores)
    score_min5 = np.min(last_scores)
    score_max5 = np.max(last_scores)

    # 最近窗口内评分上涨天数
    score_rise_days_5 = 0
    if len(last_scores) >= 2:
        for i in range(1, len(last_scores)):
            if last_scores[i] > last_scores[i - 1]:
                score_rise_days_5 += 1

    c1 = 0 if pd.isna(score_change_1d) else score_change_1d
    c3 = 0 if pd.isna(score_change_3d) else score_change_3d
    c5 = 0 if pd.isna(score_change_5d) else score_change_5d

    # 评分动量：
    # 5日变化权重最高，3日变化次之，1日变化补充，连续上涨天数额外加分。
    score_momentum = (
        c5 * 0.55 +
        c3 * 0.30 +
        c1 * 0.15 +
        score_rise_days_5 * 2
    )

    status_5d_ago = statuses[-6] if len(statuses) >= 6 else statuses[0]
    status_now = statuses[-1]
    status_changed = status_now != status_5d_ago

    if score_now >= 85 and score_min5 >= 75 and score_ma3 >= 85:
        status = "强趋势延续"
    elif score_now >= 70 and score_change_5d >= 20 and score_ma3 > score_ma5:
        status = "趋势转强"
    elif score_now >= 60 and score_change_5d >= 15:
        status = "趋势修复"
    elif score_5d_ago >= 85 and score_now <= 70 and score_ma3 < score_ma5:
        status = "高位转弱"
    elif score_now < 50 and score_ma5 < 50:
        status = "弱势延续"
    else:
        status = "震荡无趋势"

    return pd.Series({
        "score_trend_status": status,

        "score_now": score_now,
        "score_1d_ago": score_1d_ago,
        "score_3d_ago": score_3d_ago,
        "score_5d_ago": score_5d_ago,

        "score_change_1d": score_change_1d,
        "score_change_3d": score_change_3d,
        "score_change_5d": score_change_5d,

        "score_ma3": score_ma3,
        "score_ma5": score_ma5,
        "score_min5": score_min5,
        "score_max5": score_max5,

        "score_rise_days_5": score_rise_days_5,
        "score_momentum": score_momentum,

        "status_5d_ago": status_5d_ago,
        "status_changed": status_changed,
    })


# =========================================================
# 2. 分类排序增强：分类内部按评分动量排序
# =========================================================

def sort_pool_df(df_pool: pd.DataFrame) -> pd.DataFrame:
    """
    增强版排序。

    原逻辑：
    分类顺序 + 当前评分

    新逻辑：
    分类顺序 + 评分动量 + 近5日变化 + 近3日变化 + 近1日变化 + 当前评分
    """

    if df_pool is None or df_pool.empty:
        return pd.DataFrame()

    status_order = base.get_status_order()

    df_pool = df_pool.copy()
    df_pool["_order"] = df_pool["status"].map(status_order).fillna(99)

    sort_cols = ["_order"]
    ascending = [True]

    for col in [
        "score_momentum",
        "score_change_5d",
        "score_change_3d",
        "score_change_1d",
        "score",
    ]:
        if col in df_pool.columns:
            sort_cols.append(col)
            ascending.append(False)

    df_pool = (
        df_pool
        .sort_values(sort_cols, ascending=ascending)
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )

    return df_pool


# =========================================================
# 3. 构建全市场“评分变化排序”榜单
# =========================================================

def build_score_change_rank_df(pool_df: pd.DataFrame) -> pd.DataFrame:
    """
    全市场评分变化排序榜。

    这个榜单不以分类为第一排序，而是直接按照：
    评分动量 > 近5日评分变化 > 近3日评分变化 > 近1日评分变化 > 当前评分

    这样你可以直接看到最近几天评分提升最快的股票。
    """

    if pool_df is None or pool_df.empty:
        return pd.DataFrame()

    df = pool_df.copy()

    for col in [
        "score_momentum",
        "score_change_5d",
        "score_change_3d",
        "score_change_1d",
        "score_rise_days_5",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    # 必须有评分历史
    df = df[df["score_change_5d"].notna()].copy()

    if df.empty:
        return pd.DataFrame()

    # 过滤掉明显不可看的弱票。
    # 如果你想看全市场所有票，可以注释掉这几段。
    if "basic_liquid" in df.columns:
        df = df[df["basic_liquid"] == True]

    if "close" in df.columns and "ma60" in df.columns:
        df = df[df["close"] > df["ma60"]]

    if "score" in df.columns:
        df = df[df["score"] >= 50]

    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(
        [
            "score_momentum",
            "score_change_5d",
            "score_change_3d",
            "score_change_1d",
            "score",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    return df


# =========================================================
# 4. 可读版输出：增加评分动量字段
# =========================================================

def make_classification_readable(pool_df: pd.DataFrame) -> pd.DataFrame:
    """
    增强版可读输出。
    相比原版增加：
    - 1日前评分
    - 近1日评分变化
    - 近5日评分上涨天数
    - 评分动量
    """

    if pool_df is None or pool_df.empty:
        return pd.DataFrame()

    df = pool_df.copy()

    keep_cols = [
        "code", "name", "industry", "signal_date", "status", "score",

        "score_trend_status",
        "score_now",
        "score_1d_ago",
        "score_3d_ago",
        "score_5d_ago",
        "score_change_1d",
        "score_change_3d",
        "score_change_5d",
        "score_ma3",
        "score_ma5",
        "score_min5",
        "score_max5",
        "score_rise_days_5",
        "score_momentum",
        "status_5d_ago",
        "status_changed",
        "trend_confirmed_by_score",

        "market_state", "market_ok",

        "close",
        "ma5", "ma10", "ma20", "ma60", "ma120",
        "ma5_slope_3", "ma10_slope_5", "ma20_slope_5", "ma60_slope_10",

        "ret5", "ret10", "ret20", "ret60",
        "ret20_rank_pct", "ret60_rank_pct",

        "industry_ret20", "industry_ret60",
        "industry_ret20_rank_pct", "industry_ret60_rank_pct",
        "industry_strong",

        "amount_ma5", "amount_ma10", "amount_ma20", "amount_ratio_5_20",
        "turnover_ma5", "turnover_ma20",

        "is_20d_high", "is_60d_high", "dist_to_60d_high",
        "max_dd20", "atr20", "atr20_pct",
        "up_down_amount_ratio20",

        "limit_up_count_5", "heavy_bearish_candle",
        "short_term_strong", "pullback_rebound",

        "bench_ret20", "bench_ret60",

        "basic_liquid", "trend_basic", "relative_strength",
        "volume_ok", "near_breakout", "risk_ok", "short_term_ok",
        "candidate",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]
    out = df[keep_cols].copy()

    rename_map = {
        "code": "代码",
        "name": "名称",
        "industry": "行业",
        "signal_date": "信号日",
        "status": "分类",
        "score": "评分",

        "score_trend_status": "评分趋势状态",
        "score_now": "当前评分",
        "score_1d_ago": "1日前评分",
        "score_3d_ago": "3日前评分",
        "score_5d_ago": "5日前评分",
        "score_change_1d": "近1日评分变化",
        "score_change_3d": "近3日评分变化",
        "score_change_5d": "近5日评分变化",
        "score_ma3": "近3日平均评分",
        "score_ma5": "近5日平均评分",
        "score_min5": "近5日最低评分",
        "score_max5": "近5日最高评分",
        "score_rise_days_5": "近5日评分上涨天数",
        "score_momentum": "评分动量",
        "status_5d_ago": "5日前分类",
        "status_changed": "分类是否变化",
        "trend_confirmed_by_score": "评分确认转强",

        "market_state": "市场状态",
        "market_ok": "大盘环境通过",

        "close": "收盘价",
        "ma5": "MA5",
        "ma10": "MA10",
        "ma20": "MA20",
        "ma60": "MA60",
        "ma120": "MA120",

        "ma5_slope_3": "MA5近3日斜率",
        "ma10_slope_5": "MA10近5日斜率",
        "ma20_slope_5": "MA20近5日斜率",
        "ma60_slope_10": "MA60近10日斜率",

        "ret5": "近5日涨幅",
        "ret10": "近10日涨幅",
        "ret20": "近20日涨幅",
        "ret60": "近60日涨幅",

        "ret20_rank_pct": "20日强度排名百分位",
        "ret60_rank_pct": "60日强度排名百分位",

        "industry_ret20": "行业20日涨幅",
        "industry_ret60": "行业60日涨幅",
        "industry_ret20_rank_pct": "行业20日强度百分位",
        "industry_ret60_rank_pct": "行业60日强度百分位",
        "industry_strong": "行业强度达标",

        "amount_ma5": "5日均成交额",
        "amount_ma10": "10日均成交额",
        "amount_ma20": "20日均成交额",
        "amount_ratio_5_20": "量能比5/20",

        "turnover_ma5": "5日均换手率",
        "turnover_ma20": "20日均换手率",

        "is_20d_high": "是否接近20日新高",
        "is_60d_high": "是否近60日新高",
        "dist_to_60d_high": "距60日高点",

        "max_dd20": "近20日最大回撤",
        "atr20": "ATR20",
        "atr20_pct": "ATR20比例",
        "up_down_amount_ratio20": "近20日上涨/下跌成交额比",

        "limit_up_count_5": "近5日涨停次数",
        "heavy_bearish_candle": "是否放量长阴",
        "short_term_strong": "短线强势",
        "pullback_rebound": "回踩修复",

        "bench_ret20": "沪深300近20日涨幅",
        "bench_ret60": "沪深300近60日涨幅",

        "basic_liquid": "流动性达标",
        "trend_basic": "中期趋势达标",
        "relative_strength": "相对强度达标",
        "volume_ok": "量能达标",
        "near_breakout": "接近突破",
        "risk_ok": "风险过滤通过",
        "short_term_ok": "短线结构通过",
        "candidate": "是否候选",
    }

    out = out.rename(columns=rename_map)
    return out.reset_index(drop=True)


# =========================================================
# 5. 分类 Excel 增强：新增“评分变化排序”Sheet
# =========================================================

def save_classification_excel(
    output_xlsx: str,
    signal_date: pd.Timestamp,
    pool_df: pd.DataFrame,
) -> None:
    """
    增强版分类 Excel 输出。

    新增 sheet：
    - 评分变化排序
    """

    readable_df = make_classification_readable(pool_df)
    summary_df = base.build_classification_summary(pool_df)

    score_change_rank_df = build_score_change_rank_df(pool_df)
    score_change_readable_df = make_classification_readable(score_change_rank_df)

    dashboard_rows = []
    total_count = len(pool_df) if pool_df is not None else 0

    dashboard_rows.append(["信号日期", signal_date.strftime("%Y-%m-%d"), "分类所使用的交易日"])
    dashboard_rows.append(["全市场有效样本数", total_count, "满足数据长度和指标完整性的股票数量"])
    dashboard_rows.append(["最低评分参数", base.MIN_SCORE, "趋势观察池最低评分"])
    dashboard_rows.append(["强趋势评分参数", base.STRONG_SCORE, "强趋势池最低评分"])
    dashboard_rows.append(["评分趋势回看交易日", base.SCORE_TREND_LOOKBACK, "用于 score_trend_status"])
    dashboard_rows.append(["评分趋势窗口", base.SCORE_TREND_WINDOW, "默认最近5日计算 score_change_5d"])
    dashboard_rows.append(["20日均成交额门槛", base.MIN_AVG_AMOUNT_20, "流动性过滤"])
    dashboard_rows.append(["最少K线数量", base.MIN_BARS, "参与计算要求"])
    dashboard_rows.append(["大盘环境过滤", "开启" if base.MARKET_FILTER_ENABLED else "关闭", "沪深300趋势过滤"])
    dashboard_rows.append(["行业强度过滤", "开启" if base.INDUSTRY_FILTER_ENABLED else "关闭", "依赖 data/_stock_industry.csv"])
    dashboard_rows.append(["新增排序方式", "评分动量排序", "分类内部和评分变化榜单优先按近期评分变化排序"])

    if summary_df is not None and not summary_df.empty:
        for _, r in summary_df.iterrows():
            dashboard_rows.append([
                f"{r['分类']}数量",
                r["数量"],
                f"占比 {r['占比']:.2%}",
            ])

    dashboard_df = pd.DataFrame(dashboard_rows, columns=["指标", "数值", "说明"])

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        dashboard_df.to_excel(writer, sheet_name="看板", index=False)
        summary_df.to_excel(writer, sheet_name="分类统计", index=False)
        readable_df.to_excel(writer, sheet_name="全市场分类", index=False)

        if score_change_readable_df is not None and not score_change_readable_df.empty:
            score_change_readable_df.to_excel(writer, sheet_name="评分变化排序", index=False)

        important_statuses = list(base.get_status_order().keys())

        if not readable_df.empty and "分类" in readable_df.columns:
            for status in important_statuses:
                sub = readable_df[readable_df["分类"] == status].copy()

                if sub.empty:
                    continue

                sheet_name = status[:31]
                sub.to_excel(writer, sheet_name=sheet_name, index=False)

        wb = writer.book

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]

            if sheet_name == "看板":
                base.style_worksheet(ws, freeze="A2", apply_red_green=True)
                ws.column_dimensions["A"].width = 28
                ws.column_dimensions["B"].width = 20
                ws.column_dimensions["C"].width = 60
            else:
                base.style_worksheet(ws, freeze="A2", apply_red_green=True)


# =========================================================
# 6. 分类终端预览增强：核心池按评分动量排序
# =========================================================

def sort_by_score_momentum(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    sort_cols = []
    ascending = []

    for col in [
        "score_momentum",
        "score_change_5d",
        "score_change_3d",
        "score_change_1d",
        "score",
    ]:
        if col in df.columns:
            sort_cols.append(col)
            ascending.append(False)

    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    elif "score" in df.columns:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)

    return df


def analyze_pool_classification(
    signal_date: pd.Timestamp,
    stock_data,
    bench_df: pd.DataFrame,
    code_name_map,
    industry_map,
    score_trend_lookback: int,
) -> None:
    """
    增强版分类分析。
    主要变化：
    - 输出 Excel 增加“评分变化排序”
    - 终端核心股票池预览按评分动量排序
    """

    signal_date = pd.Timestamp(signal_date).normalize()

    print("\n========== 指定日期股票分类：评分动量增强版 ==========")
    print(f"信号日期: {signal_date.date()}")
    print("功能: 仅分类，不计算未来收益")
    print("排序: 分类内部优先按评分动量、近5日变化、近3日变化、近1日变化排序")
    print("====================================================")

    pool_df = base.build_final_pool_with_score_trend(
        signal_date=signal_date,
        stock_data=stock_data,
        bench_df=bench_df,
        code_name_map=code_name_map,
        industry_map=industry_map,
        score_trend_lookback=score_trend_lookback,
    )

    if pool_df is None or pool_df.empty:
        print("指定日期未能构建股票分类结果。")
        return

    os.makedirs(base.OUTPUT_DIR, exist_ok=True)

    output_xlsx = os.path.join(
        base.OUTPUT_DIR,
        f"{base.CLASSIFICATION_OUTPUT_PREFIX}_{signal_date.strftime('%Y-%m-%d')}.xlsx",
    )

    output_csv = os.path.join(
        base.OUTPUT_DIR,
        f"{base.CLASSIFICATION_OUTPUT_PREFIX}_{signal_date.strftime('%Y-%m-%d')}.csv",
    )

    try:
        save_classification_excel(
            output_xlsx=output_xlsx,
            signal_date=signal_date,
            pool_df=pool_df,
        )
        print(f"\n分类结果 Excel 已保存到：{output_xlsx}")
        print("Excel 已新增 Sheet：评分变化排序")
    except Exception as e:
        print(f"保存分类 Excel 失败：{e}")

    try:
        pool_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"分类结果 CSV 已保存到：{output_csv}")
    except Exception as e:
        print(f"保存分类 CSV 失败：{e}")

    summary_df = base.build_classification_summary(pool_df)

    if not summary_df.empty:
        print("\n========== 分类统计 ==========")

        formatters = {
            "占比": "{:.2%}".format,
            "平均评分": "{:.2f}".format,
            "最高评分": "{:.2f}".format,
            "平均20日涨幅": "{:.2%}".format,
            "平均60日涨幅": "{:.2%}".format,
            "平均20日成交额": "{:,.0f}".format,
            "平均量能比": "{:.2f}".format,
            "平均20日最大回撤": "{:.2%}".format,
            "平均ATR比例": "{:.2%}".format,
        }

        if "平均5日评分变化" in summary_df.columns:
            formatters["平均5日评分变化"] = "{:.2f}".format

        print(summary_df.to_string(index=False, formatters=formatters))

    selected_df = pool_df[pool_df["status"].isin([
        "短中线转强池",
        "短线强势池",
        "强趋势池",
        "趋势观察池",
        "回踩修复观察",
    ])].copy()

    selected_df = sort_by_score_momentum(selected_df)

    if not selected_df.empty:
        preview_cols = [
            "code", "name", "industry", "status", "score",
            "score_trend_status",
            "score_momentum",
            "score_change_1d",
            "score_change_3d",
            "score_change_5d",
            "score_rise_days_5",
            "ret20", "ret60",
            "amount_ma20", "amount_ratio_5_20",
            "max_dd20", "atr20_pct",
            "short_term_strong", "pullback_rebound",
            "trend_confirmed_by_score",
            "dist_to_60d_high",
        ]

        preview_cols = [c for c in preview_cols if c in selected_df.columns]

        print("\n========== 核心股票池预览：按评分动量排序 ==========")
        print(selected_df[preview_cols].head(30).to_string(index=False, formatters={
            "score_momentum": "{:.2f}".format,
            "score_change_1d": "{:.2f}".format,
            "score_change_3d": "{:.2f}".format,
            "score_5d": "{:.2f}".format,
            "ret20": "{:.2%}".format,
            "ret60": "{:.2%}".format,
            "amount_ma20": "{:,.0f}".format,
            "amount_ratio_5_20": "{:.2f}".format,
            "max_dd20": "{:.2%}".format,
            "atr20_pct": "{:.2%}".format,
            "dist_to_60d_high": "{:.2%}".format,
        }))
    else:
        print("\n指定日期没有股票进入核心股票池。")

    score_rank_df = build_score_change_rank_df(pool_df)

    if score_rank_df is not None and not score_rank_df.empty:
        preview_cols = [
            "code", "name", "industry", "status", "score",
            "score_trend_status",
            "score_momentum",
            "score_change_1d",
            "score_change_3d",
            "score_change_5d",
            "score_rise_days_5",
            "ret20", "ret60",
            "close", "ma20", "ma60",
            "short_term_strong", "pullback_rebound",
        ]

        preview_cols = [c for c in preview_cols if c in score_rank_df.columns]

        print("\n========== 全市场评分变化排序 Top 30 ==========")
        print(score_rank_df[preview_cols].head(30).to_string(index=False, formatters={
            "score_momentum": "{:.2f}".format,
            "score_change_1d": "{:.2f}".format,
            "score_change_3d": "{:.2f}".format,
            "score_change_5d": "{:.2f}".format,
            "ret20": "{:.2%}".format,
            "ret60": "{:.2%}".format,
        }))


# =========================================================
# 7. 回测模式增强：入选股票也按评分动量排序
# =========================================================

def analyze_pool_forward_returns(
    signal_date: pd.Timestamp,
    stock_data,
    bench_df: pd.DataFrame,
    code_name_map,
    industry_map,
    forward_days: int,
    statuses,
    score_trend_lookback: int,
) -> None:
    """
    回测模式增强版。

    为了尽量复用原始代码，这里只在构建股票池后先做一次评分动量排序。
    后续收益计算逻辑仍使用原始函数中的逻辑。
    """

    signal_date = pd.Timestamp(signal_date).normalize()

    print("\n========== 指定日期股票池未来走势分析：评分动量增强版 ==========")
    print(f"信号日期: {signal_date.date()}")
    print(f"分析池: {statuses}")
    print(f"未来交易日数: {forward_days}")
    print("排序: 入选股票优先按评分动量排序")
    print("==============================================================")

    pool_df = base.build_final_pool_with_score_trend(
        signal_date=signal_date,
        stock_data=stock_data,
        bench_df=bench_df,
        code_name_map=code_name_map,
        industry_map=industry_map,
        score_trend_lookback=score_trend_lookback,
    )

    if pool_df is None or pool_df.empty:
        print("指定日期未能构建股票池。")
        return

    selected_df = pool_df[pool_df["status"].isin(statuses)].copy()
    selected_df = sort_by_score_momentum(selected_df)

    os.makedirs(base.OUTPUT_DIR, exist_ok=True)

    if selected_df.empty:
        print(f"{signal_date.date()} 没有股票进入 {statuses}。")

        output_all_pool = os.path.join(
            base.OUTPUT_DIR,
            f"{base.FORWARD_ANALYSIS_OUTPUT_PREFIX}_{signal_date.strftime('%Y-%m-%d')}_all_pool.csv",
        )

        pool_df.to_csv(output_all_pool, index=False, encoding="utf-8-sig")
        print(f"已保存当日全市场分类到：{output_all_pool}")
        return

    print(f"入选股票数量: {len(selected_df)}")

    preview_cols = [
        "code", "name", "industry", "status", "score",
        "score_trend_status",
        "score_momentum",
        "score_change_1d",
        "score_change_3d",
        "score_change_5d",
        "score_rise_days_5",
        "ret20", "ret60",
        "amount_ma20", "amount_ratio_5_20", "atr20_pct",
        "short_term_strong", "pullback_rebound",
        "trend_confirmed_by_score",
    ]

    preview_cols = [c for c in preview_cols if c in selected_df.columns]

    print("\n入选股票预览：按评分动量排序")
    print(selected_df[preview_cols].head(30).to_string(index=False, formatters={
        "score_momentum": "{:.2f}".format,
        "score_change_1d": "{:.2f}".format,
        "score_change_3d": "{:.2f}".format,
        "score_change_5d": "{:.2f}".format,
        "ret20": "{:.2%}".format,
        "ret60": "{:.2%}".format,
        "amount_ma20": "{:,.0f}".format,
        "amount_ratio_5_20": "{:.2f}".format,
        "atr20_pct": "{:.2%}".format,
    }))

    forward_dates = base.get_forward_trading_dates(
        signal_date=signal_date,
        bench_df=bench_df,
        forward_days=forward_days,
    )

    if len(forward_dates) < forward_days:
        print(f"基准未来交易日不足：需要 {forward_days} 天，实际只有 {len(forward_dates)} 天。")
        print("请降低 --days，或者选择更早的信号日期。")
        return

    bench_path = base.calc_forward_return_path_by_calendar(
        df=bench_df,
        forward_dates=forward_dates,
        buy_price_field=base.BUY_PRICE_FIELD,
    )

    if not bench_path:
        print("基准未来走势计算失败。")
        return

    detail_rows = []

    bench_row = {
        "row_type": "benchmark",
        "code": "沪深300",
        "name": "沪深300",
        "industry": "",
        "status": "benchmark",
        "score": np.nan,
        "signal_date": signal_date.strftime("%Y-%m-%d"),
    }

    bench_row.update(bench_path)
    detail_rows.append(bench_row)

    skipped = 0

    for _, stock in selected_df.iterrows():
        code = str(stock["code"])

        if code not in stock_data:
            skipped += 1
            continue

        path = base.calc_forward_return_path_by_calendar(
            df=stock_data[code],
            forward_dates=forward_dates,
            buy_price_field=base.BUY_PRICE_FIELD,
        )

        if not path:
            skipped += 1
            continue

        row = {
            "row_type": "stock",
            "code": code,
            "name": stock.get("name", code_name_map.get(code, "")),
            "industry": stock.get("industry", ""),
            "status": stock["status"],
            "score": stock["score"],

            "score_trend_status": stock.get("score_trend_status", ""),
            "score_momentum": stock.get("score_momentum", np.nan),
            "score_change_1d": stock.get("score_change_1d", np.nan),
            "score_change_3d": stock.get("score_change_3d", np.nan),
            "score_change_5d": stock.get("score_change_5d", np.nan),
            "score_rise_days_5": stock.get("score_rise_days_5", np.nan),
            "trend_confirmed_by_score": stock.get("trend_confirmed_by_score", False),

            "signal_date": signal_date.strftime("%Y-%m-%d"),
            "close_on_signal": stock["close"],
            "ret20": stock["ret20"],
            "ret60": stock["ret60"],
            "ret20_rank_pct": stock["ret20_rank_pct"],
            "ret60_rank_pct": stock["ret60_rank_pct"],
            "amount_ma20": stock["amount_ma20"],
            "amount_ratio_5_20": stock["amount_ratio_5_20"],
            "max_dd20": stock["max_dd20"],
            "atr20_pct": stock["atr20_pct"],
            "dist_to_60d_high": stock["dist_to_60d_high"],
        }

        row.update(path)
        detail_rows.append(row)

    detail_df = pd.DataFrame(detail_rows)

    if len(detail_df) <= 1:
        print("入选股票未来数据不足，无法生成走势分析。")
        return

    if skipped > 0:
        print(f"有 {skipped} 只股票由于涨停买不进、买入日无交易、价格异常或未来数据缺失被跳过。")

    group_summary_df = base.build_forward_summary(
        detail_df=detail_df,
        forward_days=forward_days,
        group_col="status",
    )

    overall_summary_df = base.build_overall_forward_summary(
        detail_df=detail_df,
        forward_days=forward_days,
    )

    output_xlsx = os.path.join(
        base.OUTPUT_DIR,
        f"{base.FORWARD_ANALYSIS_OUTPUT_PREFIX}_{signal_date.strftime('%Y-%m-%d')}.xlsx",
    )

    output_detail_csv = os.path.join(
        base.OUTPUT_DIR,
        f"{base.FORWARD_ANALYSIS_OUTPUT_PREFIX}_{signal_date.strftime('%Y-%m-%d')}_detail.csv",
    )

    try:
        base.save_forward_excel(
            output_xlsx=output_xlsx,
            signal_date=signal_date,
            detail_df=detail_df,
            group_summary_df=group_summary_df,
            overall_summary_df=overall_summary_df,
            selected_df=selected_df,
            pool_df=pool_df,
            forward_days=forward_days,
        )

        print(f"\n友好版分析结果已保存到：{output_xlsx}")

    except Exception as e:
        print(f"保存友好版 Excel 失败：{e}")
        detail_df.to_csv(output_detail_csv, index=False, encoding="utf-8-sig")
        print(f"走势明细已保存到：{output_detail_csv}")

    print("\n========== 未来走势摘要 ==========")

    key_horizons = [1, 3, 5, 10, 20, 30]
    key_horizons = [h for h in key_horizons if h <= forward_days]

    if not overall_summary_df.empty:
        display_df = overall_summary_df[
            overall_summary_df["horizon"].isin(key_horizons)
        ].copy()

        if not display_df.empty:
            print("\n固定持有整体统计：")
            print(display_df.to_string(index=False, formatters={
                "avg_ret": "{:.2%}".format,
                "median_ret": "{:.2%}".format,
                "win_rate": "{:.2%}".format,
                "bench_ret": "{:.2%}".format,
                "avg_excess": "{:.2%}".format,
                "outperform_rate": "{:.2%}".format,
            }))

    if not group_summary_df.empty:
        display_group_df = group_summary_df[
            group_summary_df["horizon"].isin(key_horizons)
        ].copy()

        if not display_group_df.empty:
            print("\n固定持有分组统计：")
            print(display_group_df.to_string(index=False, formatters={
                "avg_ret": "{:.2%}".format,
                "median_ret": "{:.2%}".format,
                "win_rate": "{:.2%}".format,
                "bench_ret": "{:.2%}".format,
                "avg_excess": "{:.2%}".format,
                "outperform_rate": "{:.2%}".format,
            }))


# =========================================================
# 8. 打补丁：替换原文件中的对应函数
# =========================================================

def patch_base_module():
    """
    将本文件中的增强函数覆盖到原策略模块里。
    """

    base.classify_score_trend = classify_score_trend
    base.sort_pool_df = sort_pool_df
    base.make_classification_readable = make_classification_readable
    base.save_classification_excel = save_classification_excel
    base.analyze_pool_classification = analyze_pool_classification
    base.analyze_pool_forward_returns = analyze_pool_forward_returns

    print("已启用评分动量增强模块：")
    print("- 新增 score_change_1d")
    print("- 新增 score_rise_days_5")
    print("- 新增 score_momentum")
    print("- 分类内部按评分动量排序")
    print("- Excel 新增 Sheet：评分变化排序")


# =========================================================
# 9. 主入口
# =========================================================

if __name__ == "__main__":
    patch_base_module()
    base.main()