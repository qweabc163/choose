#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A股趋势策略进一步增强版：

基于 pool04_enhanced_score_trend.py 的补丁增强文件。

新增能力：
1. MA5 短线斜率信号
   - ma5_slope_1
   - ma5_slope_delta
   - ma5_turn_up
   - ma5_slope_accel_3
   - ma5_short_bull
   - ma5_timing_signal
   - short_term_start

2. 评分动量
   - score_change_1d
   - score_change_3d
   - score_change_5d
   - score_rise_days_5
   - score_momentum

3. 状态迁移
   - status_1d_ago
   - status_3d_ago
   - status_5d_ago
   - status_changed
   - status_upgrade
   - status_downgrade
   - status_path_5d

4. RS 相对强度线
   - rs_close
   - rs_ma20
   - rs_slope_20
   - rs_new_high_60
   - rs_trend_ok

5. ADX / DMI 趋势强度
   - plus_di14
   - minus_di14
   - adx14
   - dmi_trend_ok

6. 突破有效性
   - prev_high20
   - prev_high60
   - breakout_20
   - breakout_60
   - breakout_volume_ok
   - breakout_valid

7. 回踩质量
   - pullback_volume_shrink
   - pullback_depth_10
   - pullback_depth_20
   - pullback_quality

8. 传统辅助指标
   - MACD
   - RSI
   - BOLL

9. 新增分类：
   - 短线启动池

用法：

仅分类：
python pool04_enhanced_full_signal.py --date 2026-06-18 --classify-only

非交易日自动前移：
python pool04_enhanced_full_signal.py --date 2026-06-19 --use-prev-trading-day --classify-only

回测：
python pool04_enhanced_full_signal.py --date 2026-06-23 --days 30
"""

import os
import numpy as np
import pandas as pd

import pool04_enhanced_score_trend as base


# =========================================================
# 1. 基础工具
# =========================================================

def _num(row, col, default=np.nan):
    v = row.get(col, default)
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    return v


def _bool(row, col, default=False):
    v = row.get(col, default)
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    return bool(v)


def _safe_div(a, b):
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b


# =========================================================
# 2. 技术指标增强
# =========================================================

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi


def calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2
    return dif, dea, hist


def calc_adx_dmi(df: pd.DataFrame, period: int = 14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    pre_high = high.shift(1)
    pre_low = low.shift(1)
    pre_close = close.shift(1)

    up_move = high - pre_high
    down_move = pre_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = high - low
    tr2 = (high - pre_close).abs()
    tr3 = (low - pre_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period, min_periods=period).sum()

    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period, min_periods=period).sum() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period, min_periods=period).sum() / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(period, min_periods=period).mean()

    return plus_di, minus_di, adx


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    替换原 add_indicators。

    保留原有全部核心字段，同时新增：
    - MA5 短线斜率
    - MACD
    - RSI
    - BOLL
    - ADX / DMI
    - 突破有效性
    - 回踩质量
    """

    df = df.copy()

    df["pre_close"] = df["close"].shift(1)

    # 均线
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["ma120"] = df["close"].rolling(120).mean()

    # 原有斜率
    df["ma5_slope_3"] = df["ma5"] / df["ma5"].shift(3) - 1
    df["ma10_slope_5"] = df["ma10"] / df["ma10"].shift(5) - 1
    df["ma20_slope_5"] = df["ma20"] / df["ma20"].shift(5) - 1
    df["ma60_slope_10"] = df["ma60"] / df["ma60"].shift(10) - 1

    # 新增 MA5 短线斜率
    df["ma5_slope_1"] = df["ma5"] / df["ma5"].shift(1) - 1
    df["ma5_slope_2"] = df["ma5"] / df["ma5"].shift(2) - 1
    df["ma5_slope_delta"] = df["ma5_slope_1"] - df["ma5_slope_1"].shift(1)

    df["ma5_turn_up"] = (
        (df["ma5_slope_1"] > 0)
        & (df["ma5_slope_1"].shift(1) <= 0)
    )

    df["ma5_slope_accel_3"] = (
        (df["ma5_slope_1"] > df["ma5_slope_1"].shift(1))
        & (df["ma5_slope_1"].shift(1) > df["ma5_slope_1"].shift(2))
    )

    df["ma5_short_bull"] = (
        (df["close"] > df["ma5"])
        & (df["ma5_slope_1"] > 0)
    )

    df["ma5_timing_signal"] = (
        (df["ma5_turn_up"] | df["ma5_slope_accel_3"])
        & df["ma5_short_bull"]
    )

    # 收益
    df["ret5"] = df["close"] / df["close"].shift(5) - 1
    df["ret10"] = df["close"] / df["close"].shift(10) - 1
    df["ret20"] = df["close"] / df["close"].shift(20) - 1
    df["ret60"] = df["close"] / df["close"].shift(60) - 1

    # 量能
    df["amount_ma5"] = df["amount"].rolling(5).mean()
    df["amount_ma10"] = df["amount"].rolling(10).mean()
    df["amount_ma20"] = df["amount"].rolling(20).mean()
    df["amount_ratio_5_20"] = df["amount_ma5"] / df["amount_ma20"]

    df["turnover_ma5"] = df["turnover"].rolling(5).mean() if "turnover" in df.columns else np.nan
    df["turnover_ma20"] = df["turnover"].rolling(20).mean() if "turnover" in df.columns else np.nan

    # 新高和突破
    df["high20"] = df["high"].rolling(20).max()
    df["high60"] = df["high"].rolling(60).max()

    df["prev_high20"] = df["high"].shift(1).rolling(20).max()
    df["prev_high60"] = df["high"].shift(1).rolling(60).max()

    df["is_20d_high"] = df["close"] >= df["high20"] * 0.999
    df["is_60d_high"] = df["close"] >= df["high60"] * 0.999
    df["dist_to_60d_high"] = df["close"] / df["high60"] - 1

    df["breakout_20"] = df["close"] >= df["prev_high20"] * 1.001
    df["breakout_60"] = df["close"] >= df["prev_high60"] * 1.001

    df["breakout_volume_ok"] = (
        df["amount"] >= df["amount_ma20"] * 1.3
    )

    df["breakout_valid"] = (
        (df["breakout_20"] | df["breakout_60"])
        & df["breakout_volume_ok"]
        & (df["close"] > df["ma20"])
        & (df["ma20_slope_5"] > 0)
    )

    # 风险
    df["max_dd20"] = base.rolling_max_drawdown_np(df["close"], window=20)
    df["atr20"] = base.calc_atr(df, base.ATR_PERIOD)
    df["atr20_pct"] = df["atr20"] / df["close"]

    # 量价关系
    df["is_up_day"] = df["close"] > df["pre_close"]
    df["is_down_day"] = df["close"] < df["pre_close"]

    up_amount_20 = df["amount"].where(df["is_up_day"], 0.0).rolling(20, min_periods=20).sum()
    down_amount_20 = df["amount"].where(df["is_down_day"], 0.0).rolling(20, min_periods=20).sum()
    df["up_down_amount_ratio20"] = up_amount_20 / down_amount_20.replace(0, np.nan)

    df["volume_price_confirm"] = (
        (df["ret5"] > 0)
        & (df["amount_ratio_5_20"] >= 1.0)
        & (df["up_down_amount_ratio20"] > 1.0)
    )

    # 涨停、放量长阴
    df["is_limit_up"] = df["pct_chg"] >= 9.5
    df["is_limit_down"] = df["pct_chg"] <= -9.5
    df["limit_up_count_5"] = df["is_limit_up"].rolling(5).sum()

    intraday_range = df["high"] - df["low"]
    close_position = np.where(
        intraday_range > 0,
        (df["close"] - df["low"]) / intraday_range,
        np.nan,
    )

    df["heavy_bearish_candle"] = (
        (df["pct_chg"] <= -5)
        & (df["amount"] > df["amount_ma20"] * 1.5)
        & (close_position <= 0.35)
    )

    # 短线强势
    df["short_term_strong_raw"] = (
        (df["close"] > df["ma5"])
        & (df["ma5"] > df["ma10"])
        & (df["ma10"] > df["ma20"])
        & (df["ma5_slope_3"] > 0)
        & (df["ma10_slope_5"] > 0)
    )

    # 回踩修复
    df["low5"] = df["low"].rolling(5).min()
    df["pullback_to_ma10"] = (df["low5"] / df["ma10"] - 1).abs() <= 0.035
    df["pullback_to_ma20"] = (df["low5"] / df["ma20"] - 1).abs() <= 0.045
    df["reclaim_ma10"] = (df["close"] > df["ma10"]) & (df["close"].shift(1) <= df["ma10"].shift(1))

    df["pullback_depth_10"] = df["low5"] / df["ma10"] - 1
    df["pullback_depth_20"] = df["low5"] / df["ma20"] - 1

    df["pullback_volume_shrink"] = (
        df["amount_ma5"] <= df["amount_ma20"] * 0.95
    )

    df["pullback_quality"] = (
        (df["ma20"] > df["ma60"])
        & (df["close"] > df["ma60"])
        & (df["pullback_to_ma10"] | df["pullback_to_ma20"])
        & (df["max_dd20"] > -0.18)
        & (~df["heavy_bearish_candle"])
        & (
            df["pullback_volume_shrink"]
            | (df["amount_ratio_5_20"] >= 0.8)
        )
    )

    df["pullback_rebound_raw"] = (
        (df["ma20"] > df["ma60"])
        & (df["ma60"] > df["ma120"])
        & (df["close"] > df["ma60"])
        & (df["pullback_to_ma10"] | df["pullback_to_ma20"])
        & ((df["close"] > df["ma10"]) | df["reclaim_ma10"])
        & (df["amount_ratio_5_20"] >= 0.8)
    )

    # MACD
    df["macd_dif"], df["macd_dea"], df["macd_hist"] = calc_macd(df["close"])
    df["macd_gold_cross"] = (
        (df["macd_dif"] > df["macd_dea"])
        & (df["macd_dif"].shift(1) <= df["macd_dea"].shift(1))
    )
    df["macd_above_zero"] = df["macd_dif"] > 0
    df["macd_hist_rising"] = df["macd_hist"] > df["macd_hist"].shift(1)

    # RSI
    df["rsi6"] = calc_rsi(df["close"], 6)
    df["rsi14"] = calc_rsi(df["close"], 14)
    df["rsi_overheat"] = df["rsi6"] >= 85
    df["rsi_strength"] = df["rsi14"] >= 50

    # BOLL
    df["boll_mid"] = df["close"].rolling(20).mean()
    boll_std = df["close"].rolling(20).std()
    df["boll_upper"] = df["boll_mid"] + 2 * boll_std
    df["boll_lower"] = df["boll_mid"] - 2 * boll_std
    df["boll_width"] = (df["boll_upper"] - df["boll_lower"]) / df["boll_mid"]
    df["boll_width_rank_120"] = df["boll_width"].rolling(120).rank(pct=True)
    df["boll_breakout"] = (
        (df["close"] > df["boll_upper"])
        & (df["amount_ratio_5_20"] >= 1.2)
    )

    # ADX / DMI
    df["plus_di14"], df["minus_di14"], df["adx14"] = calc_adx_dmi(df, 14)
    df["dmi_trend_ok"] = (
        (df["adx14"] >= 20)
        & (df["plus_di14"] > df["minus_di14"])
    )

    return df


# =========================================================
# 3. RS 相对强度线
# =========================================================

def calc_rs_latest(
    stock_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    signal_date: pd.Timestamp,
) -> dict:
    signal_date = pd.Timestamp(signal_date).normalize()

    s = stock_df[stock_df["date"] <= signal_date][["date", "close"]].copy()
    b = bench_df[bench_df["date"] <= signal_date][["date", "close"]].copy()

    if s.empty or b.empty:
        return {
            "rs_close": np.nan,
            "rs_ma20": np.nan,
            "rs_slope_20": np.nan,
            "rs_new_high_60": False,
            "rs_trend_ok": False,
        }

    s = s.rename(columns={"close": "stock_close"})
    b = b.rename(columns={"close": "bench_close"})

    m = s.merge(b, on="date", how="inner")
    if len(m) < 60:
        return {
            "rs_close": np.nan,
            "rs_ma20": np.nan,
            "rs_slope_20": np.nan,
            "rs_new_high_60": False,
            "rs_trend_ok": False,
        }

    m["rs_close"] = m["stock_close"] / m["bench_close"]
    m["rs_ma20"] = m["rs_close"].rolling(20).mean()
    m["rs_slope_20"] = m["rs_close"] / m["rs_close"].shift(20) - 1
    m["rs_high60"] = m["rs_close"].rolling(60).max()
    m["rs_new_high_60"] = m["rs_close"] >= m["rs_high60"] * 0.995
    m["rs_trend_ok"] = (
        (m["rs_close"] > m["rs_ma20"])
        & (m["rs_slope_20"] > 0)
    )

    latest = m.iloc[-1]

    return {
        "rs_close": latest.get("rs_close", np.nan),
        "rs_ma20": latest.get("rs_ma20", np.nan),
        "rs_slope_20": latest.get("rs_slope_20", np.nan),
        "rs_new_high_60": bool(latest.get("rs_new_high_60", False)),
        "rs_trend_ok": bool(latest.get("rs_trend_ok", False)),
    }


# =========================================================
# 4. 过热判断增强
# =========================================================

def is_overheat(row: pd.Series) -> bool:
    if not pd.isna(row.get("ret10", np.nan)) and row["ret10"] > 0.30:
        return True

    if not pd.isna(row.get("ret20", np.nan)) and row["ret20"] > 0.55:
        return True

    if not pd.isna(row.get("limit_up_count_5", np.nan)) and row["limit_up_count_5"] >= 3:
        return True

    if bool(row.get("rsi_overheat", False)) and row.get("ret5", 0) > 0.12:
        return True

    if bool(row.get("boll_breakout", False)) and row.get("ret10", 0) > 0.25:
        return True

    return False


# =========================================================
# 5. 模块化评分
# =========================================================

def calc_score(row: pd.Series, bench_ret20: float, bench_ret60: float) -> int:
    """
    模块化评分，总分 100。

    模块：
    1. 中期趋势结构：30
    2. 相对强度：20
    3. 量价配合：15
    4. 短线结构：10
    5. 风险控制：15
    6. 市场行业：10
    """

    score_trend = 0
    score_rs = 0
    score_volume = 0
    score_short = 0
    score_risk = 0
    score_env = 0

    close = _num(row, "close")
    ma5 = _num(row, "ma5")
    ma10 = _num(row, "ma10")
    ma20 = _num(row, "ma20")
    ma60 = _num(row, "ma60")
    ma120 = _num(row, "ma120")

    # 1. 中期趋势结构 30
    if close > ma20:
        score_trend += 5
    if ma20 > ma60:
        score_trend += 7
    if ma60 > ma120:
        score_trend += 6
    if _num(row, "ma20_slope_5") > 0:
        score_trend += 5
    if _num(row, "ma60_slope_10") > 0:
        score_trend += 4
    if _bool(row, "dmi_trend_ok"):
        score_trend += 3

    score_trend = min(score_trend, 30)

    # 2. 相对强度 20
    if _num(row, "ret20_rank_pct") >= 0.70:
        score_rs += 5
    elif _num(row, "ret20_rank_pct") >= 0.60:
        score_rs += 3

    if _num(row, "ret60_rank_pct") >= 0.70:
        score_rs += 5
    elif _num(row, "ret60_rank_pct") >= 0.60:
        score_rs += 3

    if not pd.isna(bench_ret20):
        if _num(row, "ret20") > bench_ret20 + 0.05:
            score_rs += 3
        elif _num(row, "ret20") > bench_ret20:
            score_rs += 2

    if not pd.isna(bench_ret60):
        if _num(row, "ret60") > bench_ret60 + 0.10:
            score_rs += 3
        elif _num(row, "ret60") > bench_ret60:
            score_rs += 2

    if _bool(row, "rs_trend_ok"):
        score_rs += 3

    if _bool(row, "rs_new_high_60"):
        score_rs += 2

    score_rs = min(score_rs, 20)

    # 3. 量价配合 15
    ratio = _num(row, "amount_ratio_5_20")

    if 1.2 <= ratio <= 4:
        score_volume += 5
    elif 1.0 <= ratio < 1.2:
        score_volume += 3
    elif 0.8 <= ratio < 1.0:
        score_volume += 1

    if _num(row, "up_down_amount_ratio20") > 1.2:
        score_volume += 4
    elif _num(row, "up_down_amount_ratio20") > 1.0:
        score_volume += 2

    if _bool(row, "volume_price_confirm"):
        score_volume += 3

    if _bool(row, "breakout_volume_ok"):
        score_volume += 2

    if ratio < 4:
        score_volume += 1

    score_volume = min(score_volume, 15)

    # 4. 短线结构 10
    if close > ma5:
        score_short += 1
    if ma5 > ma10:
        score_short += 1
    if ma10 > ma20:
        score_short += 1
    if _num(row, "ma5_slope_1") > 0:
        score_short += 1
    if _num(row, "ma5_slope_delta") > 0:
        score_short += 1
    if _bool(row, "ma5_turn_up"):
        score_short += 2
    if _bool(row, "ma5_slope_accel_3"):
        score_short += 2
    if _bool(row, "short_term_strong"):
        score_short += 2
    if _bool(row, "pullback_rebound"):
        score_short += 2

    score_short = min(score_short, 10)

    # 5. 风险控制 15
    max_dd20 = _num(row, "max_dd20")
    atr20_pct = _num(row, "atr20_pct")

    if not pd.isna(max_dd20):
        if max_dd20 > -0.10:
            score_risk += 4
        elif max_dd20 > -0.15:
            score_risk += 3
        elif max_dd20 > -0.20:
            score_risk += 1

    if not pd.isna(atr20_pct):
        if atr20_pct <= 0.035:
            score_risk += 3
        elif atr20_pct <= 0.055:
            score_risk += 2
        elif atr20_pct <= 0.08:
            score_risk += 1
    else:
        score_risk += 1

    if not is_overheat(row):
        score_risk += 3

    if not _bool(row, "heavy_bearish_candle"):
        score_risk += 3

    turnover_ma5 = _num(row, "turnover_ma5")
    if pd.isna(turnover_ma5):
        score_risk += 1
    elif turnover_ma5 < 15:
        score_risk += 2
    elif turnover_ma5 < 25:
        score_risk += 1

    score_risk = min(score_risk, 15)

    # 6. 市场行业环境 10
    if base.MARKET_FILTER_ENABLED:
        if _bool(row, "market_ok"):
            score_env += 4
    else:
        score_env += 3

    if _bool(row, "industry_strong", True):
        score_env += 3

    if _bool(row, "breakout_valid"):
        score_env += 2

    if _bool(row, "pullback_quality"):
        score_env += 1

    score_env = min(score_env, 10)

    total = (
        score_trend
        + score_rs
        + score_volume
        + score_short
        + score_risk
        + score_env
    )

    return int(max(0, min(total, 100)))


# =========================================================
# 6. 分类增强
# =========================================================

def classify_status(row: pd.Series) -> str:
    if not row["basic_liquid"]:
        return "流动性不足"

    if base.MARKET_FILTER_ENABLED and not row.get("market_ok", False):
        return "市场环境弱观察"

    if row["close"] < row["ma60"]:
        return "跌破MA60剔除"

    if row["close"] < row["ma20"]:
        return "跌破MA20观察"

    if row["heavy_bearish_candle"]:
        return "放量长阴观察"

    if is_overheat(row):
        return "短期过热"

    if bool(row.get("trend_confirmed_by_score", False)):
        return "短中线转强池"

    if bool(row.get("short_term_start", False)) and row.get("score", 0) >= 70:
        return "短线启动池"

    if (
        row.get("short_term_strong", False)
        and row.get("basic_liquid", False)
        and row.get("trend_basic", False)
        and row.get("risk_ok", False)
        and row.get("short_term_ok", False)
        and row.get("ret20_rank_pct", 0) >= 0.70
        and row.get("score", 0) >= base.MIN_SCORE
    ):
        return "短线强势池"

    if row.get("pullback_rebound", False) and row["score"] >= base.MIN_SCORE:
        return "回踩修复观察"

    if row["score"] >= base.STRONG_SCORE and row["candidate"]:
        return "强趋势池"

    if row["score"] >= base.MIN_SCORE and row["candidate"]:
        return "趋势观察池"

    if (
        row["close"] > row["ma60"]
        and row["ma20"] > row["ma60"]
        and abs(row["close"] / row["ma20"] - 1) <= 0.05
        and row["ret60_rank_pct"] >= 0.60
    ):
        return "回踩观察"

    return "剔除"


def get_status_order():
    return {
        "短中线转强池": 1,
        "短线启动池": 2,
        "短线强势池": 3,
        "强趋势池": 4,
        "趋势观察池": 5,
        "回踩修复观察": 6,
        "回踩观察": 7,
        "跌破MA20观察": 8,
        "放量长阴观察": 9,
        "短期过热": 10,
        "市场环境弱观察": 11,
        "跌破MA60剔除": 12,
        "流动性不足": 13,
        "剔除": 14,
    }


# =========================================================
# 7. 构建股票池增强
# =========================================================

def build_pool_on_date(
    signal_date: pd.Timestamp,
    stock_data,
    bench_df: pd.DataFrame,
    code_name_map=None,
    industry_map=None,
    classify: bool = True,
) -> pd.DataFrame:

    if code_name_map is None:
        code_name_map = {}

    if industry_map is None:
        industry_map = {}

    signal_date = pd.Timestamp(signal_date).normalize()

    market_state = base.get_market_state(signal_date, bench_df)
    bench_ret20 = market_state.get("bench_ret20", np.nan)
    bench_ret60 = market_state.get("bench_ret60", np.nan)

    required_indicator_cols = [
        "close",
        "ma5", "ma10", "ma20", "ma60", "ma120",
        "ma5_slope_3", "ma10_slope_5",
        "ma20_slope_5", "ma60_slope_10",
        "ret5", "ret10", "ret20", "ret60",
        "amount_ma20", "amount_ratio_5_20",
        "high20", "high60", "dist_to_60d_high",
        "max_dd20", "up_down_amount_ratio20",
        "atr20", "atr20_pct",
    ]

    rows = []

    for code, df in stock_data.items():
        if df is None or df.empty:
            continue

        df_slice = df[df["date"] <= signal_date]

        if len(df_slice) < base.MIN_BARS:
            continue

        latest = df_slice.iloc[-1]

        if pd.Timestamp(latest["date"]).normalize() != signal_date:
            continue

        has_nan = False
        for col in required_indicator_cols:
            if col not in latest.index or pd.isna(latest[col]):
                has_nan = True
                break

        if has_nan:
            continue

        code = base.normalize_code(code)
        industry = industry_map.get(code, "")

        rs_info = calc_rs_latest(
            stock_df=df,
            bench_df=bench_df,
            signal_date=signal_date,
        )

        rows.append({
            "code": code,
            "name": code_name_map.get(code, ""),
            "industry": industry,
            "signal_date": signal_date.strftime("%Y-%m-%d"),

            "close": latest["close"],

            "ma5": latest["ma5"],
            "ma10": latest["ma10"],
            "ma20": latest["ma20"],
            "ma60": latest["ma60"],
            "ma120": latest["ma120"],

            "ma5_slope_1": latest.get("ma5_slope_1", np.nan),
            "ma5_slope_2": latest.get("ma5_slope_2", np.nan),
            "ma5_slope_3": latest.get("ma5_slope_3", np.nan),
            "ma5_slope_delta": latest.get("ma5_slope_delta", np.nan),
            "ma5_turn_up": bool(latest.get("ma5_turn_up", False)),
            "ma5_slope_accel_3": bool(latest.get("ma5_slope_accel_3", False)),
            "ma5_short_bull": bool(latest.get("ma5_short_bull", False)),
            "ma5_timing_signal": bool(latest.get("ma5_timing_signal", False)),

            "ma10_slope_5": latest["ma10_slope_5"],
            "ma20_slope_5": latest["ma20_slope_5"],
            "ma60_slope_10": latest["ma60_slope_10"],

            "ret5": latest["ret5"],
            "ret10": latest["ret10"],
            "ret20": latest["ret20"],
            "ret60": latest["ret60"],

            "amount_ma5": latest["amount_ma5"],
            "amount_ma10": latest["amount_ma10"],
            "amount_ma20": latest["amount_ma20"],
            "amount_ratio_5_20": latest["amount_ratio_5_20"],

            "turnover_ma5": latest.get("turnover_ma5", np.nan),
            "turnover_ma20": latest.get("turnover_ma20", np.nan),

            "high20": latest["high20"],
            "high60": latest["high60"],
            "prev_high20": latest.get("prev_high20", np.nan),
            "prev_high60": latest.get("prev_high60", np.nan),

            "is_20d_high": bool(latest["is_20d_high"]),
            "is_60d_high": bool(latest["is_60d_high"]),
            "dist_to_60d_high": latest["dist_to_60d_high"],

            "breakout_20": bool(latest.get("breakout_20", False)),
            "breakout_60": bool(latest.get("breakout_60", False)),
            "breakout_volume_ok": bool(latest.get("breakout_volume_ok", False)),
            "breakout_valid": bool(latest.get("breakout_valid", False)),

            "max_dd20": latest["max_dd20"],
            "atr20": latest["atr20"],
            "atr20_pct": latest["atr20_pct"],

            "up_down_amount_ratio20": latest["up_down_amount_ratio20"],
            "volume_price_confirm": bool(latest.get("volume_price_confirm", False)),

            "limit_up_count_5": latest.get("limit_up_count_5", np.nan),
            "heavy_bearish_candle": bool(latest.get("heavy_bearish_candle", False)),

            "short_term_strong": bool(latest.get("short_term_strong_raw", False)),
            "pullback_rebound": bool(latest.get("pullback_rebound_raw", False)),

            "pullback_volume_shrink": bool(latest.get("pullback_volume_shrink", False)),
            "pullback_depth_10": latest.get("pullback_depth_10", np.nan),
            "pullback_depth_20": latest.get("pullback_depth_20", np.nan),
            "pullback_quality": bool(latest.get("pullback_quality", False)),

            "macd_dif": latest.get("macd_dif", np.nan),
            "macd_dea": latest.get("macd_dea", np.nan),
            "macd_hist": latest.get("macd_hist", np.nan),
            "macd_gold_cross": bool(latest.get("macd_gold_cross", False)),
            "macd_above_zero": bool(latest.get("macd_above_zero", False)),
            "macd_hist_rising": bool(latest.get("macd_hist_rising", False)),

            "rsi6": latest.get("rsi6", np.nan),
            "rsi14": latest.get("rsi14", np.nan),
            "rsi_overheat": bool(latest.get("rsi_overheat", False)),
            "rsi_strength": bool(latest.get("rsi_strength", False)),

            "boll_mid": latest.get("boll_mid", np.nan),
            "boll_upper": latest.get("boll_upper", np.nan),
            "boll_lower": latest.get("boll_lower", np.nan),
            "boll_width": latest.get("boll_width", np.nan),
            "boll_width_rank_120": latest.get("boll_width_rank_120", np.nan),
            "boll_breakout": bool(latest.get("boll_breakout", False)),

            "plus_di14": latest.get("plus_di14", np.nan),
            "minus_di14": latest.get("minus_di14", np.nan),
            "adx14": latest.get("adx14", np.nan),
            "dmi_trend_ok": bool(latest.get("dmi_trend_ok", False)),

            "rs_close": rs_info["rs_close"],
            "rs_ma20": rs_info["rs_ma20"],
            "rs_slope_20": rs_info["rs_slope_20"],
            "rs_new_high_60": rs_info["rs_new_high_60"],
            "rs_trend_ok": rs_info["rs_trend_ok"],

            "bench_ret20": bench_ret20,
            "bench_ret60": bench_ret60,

            "market_ok": market_state.get("market_ok", False),
            "market_state": market_state.get("market_state", ""),
        })

    if not rows:
        return pd.DataFrame()

    df_pool = pd.DataFrame(rows)

    df_pool["ret20_rank_pct"] = df_pool["ret20"].rank(pct=True)
    df_pool["ret60_rank_pct"] = df_pool["ret60"].rank(pct=True)

    if (
        base.INDUSTRY_FILTER_ENABLED
        and "industry" in df_pool.columns
        and df_pool["industry"].astype(str).str.len().gt(0).any()
    ):
        industry_strength = (
            df_pool[df_pool["industry"].astype(str).str.len() > 0]
            .groupby("industry")
            .agg(
                industry_ret20=("ret20", "mean"),
                industry_ret60=("ret60", "mean"),
                industry_count=("code", "count"),
            )
            .reset_index()
        )

        industry_strength["industry_ret20_rank_pct"] = industry_strength["industry_ret20"].rank(pct=True)
        industry_strength["industry_ret60_rank_pct"] = industry_strength["industry_ret60"].rank(pct=True)

        df_pool = df_pool.merge(industry_strength, on="industry", how="left")

        df_pool["industry_strong"] = (
            (df_pool["industry_ret20_rank_pct"] >= 0.60)
            & (df_pool["industry_ret60_rank_pct"] >= 0.60)
        )

        df_pool["industry_strong"] = df_pool["industry_strong"].fillna(True)

    else:
        df_pool["industry_ret20"] = np.nan
        df_pool["industry_ret60"] = np.nan
        df_pool["industry_ret20_rank_pct"] = np.nan
        df_pool["industry_ret60_rank_pct"] = np.nan
        df_pool["industry_count"] = np.nan
        df_pool["industry_strong"] = True

    df_pool["basic_liquid"] = df_pool["amount_ma20"] >= base.MIN_AVG_AMOUNT_20

    df_pool["trend_basic"] = (
        (df_pool["close"] > df_pool["ma20"])
        & (df_pool["ma20"] > df_pool["ma60"])
        & (df_pool["ma60"] > df_pool["ma120"])
        & (df_pool["ma20_slope_5"] > 0)
        & (df_pool["ma60_slope_10"] > 0)
    )

    df_pool["relative_strength"] = (
        (
            (df_pool["ret20_rank_pct"] >= 0.70)
            & (df_pool["ret60_rank_pct"] >= 0.70)
        )
        | df_pool["rs_trend_ok"]
        | df_pool["rs_new_high_60"]
    )

    df_pool["volume_ok"] = (
        (df_pool["amount_ratio_5_20"] >= 1.0)
        & (df_pool["amount_ratio_5_20"] <= 4)
    )

    df_pool["near_breakout"] = (
        (df_pool["dist_to_60d_high"] >= -0.05)
        | df_pool["is_20d_high"]
        | df_pool["is_60d_high"]
        | df_pool["breakout_20"]
        | df_pool["breakout_60"]
    )

    df_pool["risk_ok"] = (
        (df_pool["close"] > df_pool["ma20"])
        & (df_pool["max_dd20"] > -0.20)
        & (~df_pool["heavy_bearish_candle"])
        & (df_pool["atr20_pct"] <= 0.10)
        & (~df_pool["rsi_overheat"])
    )

    df_pool["short_term_ok"] = (
        df_pool["short_term_strong"]
        | df_pool["pullback_rebound"]
        | df_pool["ma5_timing_signal"]
        | (
            (df_pool["close"] > df_pool["ma10"])
            & (df_pool["ma10"] > df_pool["ma20"])
        )
    )

    df_pool["short_term_start"] = (
        df_pool["basic_liquid"]
        & (df_pool["close"] > df_pool["ma20"])
        & (df_pool["ma20"] > df_pool["ma60"])
        & (df_pool["amount_ratio_5_20"] >= 1.0)
        & (df_pool["ret10"] < 0.30)
        & (~df_pool["heavy_bearish_candle"])
        & (
            df_pool["ma5_timing_signal"]
            | df_pool["ma5_turn_up"]
            | df_pool["ma5_slope_accel_3"]
        )
        & df_pool["ma5_short_bull"]
    )

    df_pool["candidate"] = (
        df_pool["basic_liquid"]
        & df_pool["trend_basic"]
        & df_pool["relative_strength"]
        & df_pool["volume_ok"]
        & df_pool["near_breakout"]
        & df_pool["risk_ok"]
        & df_pool["industry_strong"]
    )

    if base.MARKET_FILTER_ENABLED:
        df_pool["candidate"] = df_pool["candidate"] & df_pool["market_ok"]

    df_pool["score"] = df_pool.apply(
        lambda row: calc_score(row, bench_ret20, bench_ret60),
        axis=1,
    )

    df_pool["trend_confirmed_by_score"] = False

    df_pool["status"] = df_pool.apply(classify_status, axis=1)

    df_pool = sort_pool_df(df_pool)

    return df_pool


# =========================================================
# 8. 评分趋势 + 状态迁移增强
# =========================================================

def classify_score_trend(g: pd.DataFrame) -> pd.Series:
    g = g.sort_values("signal_date").copy()

    scores = g["score"].dropna().astype(float).tolist()
    statuses = g["status"].astype(str).tolist()

    empty_result = {
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

        "status_1d_ago": "",
        "status_3d_ago": "",
        "status_5d_ago": "",
        "status_changed": False,
        "status_upgrade": False,
        "status_downgrade": False,
        "status_path_5d": "",
    }

    if len(scores) < 4:
        return pd.Series(empty_result)

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

    score_rise_days_5 = 0
    if len(last_scores) >= 2:
        for i in range(1, len(last_scores)):
            if last_scores[i] > last_scores[i - 1]:
                score_rise_days_5 += 1

    c1 = 0 if pd.isna(score_change_1d) else score_change_1d
    c3 = 0 if pd.isna(score_change_3d) else score_change_3d
    c5 = 0 if pd.isna(score_change_5d) else score_change_5d

    score_momentum = (
        c5 * 0.55
        + c3 * 0.30
        + c1 * 0.15
        + score_rise_days_5 * 2
    )

    status_1d_ago = statuses[-2] if len(statuses) >= 2 else statuses[0]
    status_3d_ago = statuses[-4] if len(statuses) >= 4 else statuses[0]
    status_5d_ago = statuses[-6] if len(statuses) >= 6 else statuses[0]
    status_now = statuses[-1]

    status_changed = status_now != status_5d_ago

    status_order = get_status_order()
    order_now = status_order.get(status_now, 99)
    order_5d = status_order.get(status_5d_ago, 99)

    status_upgrade = order_now < order_5d
    status_downgrade = order_now > order_5d

    status_path_5d = " > ".join(statuses[-5:]) if len(statuses) >= 5 else " > ".join(statuses)

    if score_now >= 85 and score_min5 >= 75 and score_ma3 >= 85:
        trend_status = "强趋势延续"
    elif score_now >= 70 and score_change_5d >= 20 and score_ma3 > score_ma5:
        trend_status = "趋势转强"
    elif score_now >= 60 and score_change_5d >= 15:
        trend_status = "趋势修复"
    elif score_5d_ago >= 85 and score_now <= 70 and score_ma3 < score_ma5:
        trend_status = "高位转弱"
    elif score_now < 50 and score_ma5 < 50:
        trend_status = "弱势延续"
    else:
        trend_status = "震荡无趋势"

    return pd.Series({
        "score_trend_status": trend_status,

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

        "status_1d_ago": status_1d_ago,
        "status_3d_ago": status_3d_ago,
        "status_5d_ago": status_5d_ago,
        "status_changed": status_changed,
        "status_upgrade": status_upgrade,
        "status_downgrade": status_downgrade,
        "status_path_5d": status_path_5d,
    })


# =========================================================
# 9. 排序增强
# =========================================================

def sort_pool_df(df_pool: pd.DataFrame) -> pd.DataFrame:
    if df_pool is None or df_pool.empty:
        return pd.DataFrame()

    status_order = get_status_order()

    df_pool = df_pool.copy()
    df_pool["_order"] = df_pool["status"].map(status_order).fillna(99)

    sort_cols = ["_order"]
    ascending = [True]

    for col in [
        "status_upgrade",
        "short_term_start",
        "score_momentum",
        "score_change_5d",
        "score_change_3d",
        "score_change_1d",
        "score",
        "rs_slope_20",
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
        "rs_slope_20",
    ]:
        if col in df.columns:
            sort_cols.append(col)
            ascending.append(False)

    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    elif "score" in df.columns:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)

    return df


# =========================================================
# 10. 排名榜单
# =========================================================

def build_score_change_rank_df(pool_df: pd.DataFrame) -> pd.DataFrame:
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

    df = df[df["score_change_5d"].notna()].copy()

    if df.empty:
        return pd.DataFrame()

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

def build_short_start_rank_df(pool_df: pd.DataFrame) -> pd.DataFrame:
    """
    短线启动观察榜单。
    筛选 short_term_start == True 的股票，并按评分动量排序。
    """

    if pool_df is None or pool_df.empty:
        return pd.DataFrame()

    if "short_term_start" not in pool_df.columns:
        return pd.DataFrame()

    df = pool_df[pool_df["short_term_start"] == True].copy()

    if df.empty:
        return pd.DataFrame()

    sort_cols = []
    ascending = []

    for col in [
        "score_momentum",
        "score_change_5d",
        "score_change_3d",
        "score_change_1d",
        "score",
        "rs_slope_20",
    ]:
        if col in df.columns:
            sort_cols.append(col)
            ascending.append(False)

    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    elif "score" in df.columns:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)

    return df


def build_status_upgrade_rank_df(pool_df: pd.DataFrame) -> pd.DataFrame:
    """
    状态升级观察榜单。
    筛选 status_upgrade == True 的股票，并按评分动量排序。
    """

    if pool_df is None or pool_df.empty:
        return pd.DataFrame()

    if "status_upgrade" not in pool_df.columns:
        return pd.DataFrame()

    df = pool_df[pool_df["status_upgrade"] == True].copy()

    if df.empty:
        return pd.DataFrame()

    sort_cols = []
    ascending = []

    for col in [
        "score_momentum",
        "score_change_5d",
        "score_change_3d",
        "score_change_1d",
        "score",
        "rs_slope_20",
    ]:
        if col in df.columns:
            sort_cols.append(col)
            ascending.append(False)

    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    elif "score" in df.columns:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)

    return df

def build_short_start_df(pool_df: pd.DataFrame) -> pd.DataFrame:
    if pool_df is None or pool_df.empty:
        return pd.DataFrame()

    if "short_term_start" not in pool_df.columns:
        return pd.DataFrame()

    df = pool_df[pool_df["short_term_start"] == True].copy()

    if df.empty:
        return pd.DataFrame()

    return sort_by_score_momentum(df)


def build_status_upgrade_rank_df(pool_df: pd.DataFrame) -> pd.DataFrame:
    if pool_df is None or pool_df.empty:
        return pd.DataFrame()

    if "status_upgrade" not in pool_df.columns:
        return pd.DataFrame()

    df = pool_df[pool_df["status_upgrade"] == True].copy()

    if df.empty:
        return pd.DataFrame()

    return sort_by_score_momentum(df)


# =========================================================
# 11. 可读输出增强
# =========================================================

def make_classification_readable(pool_df: pd.DataFrame) -> pd.DataFrame:
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

        "status_1d_ago",
        "status_3d_ago",
        "status_5d_ago",
        "status_changed",
        "status_upgrade",
        "status_downgrade",
        "status_path_5d",

        "trend_confirmed_by_score",

        "market_state", "market_ok",

        "close",
        "ma5", "ma10", "ma20", "ma60", "ma120",
        "ma5_slope_1",
        "ma5_slope_2",
        "ma5_slope_3",
        "ma5_slope_delta",
        "ma5_turn_up",
        "ma5_slope_accel_3",
        "ma5_short_bull",
        "ma5_timing_signal",
        "short_term_start",

        "ma10_slope_5",
        "ma20_slope_5",
        "ma60_slope_10",

        "ret5", "ret10", "ret20", "ret60",
        "ret20_rank_pct", "ret60_rank_pct",

        "rs_close",
        "rs_ma20",
        "rs_slope_20",
        "rs_new_high_60",
        "rs_trend_ok",

        "industry_ret20", "industry_ret60",
        "industry_ret20_rank_pct", "industry_ret60_rank_pct",
        "industry_strong",

        "amount_ma5", "amount_ma10", "amount_ma20", "amount_ratio_5_20",
        "turnover_ma5", "turnover_ma20",
        "up_down_amount_ratio20",
        "volume_price_confirm",

        "is_20d_high", "is_60d_high", "dist_to_60d_high",
        "prev_high20", "prev_high60",
        "breakout_20", "breakout_60",
        "breakout_volume_ok", "breakout_valid",

        "pullback_volume_shrink",
        "pullback_depth_10",
        "pullback_depth_20",
        "pullback_quality",

        "max_dd20", "atr20", "atr20_pct",

        "limit_up_count_5", "heavy_bearish_candle",
        "short_term_strong", "pullback_rebound",

        "macd_dif", "macd_dea", "macd_hist",
        "macd_gold_cross", "macd_above_zero", "macd_hist_rising",

        "rsi6", "rsi14", "rsi_overheat", "rsi_strength",

        "boll_mid", "boll_upper", "boll_lower",
        "boll_width", "boll_width_rank_120", "boll_breakout",

        "plus_di14", "minus_di14", "adx14", "dmi_trend_ok",

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

        "status_1d_ago": "1日前分类",
        "status_3d_ago": "3日前分类",
        "status_5d_ago": "5日前分类",
        "status_changed": "分类是否变化",
        "status_upgrade": "状态是否升级",
        "status_downgrade": "状态是否降级",
        "status_path_5d": "近5日状态路径",

        "trend_confirmed_by_score": "评分确认转强",

        "market_state": "市场状态",
        "market_ok": "大盘环境通过",

        "close": "收盘价",
        "ma5": "MA5",
        "ma10": "MA10",
        "ma20": "MA20",
        "ma60": "MA60",
        "ma120": "MA120",

        "ma5_slope_1": "MA5近1日斜率",
        "ma5_slope_2": "MA5近2日斜率",
        "ma5_slope_3": "MA5近3日斜率",
        "ma5_slope_delta": "MA5斜率变化",
        "ma5_turn_up": "MA5斜率由负转正",
        "ma5_slope_accel_3": "MA5斜率连续增强",
        "ma5_short_bull": "MA5短线多头",
        "ma5_timing_signal": "MA5短线时机信号",
        "short_term_start": "短线启动信号",

        "ma10_slope_5": "MA10近5日斜率",
        "ma20_slope_5": "MA20近5日斜率",
        "ma60_slope_10": "MA60近10日斜率",

        "ret5": "近5日涨幅",
        "ret10": "近10日涨幅",
        "ret20": "近20日涨幅",
        "ret60": "近60日涨幅",
        "ret20_rank_pct": "20日强度排名百分位",
        "ret60_rank_pct": "60日强度排名百分位",

        "rs_close": "RS相对强度",
        "rs_ma20": "RS_MA20",
        "rs_slope_20": "RS近20日斜率",
        "rs_new_high_60": "RS近60日新高",
        "rs_trend_ok": "RS趋势达标",

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
        "up_down_amount_ratio20": "近20日上涨/下跌成交额比",
        "volume_price_confirm": "量价确认",

        "is_20d_high": "是否接近20日新高",
        "is_60d_high": "是否近60日新高",
        "dist_to_60d_high": "距60日高点",
        "prev_high20": "前20日高点",
        "prev_high60": "前60日高点",
        "breakout_20": "突破20日高点",
        "breakout_60": "突破60日高点",
        "breakout_volume_ok": "突破量能达标",
        "breakout_valid": "突破有效",

        "pullback_volume_shrink": "回踩缩量",
        "pullback_depth_10": "回踩MA10深度",
        "pullback_depth_20": "回踩MA20深度",
        "pullback_quality": "回踩质量",

        "max_dd20": "近20日最大回撤",
        "atr20": "ATR20",
        "atr20_pct": "ATR20比例",

        "limit_up_count_5": "近5日涨停次数",
        "heavy_bearish_candle": "是否放量长阴",
        "short_term_strong": "短线强势",
        "pullback_rebound": "回踩修复",

        "macd_dif": "MACD_DIF",
        "macd_dea": "MACD_DEA",
        "macd_hist": "MACD柱",
        "macd_gold_cross": "MACD金叉",
        "macd_above_zero": "MACD零轴上方",
        "macd_hist_rising": "MACD柱增强",

        "rsi6": "RSI6",
        "rsi14": "RSI14",
        "rsi_overheat": "RSI过热",
        "rsi_strength": "RSI强势",

        "boll_mid": "BOLL中轨",
        "boll_upper": "BOLL上轨",
        "boll_lower": "BOLL下轨",
        "boll_width": "BOLL宽度",
        "boll_width_rank_120": "BOLL宽度120日分位",
        "boll_breakout": "BOLL突破",

        "plus_di14": "+DI14",
        "minus_di14": "-DI14",
        "adx14": "ADX14",
        "dmi_trend_ok": "DMI趋势达标",

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
# 12. Excel 输出增强
# =========================================================

def save_classification_excel(
    output_xlsx: str,
    signal_date: pd.Timestamp,
    pool_df: pd.DataFrame,
) -> None:

    readable_df = make_classification_readable(pool_df)
    summary_df = base.build_classification_summary(pool_df)

    score_change_rank_df = build_score_change_rank_df(pool_df)
    score_change_readable_df = make_classification_readable(score_change_rank_df)

    short_start_df = build_short_start_rank_df(pool_df)
    short_start_readable_df = make_classification_readable(short_start_df)

    upgrade_df = build_status_upgrade_rank_df(pool_df)
    upgrade_readable_df = make_classification_readable(upgrade_df)

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
    dashboard_rows.append(["新增短线信号", "MA5斜率转正/加速", "用于识别短线启动点"])
    dashboard_rows.append(["新增相对强度", "RS Line", "股票收盘价 / 沪深300收盘价"])
    dashboard_rows.append(["新增趋势强度", "ADX / DMI", "过滤无趋势震荡"])
    dashboard_rows.append(["新增排序方式", "评分动量排序", "分类内部按近期评分变化排序"])

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

        if short_start_readable_df is not None and not short_start_readable_df.empty:
            short_start_readable_df.to_excel(writer, sheet_name="短线启动观察", index=False)

        if upgrade_readable_df is not None and not upgrade_readable_df.empty:
            upgrade_readable_df.to_excel(writer, sheet_name="状态升级观察", index=False)

        important_statuses = list(get_status_order().keys())

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
# 13. 分类统计增强
# =========================================================

def build_classification_summary(pool_df: pd.DataFrame) -> pd.DataFrame:
    if pool_df is None or pool_df.empty:
        return pd.DataFrame()

    total = len(pool_df)

    agg_dict = dict(
        数量=("code", "count"),
        平均评分=("score", "mean"),
        最高评分=("score", "max"),
        平均20日涨幅=("ret20", "mean"),
        平均60日涨幅=("ret60", "mean"),
        平均20日成交额=("amount_ma20", "mean"),
        平均量能比=("amount_ratio_5_20", "mean"),
        平均20日最大回撤=("max_dd20", "mean"),
        平均ATR比例=("atr20_pct", "mean"),
    )

    if "score_change_5d" in pool_df.columns:
        agg_dict["平均5日评分变化"] = ("score_change_5d", "mean")

    if "score_momentum" in pool_df.columns:
        agg_dict["平均评分动量"] = ("score_momentum", "mean")

    if "rs_slope_20" in pool_df.columns:
        agg_dict["平均RS20日斜率"] = ("rs_slope_20", "mean")

    if "adx14" in pool_df.columns:
        agg_dict["平均ADX14"] = ("adx14", "mean")

    summary = (
        pool_df
        .groupby("status")
        .agg(**agg_dict)
        .reset_index()
        .rename(columns={"status": "分类"})
    )

    summary["占比"] = summary["数量"] / total

    status_order = get_status_order()
    summary["排序"] = summary["分类"].map(status_order).fillna(99)
    summary = summary.sort_values(["排序", "数量"], ascending=[True, False])
    summary = summary.drop(columns=["排序"]).reset_index(drop=True)

    return summary


# =========================================================
# 14. 打补丁
# =========================================================

def patch_base_module():
    """
    将本文件中的增强函数覆盖到原策略模块里。
    """

    base.add_indicators = add_indicators

    base.calc_score = calc_score
    base.is_overheat = is_overheat
    base.classify_status = classify_status
    base.get_status_order = get_status_order

    base.build_pool_on_date = build_pool_on_date

    base.classify_score_trend = classify_score_trend

    base.sort_pool_df = sort_pool_df

    base.make_classification_readable = make_classification_readable
    base.save_classification_excel = save_classification_excel
    base.build_classification_summary = build_classification_summary

    print("已启用完整增强模块：")
    print("- MA5斜率转正 / MA5斜率加速 / 短线启动信号")
    print("- 评分动量排序")
    print("- 状态迁移：升级、降级、近5日路径")
    print("- RS相对强度线")
    print("- ADX / DMI 趋势强度")
    print("- MACD / RSI / BOLL 辅助指标")
    print("- 突破有效性")
    print("- 回踩质量")
    print("- 模块化评分")
    print("- 新增分类：短线启动池")
    print("- Excel新增 Sheet：评分变化排序、短线启动观察、状态升级观察")


# =========================================================
# 15. 主入口
# =========================================================

if __name__ == "__main__":
    patch_base_module()
    base.main()