import pandas as pd
from helpers import rsi, bollinger_bands, atr, ema

SYMBOLS        = ["XAU/USD"]
RSI_PERIOD     = 14
BB_PERIOD      = 20
BB_STDDEV      = 2
ATR_PERIOD     = 14

RSI_OVERSOLD   = 30
RSI_OVERBOUGHT = 70
RSI_BULL_ZONE  = 48
MIN_BODY_RATIO = 0.25
SL_MULTIPLIER  = 1.5
TP_MULTIPLIER  = 2.5

def _body_ratio(candle) -> float:
    total = candle["high"] - candle["low"]
    return abs(candle["close"] - candle["open"]) / total if total else 0

def _strong_candle(candle, direction: str) -> bool:
    if _body_ratio(candle) < MIN_BODY_RATIO:
        return False
    return (candle["close"] > candle["open"]) if direction == "BUY" \
           else (candle["close"] < candle["open"])

def _away_from_swing(df_1h, direction: str, lookback: int = 20) -> bool:
    """
    Price must be at least 2x average candle range away from swing high/low.
    Prevents entries right at support/resistance walls.
    """
    recent    = df_1h.iloc[-lookback:]
    price     = df_1h.iloc[-1]["close"]
    avg_range = (recent["high"] - recent["low"]).mean()
    if direction == "SELL":
        return price > recent["low"].min() + (avg_range * 2)
    return price < recent["high"].max() - (avg_range * 2)

def _daily_bias(df_1d) -> str | None:
    """
    Structural bias — daily BB midline + 5-candle majority vote.
    Prevents BUY signals in falling markets and vice versa.
    """
    if len(df_1d) < 5:
        return None
    last1d = df_1d.iloc[-1]
    if pd.isna(last1d.get("bb_mid", float("nan"))):
        return None
    price_above_mid = last1d["close"] > last1d["bb_mid"]
    bulls = sum(1 for _, c in df_1d.iloc[-5:].iterrows()
                if c["close"] > c["open"])
    if price_above_mid and bulls >= 3: return "BUY"
    if not price_above_mid and bulls <= 2: return "SELL"
    return None

def _weekly_bias(df_1w) -> str | None:
    """
    Weekly EMA20 structure filter.
    Price above weekly EMA20 = bullish, below = bearish.
    When weekly conflicts with daily — no trade.
    """
    if df_1w is None or len(df_1w) < 20:
        return None
    df_1w = df_1w.copy()
    df_1w["ema20"] = ema(df_1w["close"], 20)
    last = df_1w.iloc[-1]
    if pd.isna(last["ema20"]):
        return None
    return "BUY" if last["close"] > last["ema20"] else "SELL"

def generate_signal(df_1h, df_1d, df_1w=None, sentiment_bias: int = 0):
    """
    sentiment_bias: +1 bullish | -1 bearish | 0 neutral
    Returns (direction, last1h, signal_type, sl, tp)
    """
    # ── Indicators ──────────────────────────────────
    df_1h["rsi"] = rsi(df_1h["close"], RSI_PERIOD)
    df_1h["bb_upper"], df_1h["bb_mid"], df_1h["bb_lower"] = \
        bollinger_bands(df_1h["close"], BB_PERIOD, BB_STDDEV)
    df_1h["atr"] = atr(df_1h, ATR_PERIOD)

    df_1d["bb_upper"], df_1d["bb_mid"], df_1d["bb_lower"] = \
        bollinger_bands(df_1d["close"], BB_PERIOD, BB_STDDEV)

    last1h = df_1h.iloc[-1]
    last1d = df_1d.iloc[-1]

    rsi_val = last1h["rsi"]
    price   = last1h["close"]
    atr_val = last1h["atr"]

    if pd.isna(rsi_val) or pd.isna(atr_val) or atr_val == 0:
        return None, last1h, None, None, None

    # ── Daily bias ──────────────────────────────────
    daily_bias = _daily_bias(df_1d)
    if daily_bias is None:
        return None, last1h, None, None, None

    # ── Weekly bias — must align with daily ─────────
    weekly_bias = _weekly_bias(df_1w)
    if weekly_bias and weekly_bias != daily_bias:
        return None, last1h, None, None, None

    # ── Daily BB structure ───────────────────────────
    inside_daily_bb = last1d["bb_lower"] < last1d["close"] < last1d["bb_upper"]
    if not inside_daily_bb:
        return None, last1h, None, None, None

    # ── Signal Detection ────────────────────────────
    direction   = None
    signal_type = None

    if (daily_bias == "BUY"
            and price > last1h["bb_mid"]
            and RSI_BULL_ZONE < rsi_val < RSI_OVERBOUGHT
            and _strong_candle(last1h, "BUY")
            and _away_from_swing(df_1h, "BUY")):
        direction, signal_type = "BUY", "Trend"

    elif (daily_bias == "SELL"
            and price < last1h["bb_mid"]
            and RSI_OVERSOLD < rsi_val < RSI_BULL_ZONE
            and _strong_candle(last1h, "SELL")
            and _away_from_swing(df_1h, "SELL")):
        direction, signal_type = "SELL", "Trend"

    elif (daily_bias == "BUY"
            and price <= last1h["bb_lower"]
            and rsi_val <= RSI_OVERSOLD
            and _strong_candle(last1h, "BUY")):
        direction, signal_type = "BUY", "Reversal"

    elif (daily_bias == "SELL"
            and price >= last1h["bb_upper"]
            and rsi_val >= RSI_OVERBOUGHT
            and _strong_candle(last1h, "SELL")):
        direction, signal_type = "SELL", "Reversal"

    if not direction:
        return None, last1h, None, None, None

    # ── Sentiment Gate ──────────────────────────────
    if sentiment_bias == 1  and direction == "SELL":
        return None, last1h, None, None, None
    if sentiment_bias == -1 and direction == "BUY":
        return None, last1h, None, None, None

    # ── SL / TP ─────────────────────────────────────
    if direction == "BUY":
        sl = round(price - atr_val * SL_MULTIPLIER, 2)
        tp = round(price + atr_val * TP_MULTIPLIER, 2)
    else:
        sl = round(price + atr_val * SL_MULTIPLIER, 2)
        tp = round(price - atr_val * TP_MULTIPLIER, 2)

    return direction, last1h, signal_type, sl, tp
