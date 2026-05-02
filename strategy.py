from helpers import rsi, bollinger_bands, atr

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

def _daily_bias(df_1d) -> str | None:
    """2-candle majority vote on daily frame."""
    window = df_1d.iloc[-2:]
    if len(window) < 1:
        return None
    bulls = sum(1 for _, c in window.iterrows() if c["close"] > c["open"])
    if bulls >= 1: return "BUY"
    return "SELL"

def generate_signal(df_1h, df_1d, sentiment_bias: int = 0):
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

    if any(map(lambda x: x != x, [rsi_val, atr_val])) or atr_val == 0:
        return None, last1h, None, None, None

    # ── Filters ─────────────────────────────────────
    daily_bias = _daily_bias(df_1d)
    if daily_bias is None:
        return None, last1h, None, None, None

    inside_daily_bb = last1d["bb_lower"] < last1d["close"] < last1d["bb_upper"]
    if not inside_daily_bb:
        return None, last1h, None, None, None

    # ── Signal Detection ────────────────────────────
    direction  = None
    signal_type = None

    # Trend continuation — single candle confirmation
    if (daily_bias == "BUY"
            and price > last1h["bb_mid"]
            and RSI_BULL_ZONE < rsi_val < RSI_OVERBOUGHT
            and _strong_candle(last1h, "BUY")):
        direction, signal_type = "BUY", "Trend"

    elif (daily_bias == "SELL"
            and price < last1h["bb_mid"]
            and RSI_OVERSOLD < rsi_val < RSI_BULL_ZONE
            and _strong_candle(last1h, "SELL")):
        direction, signal_type = "SELL", "Trend"

    # Mean reversion at BB extremes
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
