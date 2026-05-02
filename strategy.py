from helpers import rsi, bollinger_bands

SYMBOLS = ["XAU/USD"]
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STDDEV = 2

# Tuned thresholds for XAU/USD
RSI_OVERSOLD = 35       # Gold needs wider bands than 30
RSI_OVERBOUGHT = 65     # Same here
RSI_BULL_ZONE = 50      # Trend filter: RSI above = bullish bias
MIN_BODY_RATIO = 0.4    # Candle must have meaningful body (not a doji)

def candle_body_ratio(candle):
    """Returns ratio of body to total candle range. Filters dojis."""
    total = abs(candle["high"] - candle["low"])
    if total == 0:
        return 0
    return abs(candle["close"] - candle["open"]) / total

def is_strong_candle(candle, direction):
    """Candle must close strongly in signal direction."""
    body_ok = candle_body_ratio(candle) >= MIN_BODY_RATIO
    if direction == "BUY":
        return candle["close"] > candle["open"] and body_ok
    return candle["close"] < candle["open"] and body_ok

def get_1d_bias(df_1d):
    """
    Multi-candle daily bias instead of single candle check.
    Uses last 3 daily candles to determine trend direction.
    """
    last3 = df_1d.iloc[-3:]
    bull_candles = sum(1 for _, c in last3.iterrows() if c["close"] > c["open"])
    if bull_candles >= 2:
        return "BUY"
    elif bull_candles <= 1:
        return "SELL"
    return None

def generate_signal(df_1h, df_1d):
    # --- Indicators ---
    df_1h["rsi"] = rsi(df_1h["close"], RSI_PERIOD)
    df_1h["bb_upper"], df_1h["bb_mid"], df_1h["bb_lower"] = bollinger_bands(
        df_1h["close"], BB_PERIOD, BB_STDDEV
    )
    df_1d["bb_upper"], df_1d["bb_mid"], df_1d["bb_lower"] = bollinger_bands(
        df_1d["close"], BB_PERIOD, BB_STDDEV
    )

    last1h = df_1h.iloc[-1]
    prev1h = df_1h.iloc[-2]   # Previous candle for confirmation
    last1d = df_1d.iloc[-1]

    rsi_val = last1h["rsi"]
    price    = last1h["close"]

    # --- 1. Daily bias (trend filter) ---
    daily_bias = get_1d_bias(df_1d)
    if daily_bias is None:
        return None, last1h, None

    # --- 2. Daily BB structure ---
    # Price should be inside daily bands (not extended/overextended)
    inside_daily_bb = last1d["bb_lower"] < last1d["close"] < last1d["bb_upper"]
    if not inside_daily_bb:
        return None, last1h, None

    # --- 3. Signal logic ---
    signal_type = None
    direction = None

    # TREND CONTINUATION: price above mid, RSI in momentum zone
    if (
        daily_bias == "BUY"
        and price > last1h["bb_mid"]
        and RSI_BULL_ZONE < rsi_val < RSI_OVERBOUGHT   # momentum but not exhausted
        and is_strong_candle(last1h, "BUY")
        and prev1h["close"] > prev1h["open"]            # 2-candle confirmation
    ):
        direction = "BUY"
        signal_type = "Trend"

    elif (
        daily_bias == "SELL"
        and price < last1h["bb_mid"]
        and RSI_OVERSOLD < rsi_val < RSI_BULL_ZONE
        and is_strong_candle(last1h, "SELL")
        and prev1h["close"] < prev1h["open"]
    ):
        direction = "SELL"
        signal_type = "Trend"

    # MEAN REVERSION: price touches/breaches BB edge, RSI extreme, daily bias agrees
    elif (
        daily_bias == "BUY"
        and price <= last1h["bb_lower"]
        and rsi_val <= RSI_OVERSOLD
        and is_strong_candle(last1h, "BUY")             # bounce candle forming
    ):
        direction = "BUY"
        signal_type = "Reversal"

    elif (
        daily_bias == "SELL"
        and price >= last1h["bb_upper"]
        and rsi_val >= RSI_OVERBOUGHT
        and is_strong_candle(last1h, "SELL")
    ):
        direction = "SELL"
        signal_type = "Reversal"

    if direction:
        return direction, last1h, signal_type

    return None, last1h, None
