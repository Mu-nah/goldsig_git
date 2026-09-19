import pandas as pd
from helpers import rsi, bollinger_bands, atr, ema, adx

SYMBOLS        = ["XAU/USD"]
RSI_PERIOD     = 14
BB_PERIOD      = 20
BB_STDDEV      = 2
ATR_PERIOD     = 14
ADX_PERIOD     = 14
EMA_TREND_PERIOD = 20

RSI_OVERSOLD   = 30
RSI_OVERBOUGHT = 70
RSI_BULL_ZONE  = 48
RSI_BUY_MAX    = 58
MIN_BODY_RATIO = 0.25
SL_MULTIPLIER  = 1.5
TP_MULTIPLIER  = 2.5

# ── Confluence scoring ──────────────────────────────
# Trigger conditions above decide WHETHER a setup exists at all.
# These weights grade HOW GOOD a qualifying setup is, 0-100. Sum = 100.
W_RSI_QUALITY   = 18   # how close RSI sits to the ideal point in its zone
W_CANDLE        = 12   # candle body strength beyond the minimum
W_WEEKLY_BIAS   = 10   # weekly EMA20 agreement with daily bias
W_SWING_ROOM    = 15   # distance past the swing-wall minimum (Trend only)
W_ADX           = 10   # trend strength (favors Trend setups, penalizes Reversal setups)
W_SESSION       = 5    # London/NY active-hours bonus
W_RSI_MOMENTUM  = 10   # RSI moving further into the zone vs. drifting out of it
W_VOLATILITY    = 10   # current ATR vs. its own recent average — avoids dead chop and news spikes
W_EMA_EXTENSION = 10   # distance from EMA20 in ATR units — avoids chasing an already-extended move

MIN_SCORE = 55   # below this, a technically-qualifying setup is discarded as low quality

def _grade(score: float) -> str:
    if score >= 85: return "A+"
    if score >= 70: return "A"
    if score >= MIN_SCORE: return "B"
    return "C"

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _rsi_quality(rsi_val: float, kind: str, direction: str) -> float:
    if kind == "Trend":
        low, high = (RSI_BULL_ZONE, RSI_BUY_MAX) if direction == "BUY" \
                    else (RSI_OVERSOLD, RSI_BULL_ZONE)
        center, half = (low + high) / 2, (high - low) / 2
        return _clip01(1 - abs(rsi_val - center) / half) if half else 0.5
    # Reversal — deeper into overbought/oversold extreme = higher quality
    if direction == "BUY":
        return _clip01((RSI_OVERSOLD - rsi_val) / 15)
    return _clip01((rsi_val - RSI_OVERBOUGHT) / 15)

def _candle_quality(candle) -> float:
    ratio = _body_ratio(candle)
    return _clip01((ratio - MIN_BODY_RATIO) / (0.6 - MIN_BODY_RATIO))

def _weekly_quality(weekly_bias: str | None, daily_bias: str) -> float:
    if weekly_bias is None:
        return 0.5   # no data — neutral, neither rewarded nor punished
    return 1.0 if weekly_bias == daily_bias else 0.0

def _swing_quality(df_1h: pd.DataFrame, direction: str, lookback: int = 20) -> float:
    recent = _clean_recent(df_1h, lookback)
    if recent.empty:
        return 0.0
    price      = df_1h.iloc[-1]["close"]
    atr_val    = df_1h.iloc[-1]["atr"]
    avg_range  = max((recent["high"] - recent["low"]).mean(), atr_val)
    if avg_range == 0:
        return 0.0
    if direction == "SELL":
        multiple = (price - recent["low"].min()) / avg_range
    else:
        multiple = (recent["high"].max() - price) / avg_range
    return _clip01((multiple - 2) / 2)   # 2x = minimum required (0 quality), 4x+ = full quality

def _adx_quality(adx_val: float, kind: str) -> float:
    if pd.isna(adx_val):
        return 0.5
    if kind == "Trend":
        return _clip01((adx_val - 15) / 15)     # trending market rewarded
    return _clip01((30 - adx_val) / 15)          # ranging market rewarded for reversals

def _session_quality(dt) -> float:
    hour = dt.hour
    if 7 <= hour < 16:   # London + London/NY overlap — best gold liquidity
        return 1.0
    if 16 <= hour < 21:  # NY afternoon
        return 0.6
    return 0.2           # Asian / off-hours — thinner, choppier

def _rsi_momentum_quality(df_1h: pd.DataFrame, direction: str, lookback: int = 3) -> float:
    """RSI moving further into the zone (favorable) vs. drifting back out (unfavorable)."""
    if len(df_1h) <= lookback:
        return 0.5
    rsi_now  = df_1h.iloc[-1]["rsi"]
    rsi_prev = df_1h.iloc[-1 - lookback]["rsi"]
    if pd.isna(rsi_now) or pd.isna(rsi_prev):
        return 0.5
    slope = rsi_now - rsi_prev
    directional_slope = slope if direction == "BUY" else -slope
    return _clip01(0.5 + directional_slope / 20)

def _volatility_quality(df_1h: pd.DataFrame, lookback: int = 50) -> float:
    """Current ATR vs. its own recent average — penalizes dead chop and abnormal spikes."""
    if len(df_1h) < lookback + 1:
        return 0.5
    atr_now  = df_1h.iloc[-1]["atr"]
    atr_hist = df_1h["atr"].iloc[-(lookback + 1):-1]
    atr_avg  = atr_hist.mean()
    if pd.isna(atr_now) or pd.isna(atr_avg) or atr_avg == 0:
        return 0.5
    ratio = atr_now / atr_avg
    if ratio < 0.7:
        return _clip01(ratio / 0.7 * 0.5)
    if ratio <= 1.5:
        return 1.0
    return _clip01(1 - (ratio - 1.5))

def _ema_extension_quality(df_1h: pd.DataFrame, direction: str) -> float:
    """Distance from EMA20 in ATR units — rewards a confirmed push, penalizes chasing an extended move."""
    last    = df_1h.iloc[-1]
    ema20   = last.get("ema20", float("nan"))
    atr_val = last["atr"]
    if pd.isna(ema20) or pd.isna(atr_val) or atr_val == 0:
        return 0.5
    price    = last["close"]
    distance = (price - ema20) if direction == "BUY" else (ema20 - price)
    atr_multiple = distance / atr_val
    if atr_multiple < 0:
        return 0.3
    if atr_multiple <= 1.5:
        return _clip01(0.3 + (atr_multiple / 1.5) * 0.7)
    return _clip01(1.0 - (atr_multiple - 1.5) / 2.5 * 0.8)

def _confluence_score(df_1h, direction: str, kind: str,
                       rsi_val: float, weekly_bias: str | None,
                       daily_bias: str, adx_val: float) -> tuple[float, dict]:
    candle = df_1h.iloc[-1]
    parts = {
        "rsi_quality":    _rsi_quality(rsi_val, kind, direction) * W_RSI_QUALITY,
        "candle":         _candle_quality(candle) * W_CANDLE,
        "weekly_bias":    _weekly_quality(weekly_bias, daily_bias) * W_WEEKLY_BIAS,
        "swing_room":     (_swing_quality(df_1h, direction) if kind == "Trend" else 1.0) * W_SWING_ROOM,
        "adx":            _adx_quality(adx_val, kind) * W_ADX,
        "session":        _session_quality(candle["datetime"]) * W_SESSION,
        "rsi_momentum":   _rsi_momentum_quality(df_1h, direction) * W_RSI_MOMENTUM,
        "volatility":     _volatility_quality(df_1h) * W_VOLATILITY,
        "ema_extension":  _ema_extension_quality(df_1h, direction) * W_EMA_EXTENSION,
    }
    return round(sum(parts.values()), 1), parts

def _body_ratio(candle) -> float:
    total = candle["high"] - candle["low"]
    return abs(candle["close"] - candle["open"]) / total if total else 0

def _strong_candle(candle, direction: str) -> bool:
    if _body_ratio(candle) < MIN_BODY_RATIO:
        return False
    return (candle["close"] > candle["open"]) if direction == "BUY" \
           else (candle["close"] < candle["open"])

def _clean_recent(df_1h: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Strict filtering for swing window only.
    Filters zero lows AND flat candles (< $1 range).
    Does NOT affect indicator calculations.
    """
    recent = df_1h.iloc[-lookback:].copy()
    return recent[
        (recent["low"] > 100) &
        (recent["high"] - recent["low"] >= 1.0)
    ]

def _away_from_swing(df_1h: pd.DataFrame, direction: str,
                     lookback: int = 20) -> bool:
    """
    Price must be at least 2x average candle range away
    from swing high/low. Prevents entries at support/resistance walls.
    """
    recent = _clean_recent(df_1h, lookback)
    if recent.empty:
        return False

    price      = df_1h.iloc[-1]["close"]
    atr_val    = df_1h.iloc[-1]["atr"]
    avg_range  = (recent["high"] - recent["low"]).mean()
    avg_range  = max(avg_range, atr_val * 1.0)
    swing_high = recent["high"].max()
    swing_low  = recent["low"].min()

    if direction == "SELL":
        return price > swing_low + (avg_range * 2)
    return price < swing_high - (avg_range * 2)

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
    df_1w = df_1w[df_1w["low"] > 100]
    if len(df_1w) < 20:
        return None
    df_1w["ema20"] = ema(df_1w["close"], 20)
    last = df_1w.iloc[-1]
    if pd.isna(last["ema20"]):
        return None
    return "BUY" if last["close"] > last["ema20"] else "SELL"

def generate_signal(df_1h, df_1d, df_1w=None, sentiment_bias: int = 0):
    """
    sentiment_bias: +1 bullish | -1 bearish | 0 neutral
    Returns (direction, last1h, signal_type, sl, tp, score, grade)
    """
    # ── Indicators ──────────────────────────────────
    df_1h["rsi"] = rsi(df_1h["close"], RSI_PERIOD)
    df_1h["bb_upper"], df_1h["bb_mid"], df_1h["bb_lower"] = \
        bollinger_bands(df_1h["close"], BB_PERIOD, BB_STDDEV)
    df_1h["atr"] = atr(df_1h, ATR_PERIOD)
    df_1h["adx"] = adx(df_1h, ADX_PERIOD)
    df_1h["ema20"] = ema(df_1h["close"], EMA_TREND_PERIOD)

    df_1d["bb_upper"], df_1d["bb_mid"], df_1d["bb_lower"] = \
        bollinger_bands(df_1d["close"], BB_PERIOD, BB_STDDEV)

    last1h = df_1h.iloc[-1]
    last1d = df_1d.iloc[-1]

    rsi_val = last1h["rsi"]
    price   = last1h["close"]
    atr_val = last1h["atr"]

    if pd.isna(rsi_val) or pd.isna(atr_val) or atr_val == 0:
        return None, last1h, None, None, None, None, None

    # ── Daily bias ──────────────────────────────────
    daily_bias = _daily_bias(df_1d)
    if daily_bias is None:
        return None, last1h, None, None, None, None, None

    # ── Weekly bias — must align with daily ─────────
    weekly_bias = _weekly_bias(df_1w)
    if weekly_bias and weekly_bias != daily_bias:
        return None, last1h, None, None, None, None, None

    # ── Daily BB structure ───────────────────────────
    inside_daily_bb = last1d["bb_lower"] < last1d["close"] < last1d["bb_upper"]
    if not inside_daily_bb:
        return None, last1h, None, None, None, None, None

    # ── Signal Detection ────────────────────────────
    direction   = None
    signal_type = None

    # Trend continuation — RSI_BUY_MAX caps BUY at 58
    if (daily_bias == "BUY"
            and price > last1h["bb_mid"]
            and RSI_BULL_ZONE < rsi_val < RSI_BUY_MAX
            and _strong_candle(last1h, "BUY")
            and _away_from_swing(df_1h, "BUY")):
        direction, signal_type = "BUY", "Trend"

    elif (daily_bias == "SELL"
            and price < last1h["bb_mid"]
            and RSI_OVERSOLD < rsi_val < RSI_BULL_ZONE
            and _strong_candle(last1h, "SELL")
            and _away_from_swing(df_1h, "SELL")):
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
        return None, last1h, None, None, None, None, None

    # ── Sentiment Gate ──────────────────────────────
    if sentiment_bias == 1  and direction == "SELL":
        return None, last1h, None, None, None, None, None
    if sentiment_bias == -1 and direction == "BUY":
        return None, last1h, None, None, None, None, None

    # ── Confluence Score ─────────────────────────────
    score, _breakdown = _confluence_score(
        df_1h, direction, signal_type, rsi_val, weekly_bias,
        daily_bias, last1h["adx"]
    )
    if score < MIN_SCORE:
        return None, last1h, None, None, None, score, _grade(score)

    # ── SL / TP ─────────────────────────────────────
    if direction == "BUY":
        sl = round(price - atr_val * SL_MULTIPLIER, 2)
        tp = round(price + atr_val * TP_MULTIPLIER, 2)
    else:
        sl = round(price + atr_val * SL_MULTIPLIER, 2)
        tp = round(price - atr_val * TP_MULTIPLIER, 2)

    return direction, last1h, signal_type, sl, tp, score, _grade(score)
