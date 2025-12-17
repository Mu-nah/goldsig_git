from helpers import rsi, bollinger_bands

SYMBOLS = ["XAU/USD"]

RSI_PERIOD = 14
BB_PERIOD = 20
BB_STDDEV = 2
MIN_PIP_DISTANCE = 1.0

def generate_signal(df_1h, df_1d):
    df_1h["rsi"] = rsi(df_1h["close"], RSI_PERIOD)
    df_1h["bb_upper"], df_1h["bb_mid"], df_1h["bb_lower"] = bollinger_bands(df_1h["close"], BB_PERIOD, BB_STDDEV)
    df_1d["bb_upper"], _, df_1d["bb_lower"] = bollinger_bands(df_1d["close"], BB_PERIOD, BB_STDDEV)

    last1h, last1d = df_1h.iloc[-1], df_1d.iloc[-1]
    direction = "BUY" if last1h["close"] > last1h["open"] else "SELL"

    trend = (direction == "BUY" and last1h["close"] > last1h["bb_mid"] + MIN_PIP_DISTANCE) or \
            (direction == "SELL" and last1h["close"] < last1h["bb_mid"] - MIN_PIP_DISTANCE)
    reversal = (direction == "BUY" and last1h["close"] < last1h["bb_mid"]) or \
               (direction == "SELL" and last1h["close"] > last1h["bb_mid"])
    confirm1d = (direction == "BUY" and last1d["close"] > last1d["open"]) or \
                (direction == "SELL" and last1d["close"] < last1d["open"])
    inside_bb1d = last1d["close"] < last1d["bb_upper"] and last1d["close"] > last1d["bb_lower"]

    if (trend or reversal) and confirm1d and inside_bb1d:
        sig_type = "Trend" if trend else "Reversal"
        return direction, last1h, sig_type

    return None, last1h, None
