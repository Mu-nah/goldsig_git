import pandas as pd
from datetime import datetime, timezone, timedelta
from helpers import fetch_data, send_alert, rsi, bollinger_bands, atr

# ──────────────────────────────
# CONFIG
# ──────────────────────────────
SYMBOLS        = ["XAU/USD"]
INITIAL_EQUITY = 10_000
RISK_PER_TRADE = 0.01
LOOKBACK_DAYS  = 60
WAT            = timezone(timedelta(hours=1))

RSI_PERIOD     = 14
BB_PERIOD      = 20
BB_STDDEV      = 2
ATR_PERIOD     = 14
SL_MULTIPLIER  = 1.5
TP_MULTIPLIER  = 2.5

RSI_OVERSOLD   = 30
RSI_OVERBOUGHT = 70
RSI_BULL_ZONE  = 48
RSI_BUY_MAX    = 58
MIN_BODY_RATIO = 0.25

# ──────────────────────────────
# HELPERS
# ──────────────────────────────
def body_ratio(candle) -> float:
    total = candle["high"] - candle["low"]
    return abs(candle["close"] - candle["open"]) / total if total else 0

def strong_candle(candle, direction: str) -> bool:
    if body_ratio(candle) < MIN_BODY_RATIO:
        return False
    return (candle["close"] > candle["open"]) if direction == "BUY" \
           else (candle["close"] < candle["open"])

def clean_recent(df_1h: pd.DataFrame, i: int, lookback: int = 20) -> pd.DataFrame:
    """
    Single source of truth for swing window.
    Absolute floors for XAU/USD:
    - low > 100  → filters zero/corrupted candles (gold never < $100)
    - range >= 1.0 → filters flat/illiquid candles (< $1 range is noise)
    """
    start  = max(0, i - lookback)
    recent = df_1h.iloc[start: i + 1].copy()
    recent = recent[
        (recent["low"] > 100) &
        (recent["high"] - recent["low"] >= 1.0)
    ]
    return recent

def away_from_swing(df_1h: pd.DataFrame, i: int,
                    direction: str, lookback: int = 20) -> bool:
    recent = clean_recent(df_1h, i, lookback)
    if recent.empty:
        return False

    price      = df_1h.iloc[i]["close"]
    atr_val    = df_1h.iloc[i]["atr"]
    avg_range  = (recent["high"] - recent["low"]).mean()
    avg_range  = max(avg_range, atr_val * 1.0)
    swing_high = recent["high"].max()
    swing_low  = recent["low"].min()

    if direction == "SELL":
        return price > swing_low + (avg_range * 2)
    return price < swing_high - (avg_range * 2)

def in_sl_zone(price: float, direction: str,
               sl_zones: list[dict], atr_val: float) -> bool:
    for zone in sl_zones:
        if zone["direction"] != direction:
            continue
        if abs(price - zone["price"]) <= atr_val * 1.0:
            return True
    return False

def daily_bias(df_1d: pd.DataFrame, idx: int) -> str | None:
    if idx < 4:
        return None
    last1d = df_1d.iloc[idx]
    if pd.isna(last1d.get("bb_mid", float("nan"))):
        return None
    price_above_mid = last1d["close"] > last1d["bb_mid"]
    window = df_1d.iloc[max(0, idx - 4): idx + 1]
    bulls  = sum(1 for _, c in window.iterrows() if c["close"] > c["open"])
    if price_above_mid and bulls >= 3: return "BUY"
    if not price_above_mid and bulls <= 2: return "SELL"
    return None

def weekly_bias_at(df_1w: pd.DataFrame, w_idx: int) -> str | None:
    if df_1w is None or w_idx < 19:
        return None
    window = df_1w.iloc[: w_idx + 1].copy()
    window = window[window["low"] > 100]
    if len(window) < 20:
        return None
    window["ema20"] = window["close"].ewm(span=20, adjust=False).mean()
    last = window.iloc[-1]
    if pd.isna(last["ema20"]):
        return None
    return "BUY" if last["close"] > last["ema20"] else "SELL"

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Absolute floors — same logic as clean_recent
    df = df[df["low"] > 100]
    df = df[df["high"] - df["low"] >= 1.0].reset_index(drop=True)
    df["rsi"] = rsi(df["close"], RSI_PERIOD)
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = bollinger_bands(
        df["close"], BB_PERIOD, BB_STDDEV
    )
    df["atr"] = atr(df, ATR_PERIOD)
    return df

# ──────────────────────────────
# SIGNAL SCANNER
# ──────────────────────────────
def scan_signal(df_1h: pd.DataFrame, i: int,
                df_1d: pd.DataFrame, d_idx: int,
                df_1w: pd.DataFrame = None, w_idx: int = 0,
                debug: bool = False):
    if i < 1:
        return None, None, None, None

    last1h = df_1h.iloc[i]
    last1d = df_1d.iloc[d_idx]

    rsi_val = last1h["rsi"]
    price   = last1h["close"]
    atr_val = last1h["atr"]

    if pd.isna(rsi_val) or pd.isna(atr_val) or atr_val == 0:
        return None, None, None, None

    bias = daily_bias(df_1d, d_idx)
    if bias is None:
        return None, None, None, None

    w_bias = weekly_bias_at(df_1w, w_idx)
    if w_bias and w_bias != bias:
        return None, None, None, None

    inside_daily_bb = last1d["bb_lower"] < last1d["close"] < last1d["bb_upper"]
    if not inside_daily_bb:
        return None, None, None, None

    direction = None
    sig_type  = None

    if (bias == "BUY"
            and price > last1h["bb_mid"]
            and RSI_BULL_ZONE < rsi_val < RSI_BUY_MAX
            and strong_candle(last1h, "BUY")
            and away_from_swing(df_1h, i, "BUY")):
        direction, sig_type = "BUY", "Trend"

    elif (bias == "SELL"
            and price < last1h["bb_mid"]
            and RSI_OVERSOLD < rsi_val < RSI_BULL_ZONE
            and strong_candle(last1h, "SELL")
            and away_from_swing(df_1h, i, "SELL")):
        direction, sig_type = "SELL", "Trend"

    elif (bias == "BUY"
            and price <= last1h["bb_lower"]
            and rsi_val <= RSI_OVERSOLD
            and strong_candle(last1h, "BUY")):
        direction, sig_type = "BUY", "Reversal"

    elif (bias == "SELL"
            and price >= last1h["bb_upper"]
            and rsi_val >= RSI_OVERBOUGHT
            and strong_candle(last1h, "SELL")):
        direction, sig_type = "SELL", "Reversal"

    if debug:
        recent_dbg = clean_recent(df_1h, i)
        avg_range  = (recent_dbg["high"] - recent_dbg["low"]).mean() \
                     if not recent_dbg.empty else 0
        avg_range  = max(avg_range, atr_val * 1.0)
        swing_high = recent_dbg["high"].max() if not recent_dbg.empty else 0
        swing_low  = recent_dbg["low"].min()  if not recent_dbg.empty else 0
        needed     = avg_range * 2
        bar_time   = df_1h.iloc[i]["datetime"] \
                     if "datetime" in df_1h.columns else i
        label      = direction if direction else "NO-SIGNAL"
        swing_ok   = away_from_swing(df_1h, i, direction) \
                     if direction else "-"
        print(
            f"[SIGNAL] {label} {sig_type or '-'} | "
            f"Time: {bar_time} | "
            f"Price: {price:.2f} | RSI: {rsi_val:.1f} | "
            f"DailyBias: {bias} | WeeklyBias: {w_bias} | "
            f"SwingHigh: {swing_high:.2f} | SwingLow: {swing_low:.2f} | "
            f"GapToHigh: {swing_high - price:.2f} | "
            f"GapToLow: {price - swing_low:.2f} | "
            f"Needed: {needed:.2f} | SwingOK: {swing_ok}"
        )

    if not direction:
        return None, None, None, None

    sl = round(price - atr_val * SL_MULTIPLIER, 2) if direction == "BUY" \
         else round(price + atr_val * SL_MULTIPLIER, 2)
    tp = round(price + atr_val * TP_MULTIPLIER, 2) if direction == "BUY" \
         else round(price - atr_val * TP_MULTIPLIER, 2)

    return direction, sig_type, sl, tp

# ──────────────────────────────
# TRADE SIMULATOR
# ──────────────────────────────
def simulate_trades(df_1h: pd.DataFrame,
                    df_1d: pd.DataFrame,
                    df_1w: pd.DataFrame = None) -> list[dict]:
    trades   = []
    equity   = INITIAL_EQUITY
    trade    = None
    sl_zones = []

    df_1h = df_1h.copy()
    df_1d = df_1d.copy()
    df_1d["date"] = df_1d["datetime"].dt.date
    df_1h["date"] = df_1h["datetime"].dt.date
    d_dates = df_1d["date"].tolist()

    df_1w_copy = None
    w_dates    = []
    if df_1w is not None:
        df_1w_copy = df_1w.copy()
        df_1w_copy["date"] = df_1w_copy["datetime"].dt.date
        w_dates = df_1w_copy["date"].tolist()

    def get_d_idx(date):
        matches = [i for i, d in enumerate(d_dates) if d <= date]
        return matches[-1] if matches else 0

    def get_w_idx(date):
        matches = [i for i, d in enumerate(w_dates) if d <= date]
        return matches[-1] if matches else 0

    for i in range(BB_PERIOD + 1, len(df_1h)):
        bar   = df_1h.iloc[i]
        d_idx = get_d_idx(bar["date"])
        w_idx = get_w_idx(bar["date"]) if w_dates else 0

        sl_zones = [z for z in sl_zones if i - z["bar"] <= 30]

        if trade:
            hit_sl = (trade["direction"] == "BUY"  and bar["low"]  <= trade["sl"]) or \
                     (trade["direction"] == "SELL" and bar["high"] >= trade["sl"])
            hit_tp = (trade["direction"] == "BUY"  and bar["high"] >= trade["tp"]) or \
                     (trade["direction"] == "SELL" and bar["low"]  <= trade["tp"])

            if hit_tp or hit_sl:
                exit_price = trade["tp"] if hit_tp else trade["sl"]
                pnl_pips   = (exit_price - trade["entry"]) \
                             if trade["direction"] == "BUY" \
                             else (trade["entry"] - exit_price)

                risk_amt   = equity * RISK_PER_TRADE
                sl_dist    = abs(trade["entry"] - trade["sl"])
                lot_size   = risk_amt / sl_dist if sl_dist else 0
                pnl_dollar = round(pnl_pips * lot_size, 2)
                equity     = round(equity + pnl_dollar, 2)

                trade["exit"]       = exit_price
                trade["exit_time"]  = bar["datetime"]
                trade["result"]     = "TP" if hit_tp else "SL"
                trade["pnl_pips"]   = round(pnl_pips, 2)
                trade["pnl_dollar"] = pnl_dollar
                trade["equity"]     = equity

                if trade["result"] == "SL":
                    sl_zones.append({
                        "direction": trade["direction"],
                        "price":     exit_price,
                        "bar":       i
                    })
                    print(f"[SL-ZONE] {trade['direction']} zone set at "
                          f"{exit_price:.2f} bar {i}")

                trades.append(trade)
                trade = None
            continue

        direction, sig_type, sl, tp = scan_signal(
            df_1h, i, df_1d, d_idx, df_1w_copy, w_idx, debug=True
        )

        if direction:
            atr_val = df_1h.iloc[i]["atr"]
            price   = df_1h.iloc[i]["close"]

            if in_sl_zone(price, direction, sl_zones, atr_val):
                print(f"[SL-ZONE BLOCK] {direction} at {price:.2f} blocked")
                continue

            trade = {
                "symbol":     "XAU/USD",
                "direction":  direction,
                "type":       sig_type,
                "entry":      bar["close"],
                "sl":         sl,
                "tp":         tp,
                "entry_time": bar["datetime"],
            }

    if trade:
        trade.update({
            "exit": None, "exit_time": None,
            "result": "OPEN", "pnl_pips": None,
            "pnl_dollar": None, "equity": equity
        })
        trades.append(trade)

    return trades

# ──────────────────────────────
# STATS
# ──────────────────────────────
def calc_stats(trades: list[dict], initial_equity: float) -> dict:
    closed = [t for t in trades if t["result"] != "OPEN"]
    if not closed:
        return {}

    wins   = [t for t in closed if t["result"] == "TP"]
    losses = [t for t in closed if t["result"] == "SL"]

    total_pnl     = sum(t["pnl_dollar"] for t in closed)
    win_rate      = len(wins) / len(closed) * 100
    avg_win       = sum(t["pnl_dollar"] for t in wins)   / len(wins)   if wins   else 0
    avg_loss      = sum(t["pnl_dollar"] for t in losses) / len(losses) if losses else 0
    profit_factor = abs(
        sum(t["pnl_dollar"] for t in wins) /
        sum(t["pnl_dollar"] for t in losses)
    ) if losses else float("inf")

    equity_curve = [initial_equity] + [t["equity"] for t in closed]
    peak, max_dd = equity_curve[0], 0.0
    for e in equity_curve:
        if e > peak: peak = e
        dd = (peak - e) / peak * 100
        if dd > max_dd: max_dd = dd

    trend_trades    = [t for t in closed if t["type"] == "Trend"]
    reversal_trades = [t for t in closed if t["type"] == "Reversal"]
    trend_wr    = len([t for t in trend_trades    if t["result"] == "TP"]) \
                  / len(trend_trades)    * 100 if trend_trades    else 0
    reversal_wr = len([t for t in reversal_trades if t["result"] == "TP"]) \
                  / len(reversal_trades) * 100 if reversal_trades else 0

    final_equity = closed[-1]["equity"]

    return {
        "total_trades":  len(closed),
        "wins":          len(wins),
        "losses":        len(losses),
        "win_rate":      round(win_rate, 1),
        "total_pnl":     round(total_pnl, 2),
        "avg_win":       round(avg_win, 2),
        "avg_loss":      round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown":  round(max_dd, 1),
        "final_equity":  round(final_equity, 2),
        "return_pct":    round((final_equity - initial_equity) / initial_equity * 100, 2),
        "trend_wr":      round(trend_wr, 1),
        "reversal_wr":   round(reversal_wr, 1),
        "open_trades":   len([t for t in trades if t["result"] == "OPEN"]),
    }

# ──────────────────────────────
# REPORT
# ──────────────────────────────
def build_report(stats: dict, trades: list[dict], symbol: str) -> str:
    now = datetime.now(WAT).strftime("%Y-%m-%d %H:%M")

    if not stats:
        return (
            f"📉 <b>{symbol} Backtest — {LOOKBACK_DAYS}d</b>\n"
            f"No closed trades found in this period.\n"
            f"Strategy may be too selective — loosen MIN_BODY_RATIO or RSI zones.\n"
            f"Run: {now} WAT"
        )

    grade = (
        "🏆 Excellent" if stats["win_rate"] >= 60 and stats["profit_factor"] >= 2   else
        "✅ Good"      if stats["win_rate"] >= 50 and stats["profit_factor"] >= 1.5 else
        "⚠️ Marginal"  if stats["profit_factor"] >= 1                               else
        "❌ Needs Work"
    )

    closed = [t for t in trades if t["result"] != "OPEN"][-5:]
    trade_log = ""
    for t in closed:
        icon = "🟢" if t["result"] == "TP" else "🔴"
        ts   = t["entry_time"].strftime("%m-%d %H:%M") \
               if hasattr(t["entry_time"], "strftime") else str(t["entry_time"])
        trade_log += (
            f"{icon} {t['direction']} {t['type']} | "
            f"Entry {t['entry']:.2f} → {t['result']} "
            f"${t['pnl_dollar']:+.0f} [{ts}]\n"
        )

    return (
        f"📊 <b>{symbol} Backtest — Last {LOOKBACK_DAYS} Days</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"Grade          : {grade}\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"Trades         : {stats['total_trades']} "
        f"(✅{stats['wins']} ❌{stats['losses']})\n"
        f"Win Rate       : {stats['win_rate']}%\n"
        f"Profit Factor  : {stats['profit_factor']}\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"Total P&L      : ${stats['total_pnl']:+,.2f}\n"
        f"Return         : {stats['return_pct']:+.2f}%\n"
        f"Final Equity   : ${stats['final_equity']:,.2f}\n"
        f"Max Drawdown   : {stats['max_drawdown']}%\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"Avg Win        : ${stats['avg_win']:+.2f}\n"
        f"Avg Loss       : ${stats['avg_loss']:+.2f}\n"
        f"Trend WR       : {stats['trend_wr']}%\n"
        f"Reversal WR    : {stats['reversal_wr']}%\n"
        f"Open Trades    : {stats['open_trades']}\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Last 5 Trades</b>\n{trade_log}"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"Initial Equity : ${INITIAL_EQUITY:,.0f}\n"
        f"Risk/Trade     : {RISK_PER_TRADE*100:.0f}%\n"
        f"Run            : {now} WAT"
    )

# ──────────────────────────────
# MAIN
# ──────────────────────────────
def main():
    print("[INFO] Starting backtest...")

    for symbol in SYMBOLS:
        print(f"[INFO] Fetching data for {symbol}...")
        df_1h = fetch_data(symbol, "1h",    500)
        df_1d = fetch_data(symbol, "1day",  120)
        df_1w = fetch_data(symbol, "1week", 30)

        if df_1h is None or df_1d is None:
            print(f"[ERROR] No data for {symbol}")
            continue

        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=LOOKBACK_DAYS)
        df_1h  = df_1h[df_1h["datetime"] >= cutoff].reset_index(drop=True)
        df_1d  = df_1d[df_1d["datetime"] >= cutoff].reset_index(drop=True)

        if len(df_1h) < BB_PERIOD + 5:
            print(f"[WARN] Not enough bars ({len(df_1h)}) for {symbol}")
            continue

        print(f"[INFO] {len(df_1h)} × 1h | {len(df_1d)} × 1d | "
              f"{len(df_1w) if df_1w is not None else 0} × 1w bars")

        df_1h = compute_indicators(df_1h)
        df_1d["bb_upper"], df_1d["bb_mid"], df_1d["bb_lower"] = bollinger_bands(
            df_1d["close"], BB_PERIOD, BB_STDDEV
        )

        print("[INFO] Simulating trades...")
        trades = simulate_trades(df_1h, df_1d, df_1w)
        print(f"[INFO] {len(trades)} trades found "
              f"({len([t for t in trades if t['result'] != 'OPEN'])} closed, "
              f"{len([t for t in trades if t['result'] == 'OPEN'])} open)")

        stats  = calc_stats(trades, INITIAL_EQUITY)
        report = build_report(stats, trades, symbol)

        print(report)
        send_alert(report)
        print("[INFO] Report sent to Telegram ✅")

if __name__ == "__main__":
    main()
