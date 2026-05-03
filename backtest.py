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
COOLDOWN_BARS  = 10    # bars to skip after an SL before re-entering same direction

RSI_PERIOD     = 14
BB_PERIOD      = 20
BB_STDDEV      = 2
ATR_PERIOD     = 14
SL_MULTIPLIER  = 1.5
TP_MULTIPLIER  = 2.5

RSI_OVERSOLD   = 30
RSI_OVERBOUGHT = 70
RSI_BULL_ZONE  = 48
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

def away_from_swing(df_1h: pd.DataFrame, i: int,
                    direction: str, lookback: int = 20) -> bool:
    """
    Price must be at least 2x average candle range away from swing high/low.
    Prevents entries right at support/resistance walls.
    """
    start     = max(0, i - lookback)
    recent    = df_1h.iloc[start: i + 1]
    price     = df_1h.iloc[i]["close"]
    avg_range = (recent["high"] - recent["low"]).mean()

    if direction == "SELL":
        swing_low = recent["low"].min()
        return price > swing_low + (avg_range * 2)
    else:
        swing_high = recent["high"].max()
        return price < swing_high - (avg_range * 2)

def daily_bias(df_1d: pd.DataFrame, idx: int) -> str | None:
    """
    Structural bias — daily BB midline + 5-candle majority vote.
    """
    if idx < 4:
        return None

    last1d = df_1d.iloc[idx]

    if pd.isna(last1d.get("bb_mid", float("nan"))):
        return None

    price_above_mid = last1d["close"] > last1d["bb_mid"]
    window = df_1d.iloc[max(0, idx - 4): idx + 1]
    bulls  = sum(1 for _, c in window.iterrows() if c["close"] > c["open"])

    if price_above_mid and bulls >= 3:
        return "BUY"
    if not price_above_mid and bulls <= 2:
        return "SELL"

    return None

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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
                df_1d: pd.DataFrame, d_idx: int):
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

    inside_daily_bb = last1d["bb_lower"] < last1d["close"] < last1d["bb_upper"]
    if not inside_daily_bb:
        return None, None, None, None

    direction = None
    sig_type  = None

    # Trend continuation
    if (bias == "BUY"
            and price > last1h["bb_mid"]
            and RSI_BULL_ZONE < rsi_val < RSI_OVERBOUGHT
            and strong_candle(last1h, "BUY")
            and away_from_swing(df_1h, i, "BUY")):
        direction, sig_type = "BUY", "Trend"

    elif (bias == "SELL"
            and price < last1h["bb_mid"]
            and RSI_OVERSOLD < rsi_val < RSI_BULL_ZONE
            and strong_candle(last1h, "SELL")
            and away_from_swing(df_1h, i, "SELL")):
        direction, sig_type = "SELL", "Trend"

    # Mean reversion
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
                    df_1d: pd.DataFrame) -> list[dict]:
    trades   = []
    equity   = INITIAL_EQUITY
    trade    = None
    cooldown = {"BUY": 0, "SELL": 0}   # bar index cooldown per direction

    df_1h = df_1h.copy()
    df_1d = df_1d.copy()
    df_1d["date"] = df_1d["datetime"].dt.date
    df_1h["date"] = df_1h["datetime"].dt.date
    d_dates = df_1d["date"].tolist()

    def get_d_idx(date):
        matches = [i for i, d in enumerate(d_dates) if d <= date]
        return matches[-1] if matches else 0

    for i in range(BB_PERIOD + 1, len(df_1h)):
        bar   = df_1h.iloc[i]
        d_idx = get_d_idx(bar["date"])

        # ── Check open trade ──────────────────────────
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

                # ── Cooldown after SL ─────────────────
                if trade["result"] == "SL":
                    cooldown[trade["direction"]] = i + COOLDOWN_BARS

                trades.append(trade)
                trade = None
            continue

        # ── Look for new signal ───────────────────────
        direction, sig_type, sl, tp = scan_signal(df_1h, i, df_1d, d_idx)

        # Block if still in cooldown for this direction
        if direction and i < cooldown.get(direction, 0):
            continue

        if direction:
            trade = {
                "symbol":     "XAU/USD",
                "direction":  direction,
                "type":       sig_type,
                "entry":      bar["close"],
                "sl":         sl,
                "tp":         tp,
                "entry_time": bar["datetime"],
            }

    # Mark any still-open trade
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
        df_1h = fetch_data(symbol, "1h",   500)
        df_1d = fetch_data(symbol, "1day", 120)

        if df_1h is None or df_1d is None:
            print(f"[ERROR] No data for {symbol}")
            continue

        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=LOOKBACK_DAYS)
        df_1h  = df_1h[df_1h["datetime"] >= cutoff].reset_index(drop=True)
        df_1d  = df_1d[df_1d["datetime"] >= cutoff].reset_index(drop=True)

        if len(df_1h) < BB_PERIOD + 5:
            print(f"[WARN] Not enough bars ({len(df_1h)}) for {symbol}")
            continue

        print(f"[INFO] {len(df_1h)} × 1h | {len(df_1d)} × 1d bars")

        df_1h = compute_indicators(df_1h)
        df_1d["bb_upper"], df_1d["bb_mid"], df_1d["bb_lower"] = bollinger_bands(
            df_1d["close"], BB_PERIOD, BB_STDDEV
        )

        print("[INFO] Simulating trades...")
        trades = simulate_trades(df_1h, df_1d)
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
