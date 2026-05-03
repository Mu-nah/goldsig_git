import os
from datetime import datetime, timezone, timedelta
from helpers import fetch_data, analyze_sentiment, send_alert
from strategy import generate_signal, SYMBOLS
from state import get_last_signal, set_last_signal

WAT = timezone(timedelta(hours=1))

def main():
    run_mode = os.getenv("RUN_MODE", "normal")
    now_wat  = datetime.now(WAT)

    for symbol in SYMBOLS:
        df_1h = fetch_data(symbol, "1h",    100)
        df_1d = fetch_data(symbol, "1day",  50)
        df_1w = fetch_data(symbol, "1week", 20)

        if df_1h is None or df_1d is None:
            print(f"[WARN] No data for {symbol}, skipping.")
            continue

        pos, neg, neu, bias = analyze_sentiment(symbol)

        signal, last1h, sig_type, sl, tp = generate_signal(
            df_1h, df_1d, df_1w, bias
        )
        current_signal = f"{signal}_{sig_type}" if signal and sig_type else None

        # ── NORMAL MODE ─────────────────────────────
        if run_mode == "normal":
            if not current_signal:
                continue
            if current_signal == get_last_signal(symbol):
                continue

            rr = round((tp - last1h["close"]) / (last1h["close"] - sl), 2) \
                 if signal == "BUY" else \
                 round((last1h["close"] - tp) / (sl - last1h["close"]), 2)

            msg = (
                f"📊 <b>{symbol} Signal Alert</b>\n"
                f"Signal : {signal} ({sig_type})\n"
                f"Close  : {last1h['close']:.4f}\n"
                f"RSI    : {last1h['rsi']:.2f}\n"
                f"SL     : {sl} | TP: {tp}\n"
                f"R:R    : 1:{rr}\n"
                f"News   : 🟢 {pos:.1f}% | 🔴 {neg:.1f}% | ⚪ {neu:.1f}%\n"
                f"Time   : {now_wat.strftime('%Y-%m-%d %H:%M')} WAT"
            )
            send_alert(msg)
            set_last_signal(symbol, current_signal)

        # ── DAILY MODE ──────────────────────────────
        elif run_mode == "daily":
            sl_str = f" | SL: {sl} TP: {tp}" if signal else ""
            msg = (
                f"⏰ <b>{symbol} — Daily Briefing</b>\n"
                f"Signal : {signal if signal else 'No clear signal'}"
                + (f" ({sig_type})" if sig_type else "") + sl_str + "\n"
                f"Close  : {last1h['close']:.4f}\n"
                f"RSI    : {last1h['rsi']:.2f}\n"
                f"News   : 🟢 {pos:.1f}% | 🔴 {neg:.1f}% | ⚪ {neu:.1f}%\n"
                f"Date   : {now_wat.strftime('%Y-%m-%d')} WAT"
            )
            send_alert(msg)

if __name__ == "__main__":
    main()
