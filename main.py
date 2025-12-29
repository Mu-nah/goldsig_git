import os
from datetime import datetime, timezone, timedelta

from helpers import fetch_data, analyze_sentiment, send_alert
from strategy import generate_signal, SYMBOLS
from state import get_last_signal, set_last_signal

# ──────────────────────────────
# Timezone
# ──────────────────────────────
WAT = timezone(timedelta(hours=1))  # UTC+1


def main():
    run_mode = os.getenv("RUN_MODE", "normal")  # normal | daily
    now_wat = datetime.now(WAT)

    for symbol in SYMBOLS:
        df_1h = fetch_data(symbol, "1h", 100)
        df_1d = fetch_data(symbol, "1day", 50)

        if df_1h is None or df_1d is None:
            continue

        signal, last1h, sig_type = generate_signal(df_1h, df_1d)

        current_signal = f"{signal}_{sig_type}" if signal and sig_type else None

        # ──────────────────────────────
        # NORMAL MODE → USE STATE
        # ──────────────────────────────
        if run_mode == "normal":
            if not current_signal:
                continue

            last_signal = get_last_signal(symbol)

            if current_signal == last_signal:
                continue

            pos, neg, neu = analyze_sentiment(symbol)

            msg = (
                f"📊 {symbol} Signal Alert\n"
                f"Signal: {signal} ({sig_type})\n"
                f"Close: {last1h['close']:.4f}\n"
                f"RSI: {last1h['rsi']:.2f}\n"
                f"Sentiment → 🟢 {pos:.1f}% | 🔴 {neg:.1f}% | ⚪ {neu:.1f}%\n"
                f"Time (WAT): {now_wat.strftime('%Y-%m-%d %H:%M')}"
            )

            send_alert(msg)
            set_last_signal(symbol, current_signal)

        # ──────────────────────────────
        # DAILY MODE → NO STATE
        # ──────────────────────────────
        elif run_mode == "daily":
            pos, neg, neu = analyze_sentiment(symbol)

            msg = (
                f"⏰ {symbol} — 1AM WAT Daily Status\n"
                f"Signal: {signal if signal else 'No clear signal'}"
                + (f" ({sig_type})" if sig_type else "") + "\n"
                f"Close: {last1h['close']:.4f}\n"
                f"RSI: {last1h['rsi']:.2f}\n"
                f"Sentiment → 🟢 {pos:.1f}% | 🔴 {neg:.1f}% | ⚪ {neu:.1f}%\n"
                f"Date: {now_wat.strftime('%Y-%m-%d')}"
            )

            send_alert(msg)


if __name__ == "__main__":
    main()
