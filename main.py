from datetime import datetime, timezone, timedelta
from helpers import fetch_data, analyze_sentiment, send_alert
from strategy import generate_signal, SYMBOLS
from state import get_last_signal, set_last_signal

# Timezone
WAT = timezone(timedelta(hours=1))  # UTC+1
now_wat = datetime.now(WAT)

# Determine run type automatically
run_type = "daily" if now_wat.hour == 1 and now_wat.minute < 20 else "normal"

def main(run_type=run_type):
    for symbol in SYMBOLS:
        df_1h = fetch_data(symbol, "1h", 100)
        df_1d = fetch_data(symbol, "1day", 50)

        if df_1h is None or df_1d is None:
            continue

        signal, last1h, sig_type = generate_signal(df_1h, df_1d)

        # Full signal identity (None if no signal)
        current_signal = f"{signal}_{sig_type}" if signal and sig_type else None

        last_signal = get_last_signal(symbol)

        if run_type == "normal":
            # Skip if no signal or signal hasn't changed
            if not current_signal or current_signal == last_signal:
                continue

            pos, neg, neu = analyze_sentiment(symbol)
            msg = (
                f"📊 {symbol} Signal ({signal})"
                + (f" [{sig_type}]" if sig_type else "") + "\n"
                f"Close: {last1h['close']:.4f}\n"
                f"RSI: {last1h['rsi']:.2f}\n"
                f"Sentiment → 🟢 {pos:.1f}% | 🔴 {neg:.1f}% | ⚪ {neu:.1f}%\n"
                f"Time: {now_wat}"
            )
            send_alert(msg)
            set_last_signal(symbol, current_signal)

        elif run_type == "daily":
            pos, neg, neu = analyze_sentiment(symbol)
            msg = (
                f"⏰ {symbol} 1AM WAT Status\n"
                f"Signal: {signal if signal else 'No clear signal'}"
                + (f" ({sig_type})" if sig_type else "") + "\n"
                f"Close: {last1h['close']:.4f}\n"
                f"RSI: {last1h['rsi']:.2f}\n"
                f"Sentiment → 🟢 {pos:.1f}% | 🔴 {neg:.1f}% | ⚪ {neu:.1f}%"
            )
            send_alert(msg)
            # Store even "no signal" to track last state
            set_last_signal(symbol, current_signal if current_signal else "no_signal")

if __name__ == "__main__":
    main(run_type)
