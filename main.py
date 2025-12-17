import os
import json
from datetime import datetime, timezone, timedelta
from helpers import fetch_data, analyze_sentiment, send_alert
from strategy import generate_signal, SYMBOLS

SIGNAL_FILE = "signals.json"
WAT = timezone(timedelta(hours=1))  # UTC+1

# Load previous signals to prevent duplicates
if os.path.exists(SIGNAL_FILE):
    with open(SIGNAL_FILE, "r") as f:
        last_signal_dict = json.load(f)
else:
    last_signal_dict = {symbol: None for symbol in SYMBOLS}

def main(run_type="normal"):
    now_wat = datetime.now(WAT)

    for symbol in SYMBOLS:
        df_1h = fetch_data(symbol, "1h", 100)
        df_1d = fetch_data(symbol, "1day", 50)
        if df_1h is None or df_1d is None:
            continue

        signal, last1h, sig_type = generate_signal(df_1h, df_1d)
        pos, neg, neu = analyze_sentiment(symbol)

        if run_type == "normal":
            if signal and signal != last_signal_dict[symbol]:
                msg = (
                    f"📊 {symbol} Signal ({signal})" + (f" [{sig_type}]" if sig_type else "") + "\n"
                    f"Close: {last1h['close']:.4f}\n"
                    f"RSI: {last1h['rsi']:.2f}\n"
                    f"Sentiment → 🟢 {pos:.1f}% | 🔴 {neg:.1f}% | ⚪ {neu:.1f}%\n"
                    f"Time: {now_wat}"
                )
                send_alert(msg)
                last_signal_dict[symbol] = signal

        elif run_type == "daily":
            msg = (
                f"⏰ {symbol} 1AM WAT Status\n"
                f"Signal: {signal if signal else 'No clear signal'}"
                + (f" ({sig_type})" if sig_type else "") + "\n"
                f"Close: {last1h['close']:.4f}\n"
                f"RSI: {last1h['rsi']:.2f}\n"
                f"Sentiment → 🟢 {pos:.1f}% | 🔴 {neg:.1f}% | ⚪ {neu:.1f}%"
            )
            send_alert(msg)

    # Save updated signals
    with open(SIGNAL_FILE, "w") as f:
        json.dump(last_signal_dict, f)

if __name__ == "__main__":
    import sys
    run_type = sys.argv[1] if len(sys.argv) > 1 else "normal"
    main(run_type)
