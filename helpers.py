import os
import asyncio
import requests
import pandas as pd
import feedparser
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from telegram import Bot
from telegram.request import HTTPXRequest
from telegram.error import TimedOut, NetworkError
from urllib.parse import quote

# ──────────────────────────────
# CONFIG
# ──────────────────────────────
SYMBOLS = ["XAU/USD"]

API_KEYS = os.getenv("TD_API_KEYS", "").split(",")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

request = HTTPXRequest(
    connect_timeout=30,
    read_timeout=30,
    write_timeout=30,
    pool_timeout=30,
)

bot = Bot(token=TELEGRAM_TOKEN, request=request)

# HuggingFace cache (safe for GitHub Actions)
os.environ["HF_HOME"] = "/tmp/.cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache"

# ──────────────────────────────
# FINBERT SETUP
# ──────────────────────────────
labels = ["Positive", "Negative", "Neutral"]

tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

# ──────────────────────────────
# TELEGRAM (HARDENED, NON-BLOCKING)
# ──────────────────────────────
async def _send(msg: str):
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
    except (TimedOut, NetworkError) as e:
        print(f"[WARN] Telegram timeout/network error: {e}")
    except Exception as e:
        print(f"[WARN] Telegram unexpected error: {e}")

def send_alert(msg: str):
    try:
        asyncio.run(_send(msg))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_send(msg))
        loop.close()

# ──────────────────────────────
# MARKET DATA
# ──────────────────────────────
def fetch_data(symbol: str, interval: str, limit: int = 100):
    base_url = "https://api.twelvedata.com/time_series"

    for key in API_KEYS:
        try:
            r = requests.get(
                f"{base_url}?symbol={symbol}&interval={interval}&outputsize={limit}&apikey={key.strip()}",
                timeout=15
            )

            if r.status_code == 200:
                data = r.json()
                if "values" in data:
                    df = pd.DataFrame(data["values"])
                    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
                    df = df.sort_values("datetime")

                    df = df.astype({
                        "open": float,
                        "high": float,
                        "low": float,
                        "close": float
                    })

                    return df
        except Exception:
            continue

    return None

# ──────────────────────────────
# INDICATORS (USED BY strategy.py)
# ──────────────────────────────
def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: int = 2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()

    upper = sma + std_dev * std
    lower = sma - std_dev * std

    return upper, sma, lower

# ──────────────────────────────
# NEWS & SENTIMENT
# ──────────────────────────────
def fetch_news(query: str, n: int = 15):
    rss_url = f"https://news.google.com/rss/search?q={quote(query)}"
    feed = feedparser.parse(rss_url)
    return [entry.title for entry in feed.entries[:n]]


def analyze_sentiment(symbol: str):
    query = "gold market" if symbol == "XAU/USD" else symbol
    titles = fetch_news(query, 15)

    summary = {"Positive": 0, "Negative": 0, "Neutral": 0}

    for title in titles:
        inputs = tokenizer(
            title,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
        summary[labels[probs.argmax()]] += 1

    total = sum(summary.values()) or 1

    return (
        summary["Positive"] / total * 100,
        summary["Negative"] / total * 100,
        summary["Neutral"] / total * 100,
    )