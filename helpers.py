import os
import asyncio
import warnings
import requests
import pandas as pd
import feedparser
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from telegram import Bot
from telegram.request import HTTPXRequest
from telegram.error import TimedOut, NetworkError
from urllib.parse import quote

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

SYMBOLS = ["XAU/USD"]
API_KEYS = os.getenv("TD_API_KEYS", "").split(",")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

os.environ["HF_HOME"] = "/tmp/.cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache"

request = HTTPXRequest(connect_timeout=30, read_timeout=30,
                       write_timeout=30, pool_timeout=30)
bot = Bot(token=TELEGRAM_TOKEN, request=request)

labels = ["Positive", "Negative", "Neutral"]
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone", use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

# ── Telegram ──────────────────────────────────────────
async def _send(msg: str):
    try:
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=msg,
            parse_mode="HTML"
        )
    except (TimedOut, NetworkError) as e:
        print(f"[WARN] Telegram error: {e}")
    except Exception as e:
        print(f"[WARN] Telegram unexpected: {e}")

def send_alert(msg: str):
    try:
        asyncio.run(_send(msg))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_send(msg))
        loop.close()

# ── Market Data ───────────────────────────────────────
def fetch_data(symbol: str, interval: str, limit: int = 100):
    base_url = "https://api.twelvedata.com/time_series"
    for key in API_KEYS:
        try:
            r = requests.get(
                f"{base_url}?symbol={symbol}&interval={interval}"
                f"&outputsize={limit}&apikey={key.strip()}",
                timeout=15
            )
            if r.status_code == 200:
                data = r.json()
                if "values" in data:
                    df = pd.DataFrame(data["values"])
                    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
                    df = df.sort_values("datetime").reset_index(drop=True)
                    df = df.astype({"open": float, "high": float,
                                    "low": float, "close": float})
                    return df
        except Exception:
            continue
    return None

# ── Indicators ────────────────────────────────────────
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's smoothed RSI."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, float("inf"))
    return 100 - (100 / (1 + rs))

def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std(ddof=0)
    return sma + std_dev * std, sma, sma - std_dev * std

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high  = df["high"]
    low   = df["low"]
    close = df["close"].shift(1)
    tr    = pd.concat([
        high - low,
        (high - close).abs(),
        (low  - close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()

# ── Sentiment ─────────────────────────────────────────
_sentiment_cache: dict = {}

def analyze_sentiment(symbol: str, force: bool = False):
    """
    Runs FinBERT on latest news headlines.
    Cached per symbol per run.
    Returns (positive%, negative%, neutral%, net_bias)
    net_bias: +1 bullish | -1 bearish | 0 neutral
    """
    if symbol in _sentiment_cache and not force:
        return _sentiment_cache[symbol]

    query   = "gold price forecast" if symbol == "XAU/USD" else symbol
    rss_url = f"https://news.google.com/rss/search?q={quote(query)}"
    feed    = feedparser.parse(rss_url)
    titles  = [e.title for e in feed.entries[:15]]

    summary = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for title in titles:
        inputs = tokenizer(title, return_tensors="pt",
                           truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
        summary[labels[probs.argmax()]] += 1

    total = sum(summary.values()) or 1
    pos   = summary["Positive"] / total * 100
    neg   = summary["Negative"] / total * 100
    neu   = summary["Neutral"]  / total * 100
    bias  = 1 if pos - neg >= 20 else -1 if neg - pos >= 20 else 0

    result = (pos, neg, neu, bias)
    _sentiment_cache[symbol] = result
    return result
