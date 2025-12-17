import os
import requests
import pandas as pd
import feedparser
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from telegram import Bot
from urllib.parse import quote

SYMBOLS = ["XAU/USD"]
API_KEYS = os.getenv("TD_API_KEYS", "").split(",")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

bot = Bot(token=TELEGRAM_TOKEN)

# Cache-safe paths
os.environ["HF_HOME"] = "/tmp/.cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache"

labels = ["Positive", "Negative", "Neutral"]

tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

def send_alert(msg):
    bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)

def fetch_data(symbol, interval, limit=100):
    base_url = "https://api.twelvedata.com/time_series"
    for key in API_KEYS:
        try:
            r = requests.get(
                f"{base_url}?symbol={symbol}&interval={interval}&outputsize={limit}&apikey={key.strip()}",
                timeout=15
            )
            if r.status_code == 200 and "values" in r.json():
                df = pd.DataFrame(r.json()["values"])
                df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
                df = df.sort_values("datetime")
                df = df.astype(float, errors="ignore")
                return df
        except:
            continue
    return None

def fetch_news(query, n=15):
    rss = f"https://news.google.com/rss/search?q={quote(query)}"
    feed = feedparser.parse(rss)
    return [e.title for e in feed.entries[:n]]

def analyze_sentiment(symbol):
    titles = fetch_news("gold market" if symbol == "XAU/USD" else symbol)
    summary = {"Positive": 0, "Negative": 0, "Neutral": 0}

    for t in titles:
        inputs = tokenizer(t, return_tensors="pt", truncation=True, max_length=256)
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
