import os
import json
import gspread
from google.oauth2.service_account import Credentials

SHEET_NAME = "TradingBotState"
SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# Load service account
creds_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
if not creds_json:
    raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON secret is not set!")

creds_dict = json.loads(creds_json)
creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPE)
gc = gspread.authorize(creds)
sheet = gc.open(SHEET_NAME).sheet1

# In-memory cache to minimize Google Sheets reads
_signal_cache = {}

def _load_cache():
    global _signal_cache
    if not _signal_cache:
        records = sheet.get_all_records()
        _signal_cache = {row["symbol"]: row["last_signal"] for row in records}

def get_last_signal(symbol):
    _load_cache()
    return _signal_cache.get(symbol)

def set_last_signal(symbol, signal):
    _load_cache()
    # Only update if changed
    if _signal_cache.get(symbol) == signal:
        return
    _signal_cache[symbol] = signal

    records = sheet.get_all_records()
    for i, row in enumerate(records, start=2):
        if row.get("symbol") == symbol:
            sheet.update_cell(i, 2, signal)
            return
    # If new symbol
    sheet.append_row([symbol, signal])
