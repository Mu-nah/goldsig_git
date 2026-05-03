import os
import json
import gspread
from google.oauth2.service_account import Credentials

SHEET_NAME = "TradingBotState"
SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

creds_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
if not creds_json:
    raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON secret is not set!")

creds = Credentials.from_service_account_info(
    json.loads(creds_json), scopes=SCOPE
)
gc    = gspread.authorize(creds)
sheet = gc.open(SHEET_NAME).sheet1

_cache:  dict = {}
_loaded: bool = False

def _load():
    global _cache, _loaded
    if not _loaded:
        records = sheet.get_all_records()
        _cache  = {r["symbol"]: r["last_signal"] for r in records}
        _loaded = True

def get_last_signal(symbol: str) -> str | None:
    _load()
    return _cache.get(symbol)

def set_last_signal(symbol: str, signal: str):
    _load()
    if _cache.get(symbol) == signal:
        return
    _cache[symbol] = signal
    records = sheet.get_all_records()
    for i, row in enumerate(records, start=2):
        if row.get("symbol") == symbol:
            sheet.update_cell(i, 2, signal)
            return
    sheet.append_row([symbol, signal])
