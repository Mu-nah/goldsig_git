import os
import json
import gspread
from google.oauth2.service_account import Credentials

# ──────────────────────────────
# CONFIG
# ──────────────────────────────
SHEET_NAME = "TradingBotState"
SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# ──────────────────────────────
# LOAD GOOGLE SERVICE ACCOUNT
# ──────────────────────────────
creds_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
if not creds_json:
    raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON secret is not set!")

creds_dict = json.loads(creds_json)
creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPE)

# ──────────────────────────────
# AUTHORIZE SHEET
# ──────────────────────────────
gc = gspread.authorize(creds)
sheet = gc.open(SHEET_NAME).sheet1

# ──────────────────────────────
# GET / SET LAST SIGNAL
# ──────────────────────────────
def get_last_signal(symbol):
    """Retrieve the last sent signal for a symbol from Google Sheet."""
    records = sheet.get_all_records()

    for row in records:
        if row.get("symbol") == symbol:
            return row.get("last_signal")

    return None


def set_last_signal(symbol, signal):
    """Update the last sent signal for a symbol in Google Sheet."""
    records = sheet.get_all_records()

    for i, row in enumerate(records, start=2):
        if row.get("symbol") == symbol:
            sheet.update_cell(i, 2, signal)
            return

    # First time seeing this symbol
    sheet.append_row([symbol, signal])
