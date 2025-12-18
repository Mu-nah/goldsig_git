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

# Expected columns:
# | key | last_signal |

# ──────────────────────────────
# GET / SET LAST SIGNAL
# ──────────────────────────────
def get_last_signal(key):
    records = sheet.get_all_records()

    for row in records:
        if row.get("key") == key:
            return row.get("last_signal")

    return None


def set_last_signal(key, signal):
    records = sheet.get_all_records()

    for i, row in enumerate(records, start=2):
        if row.get("key") == key:
            sheet.update_cell(i, 2, signal)
            return

    # First time seeing this key
    sheet.append_row([key, signal])