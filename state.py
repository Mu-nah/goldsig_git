import os
import gspread
from google.oauth2.service_account import Credentials

SHEET_NAME = "TradingBotState"
SCOPE = ["https://www.googleapis.com/auth/spreadsheets"]

creds = Credentials.from_service_account_info(
    eval(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")),
    scopes=SCOPE
)

gc = gspread.authorize(creds)
sheet = gc.open(SHEET_NAME).sheet1

def get_last_signal(symbol):
    records = sheet.get_all_records()
    for row in records:
        if row["symbol"] == symbol:
            return row["last_signal"]
    return None

def set_last_signal(symbol, signal):
    records = sheet.get_all_records()
    for i, row in enumerate(records, start=2):
        if row["symbol"] == symbol:
            sheet.update_cell(i, 2, signal)
            return
    sheet.append_row([symbol, signal])

