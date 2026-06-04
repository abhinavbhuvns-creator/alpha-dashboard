import os
import json
import time
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials

warnings.simplefilter(action='ignore')
pd.options.mode.chained_assignment = None

# ==========================================
# 1. AUTHENTICATE USING GITHUB SECRETS
# ==========================================
print("Authenticating with Google Service Account...")

# Fetch the secret key from GitHub's hidden environment
creds_json = os.environ.get('GOOGLE_CREDENTIALS')
if not creds_json:
    raise SystemExit("🛑 Error: GOOGLE_CREDENTIALS not found in GitHub Secrets.")

creds_dict = json.loads(creds_json)
scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
gc = gspread.authorize(creds)

# ==========================================
# 2. READ MASTER SHEET
# ==========================================
SHEET_URL = 'https://docs.google.com/spreadsheets/d/1A2fUfXGKXXQxzFnoR30cFVqtmb-28KTi4fR4N0e507g/edit?usp=sharing'
spreadsheet = gc.open_by_url(SHEET_URL)

MASTER_TAB_NAME = "Avg_Rupee_Volume_Master"
TARGET_TAB_NAME = "Daily_Technicals"

print(f"Reading existing data from '{MASTER_TAB_NAME}'...")
try:
    master_ws = spreadsheet.worksheet(MASTER_TAB_NAME)
    df_master = pd.DataFrame(master_ws.get_all_records())
except Exception as e:
    raise SystemExit(f"🛑 Error: Could not read Master tab. {e}")

tickers = (df_master['Symbol'].str.strip() + '.NS').tolist()

# ==========================================
# 3. BATCH DOWNLOAD (1 YEAR OF DATA)
# ==========================================
print(f"Downloading 1 year of price history for {len(tickers)} stocks...")

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

ticker_chunks = list(chunker(tickers, 250))
all_chunks = []

for i, chunk in enumerate(ticker_chunks):
    print(f" -> Fetching batch {i+1} of {len(ticker_chunks)}...")
    chunk_data = yf.download(chunk, period="1y", group_by="ticker", threads=True, progress=False)
    all_chunks.append(chunk_data)
    if i < len(ticker_chunks) - 1:
        time.sleep(2) 

data = pd.concat(all_chunks, axis=1)

# ==========================================
# 4. CALCULATE NEW COLUMNS & SQUEEZE METRIC
# ==========================================
print("Calculating Returns, EMAs, Highs, and Custom Momentum...")
tech_metrics = {}

for ticker in tickers:
    stock_sym = ticker.replace('.NS', '')
    try:
        if len(tickers) == 1: df = data.copy()
        else:
            if ticker not in data.columns.levels[0]: continue
            df = data[ticker].copy()

        df = df.dropna(subset=['Close'])
        if len(df) < 2: continue

        close = df['Close'].iloc[-1]
        high_52w = df['High'].max()
        ema_21 = df['Close'].ewm(span=21, adjust=False).mean().iloc[-1]

        ret_1d = df['Close'].pct_change(periods=1).iloc[-1] * 100
        ret_1w = df['Close'].pct_change(periods=5).iloc[-1] * 100
        ret_1m = df['Close'].pct_change(periods=21).iloc[-1] * 100 if len(df) >= 22 else None
        ret_3m = df['Close'].pct_change(periods=63).iloc[-1] * 100 if len(df) >= 64 else None
        ret_6m = df['Close'].pct_change(periods=126).iloc[-1] * 100 if len(df) >= 127 else None

        dist_ema21 = ((close - ema_21) / ema_21) * 100
        dist_high = ((close - high_52w) / high_52w) * 100

        if len(df) >= 64:
            close_1m_ago = df['Close'].iloc[-22]
            close_3m_ago = df['Close'].iloc[-64]
            ret_past_2m_till_last_month = ((close_1m_ago / close_3m_ago) - 1) * 100
        else:
            ret_past_2m_till_last_month = None

        tech_metrics[stock_sym] = {
            "1 Day Return %": round(ret_1d, 2) if pd.notna(ret_1d) else "",
            "1 Week Return %": round(ret_1w, 2) if pd.notna(ret_1w) else "",
            "1 Month Return %": round(ret_1m, 2) if pd.notna(ret_1m) else "",
            "Prev 2M Return (Ending 1M Ago) %": round(ret_past_2m_till_last_month, 2) if pd.notna(ret_past_2m_till_last_month) else "",
            "3 Month Return %": round(ret_3m, 2) if pd.notna(ret_3m) else "",
            "6 Month Return %": round(ret_6m, 2) if pd.notna(ret_6m) else "",
            "% Dist from 21 EMA": round(dist_ema21, 2) if pd.notna(dist_ema21) else "",
            "% Dist from 52W High": round(dist_high, 2) if pd.notna(dist_high) else ""
        }
    except Exception:
        continue

# ==========================================
# 5. MERGE DATA & WRITE TO GOOGLE SHEET
# ==========================================
print("Merging data and formatting output...")
df_tech = pd.DataFrame.from_dict(tech_metrics, orient='index').reset_index()
df_tech.rename(columns={'index': 'Symbol'}, inplace=True)
df_final = pd.merge(df_master, df_tech, on='Symbol', how='left').fillna("")

print(f"Writing data to new tab: '{TARGET_TAB_NAME}'...")
try:
    target_ws = spreadsheet.worksheet(TARGET_TAB_NAME)
    target_ws.clear()
except gspread.exceptions.WorksheetNotFound:
    target_ws = spreadsheet.add_worksheet(title=TARGET_TAB_NAME, rows="2500", cols="15")

sheet_output = [df_final.columns.values.tolist()] + df_final.values.tolist()
target_ws.update(values=sheet_output, range_name='A1', value_input_option='USER_ENTERED')
target_ws.freeze(rows=1)
target_ws.format('A1:Z1', {'textFormat': {'bold': True}, "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9}})

print(f"✅ Success! Generated '{TARGET_TAB_NAME}' with {len(df_final)} enriched stocks.")
