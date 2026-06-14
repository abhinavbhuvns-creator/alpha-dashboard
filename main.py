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

creds_json = os.environ.get('GOOGLE_CREDENTIALS')
if not creds_json:
    raise SystemExit("🛑 Error: GOOGLE_CREDENTIALS not found in GitHub Secrets.")

creds_dict = json.loads(creds_json)
scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
gc = gspread.authorize(creds)

# ==========================================
# 2. READ UNIFIED MASTER SHEET
# ==========================================
SINGLE_SHEET_URL = 'https://docs.google.com/spreadsheets/d/1A2fUfXGKXXQxzFnoR30cFVqtmb-28KTi4fR4N0e507g/edit?usp=sharing'
spreadsheet = gc.open_by_url(SINGLE_SHEET_URL)

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
# 3. BATCH DOWNLOAD (2 YEARS OF DATA)
# ==========================================
print(f"Downloading 2 years of price history for {len(tickers)} stocks...")

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

ticker_chunks = list(chunker(tickers, 250))
all_chunks = []

for i, chunk in enumerate(ticker_chunks):
    print(f" -> Fetching batch {i+1} of {len(ticker_chunks)}...")
    chunk_data = yf.download(chunk, period="2y", group_by="ticker", threads=True, progress=False)
    all_chunks.append(chunk_data)
    if i < len(ticker_chunks) - 1:
        time.sleep(2) 

data = pd.concat(all_chunks, axis=1)

# ==========================================
# 4. CALCULATE NEW COLUMNS, ATR & ADR
# ==========================================
print("Calculating Returns, EMAs, ATR Distances, Pocket Pivots, and Custom Momentum...")
tech_metrics = {}

for ticker in tickers:
    stock_sym = ticker.replace('.NS', '')
    try:
        if len(tickers) == 1: df = data.copy()
        else:
            if ticker not in data.columns.levels[0]: continue
            df = data[ticker].copy()

        df = df.dropna(subset=['Close'])
        if len(df) < 60: continue 

        # --- Base Metrics ---
        close = df['Close'].iloc[-1]
        high_52w = df['High'].tail(252).max() 
        
        # --- ATR & ADR Math ---
        df['tr1'] = df['High'] - df['Low']
        df['tr2'] = abs(df['High'] - df['Close'].shift(1))
        df['tr3'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        atr_14 = df['TR'].rolling(14).mean().iloc[-1]
        
        daily_range = (df['High'] / df['Low']) - 1
        adr_20 = daily_range.rolling(window=20).mean().iloc[-1] * 100

        # --- RVOL (20 Days) ---
        avg_vol_20 = df['Volume'].rolling(window=20).mean().iloc[-1]
        rvol_20 = (df['Volume'].iloc[-1] / avg_vol_20) if avg_vol_20 > 0 else np.nan

        # --- Daily Pocket Pivot Logic (10D Count) ---
        is_down_day = df['Close'] < df['Close'].shift(1)
        daily_down_vols = df['Volume'].where(is_down_day, 0)
        max_down_vol_10d = daily_down_vols.shift(1).rolling(10).max()
        is_up_day = df['Close'] > df['Close'].shift(1)
        daily_pp = is_up_day & (df['Volume'] > max_down_vol_10d)
        ppv_10d_count = daily_pp.tail(10).sum()

        # --- Moving Averages ---
        ema_4 = df['Close'].ewm(span=4, adjust=False).mean().iloc[-1]
        ema_6 = df['Close'].ewm(span=6, adjust=False).mean().iloc[-1]
        ema_9 = df['Close'].ewm(span=9, adjust=False).mean().iloc[-1]
        ema_21 = df['Close'].ewm(span=21, adjust=False).mean().iloc[-1]
        ema_50 = df['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
        
        # Resample for Weekly Data (Close and Volume)
        weekly_df = df.resample('W-FRI').agg({'Close': 'last', 'Volume': 'sum'}).dropna()
        if len(weekly_df) >= 10:
            wk_sma_4 = weekly_df['Close'].rolling(4).mean().iloc[-1]
            wk_sma_10 = weekly_df['Close'].rolling(10).mean().iloc[-1]
        else:
            wk_sma_4 = np.nan
            wk_sma_10 = np.nan

        # --- Weekly Pocket Pivot Logic (4W Count) ---
        if len(weekly_df) >= 10:
            is_down_week = weekly_df['Close'] < weekly_df['Close'].shift(1)
            weekly_down_vols = weekly_df['Volume'].where(is_down_week, 0)
            max_down_vol_10w = weekly_down_vols.shift(1).rolling(10).max()
            is_up_week = weekly_df['Close'] > weekly_df['Close'].shift(1)
            weekly_pp = is_up_week & (weekly_df['Volume'] > max_down_vol_10w)
            ppv_4w_count = weekly_pp.tail(4).sum()
        else:
            ppv_4w_count = 0

        # --- ATR Distances ---
        dist_4 = (close - ema_4) / atr_14 if pd.notna(atr_14) and atr_14 != 0 else np.nan
        dist_6 = (close - ema_6) / atr_14 if pd.notna(atr_14) and atr_14 != 0 else np.nan
        dist_9 = (close - ema_9) / atr_14 if pd.notna(atr_14) and atr_14 != 0 else np.nan
        dist_21 = (close - ema_21) / atr_14 if pd.notna(atr_14) and atr_14 != 0 else np.nan
        dist_50 = (close - ema_50) / atr_14 if pd.notna(atr_14) and atr_14 != 0 else np.nan
        dist_w4 = (close - wk_sma_4) / atr_14 if pd.notna(atr_14) and atr_14 != 0 else np.nan
        dist_w10 = (close - wk_sma_10) / atr_14 if pd.notna(atr_14) and atr_14 != 0 else np.nan

        # --- Returns ---
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

        # Build Output Dictionary
        tech_metrics[stock_sym] = {
            "ADR %": round(adr_20, 2) if pd.notna(adr_20) else "",
            "RVOL (20D)": round(rvol_20, 2) if pd.notna(rvol_20) else "",
            "PPV (10D)": int(ppv_10d_count) if pd.notna(ppv_10d_count) else 0,
            "PPV (4W)": int(ppv_4w_count) if pd.notna(ppv_4w_count) else 0,
            
            "1 Day Return %": round(ret_1d, 2) if pd.notna(ret_1d) else "",
            "1 Week Return %": round(ret_1w, 2) if pd.notna(ret_1w) else "",
            "1 Month Return %": round(ret_1m, 2) if pd.notna(ret_1m) else "",
            "Prev 2M Return (Ending 1M Ago) %": round(ret_past_2m_till_last_month, 2) if pd.notna(ret_past_2m_till_last_month) else "",
            "3 Month Return %": round(ret_3m, 2) if pd.notna(ret_3m) else "",
            "6 Month Return %": round(ret_6m, 2) if pd.notna(ret_6m) else "",
            "% Dist from 21 EMA": round(dist_ema21, 2) if pd.notna(dist_ema21) else "",
            "% Dist from 52W High": round(dist_high, 2) if pd.notna(dist_high) else "",
            
            "4 EMA": round(ema_4, 2) if pd.notna(ema_4) else "",
            "6 EMA": round(ema_6, 2) if pd.notna(ema_6) else "",
            "9 EMA": round(ema_9, 2) if pd.notna(ema_9) else "",
            "21 EMA": round(ema_21, 2) if pd.notna(ema_21) else "",
            "50 EMA": round(ema_50, 2) if pd.notna(ema_50) else "",

            "4 EMA (ATR)": round(dist_4, 2) if pd.notna(dist_4) else "",
            "6 EMA (ATR)": round(dist_6, 2) if pd.notna(dist_6) else "",
            "9 EMA (ATR)": round(dist_9, 2) if pd.notna(dist_9) else "",
            "21 EMA (ATR)": round(dist_21, 2) if pd.notna(dist_21) else "",
            "50 EMA (ATR)": round(dist_50, 2) if pd.notna(dist_50) else "",
            "4W SMA (ATR)": round(dist_w4, 2) if pd.notna(dist_w4) else "",
            "10W SMA (ATR)": round(dist_w10, 2) if pd.notna(dist_w10) else ""
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
    target_ws = spreadsheet.add_worksheet(title=TARGET_TAB_NAME, rows="2500", cols="40")

sheet_output = [df_final.columns.values.tolist()] + df_final.values.tolist()
target_ws.update(values=sheet_output, range_name='A1', value_input_option='USER_ENTERED')
target_ws.freeze(rows=1)
target_ws.format('A1:Z1', {'textFormat': {'bold': True}, "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9}})

print(f"✅ Success! Generated '{TARGET_TAB_NAME}' with {len(df_final)} enriched stocks.")
