import os
import json
import time
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
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
# 2. READ UNIFIED SHEET
# ==========================================
SINGLE_SHEET_URL = 'https://docs.google.com/spreadsheets/d/1A2fUfXGKXXQxzFnoR30cFVqtmb-28KTi4fR4N0e507g/edit?usp=sharing'

master_ss = gc.open_by_url(SINGLE_SHEET_URL)
target_ss = gc.open_by_url(SINGLE_SHEET_URL)

print("Reading Tickers, Volumes, and Industries from Unified Sheet...")
try:
    master_ws = master_ss.worksheet("Avg_Rupee_Volume_Master")
    master_data = master_ws.get_all_records()
except Exception as e:
    raise SystemExit(f"🛑 Error reading Master Data: {e}")

master_df = pd.DataFrame(master_data)

if 'Symbol' not in master_df.columns:
    raise SystemExit("🛑 Error: Could not find 'Symbol' column in the Master Sheet.")

tickers = (master_df['Symbol'].str.strip() + '.NS').tolist()

industry_map = master_df.set_index('Symbol')['Industry Group'].to_dict()
volume_map = master_df.set_index('Symbol')['Avg_Rupee_Volume_Crores'].to_dict()

# ==========================================
# 3. VECTORIZED DATA DOWNLOAD (SAFE BATCHED)
# ==========================================
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

print(f"Downloading 2 years of history for {len(tickers)} stocks in safe batches...")
all_chunks = []
ticker_chunks = list(chunker(tickers, 400))

for i, chunk in enumerate(ticker_chunks):
    print(f" -> Downloading price batch {i+1} of {len(ticker_chunks)}...")
    chunk_data = yf.download(chunk, period="2y", group_by="ticker", threads=True, progress=False)
    all_chunks.append(chunk_data)
    if i < len(ticker_chunks) - 1:
        time.sleep(2)

data = pd.concat(all_chunks, axis=1)

# 🟢 SQUASH TIMESTAMPS: Align all chunks to midnight so vectorization works perfectly.
if not data.empty:
    data.index = pd.to_datetime(data.index).normalize()
    data = data.groupby(data.index).max()

close_df = pd.DataFrame({t: data[t]['Close'] for t in tickers if t in data.columns.levels[0]})
vol_df = pd.DataFrame({t: data[t]['Volume'] for t in tickers if t in data.columns.levels[0]})
low_df = pd.DataFrame({t: data[t]['Low'] for t in tickers if t in data.columns.levels[0]})
high_df = pd.DataFrame({t: data[t]['High'] for t in tickers if t in data.columns.levels[0]})

close_df.index = close_df.index.tz_localize(None)
vol_df.index = vol_df.index.tz_localize(None)
low_df.index = low_df.index.tz_localize(None)
high_df.index = high_df.index.tz_localize(None)

# ==========================================
# 4. CRUNCH TECHNICALS, POCKET PIVOT & ATR DISTANCES
# ==========================================
print("Crunching technical criteria, ATR Distances, and Weekly Pocket Pivots...")

low_52w = low_df.rolling(window=252, min_periods=200).min()
avg_vol_50 = vol_df.rolling(window=50).mean()
max_vol_252 = vol_df.rolling(window=252, min_periods=200).max()
prev_max_close_252 = close_df.shift(1).rolling(window=252, min_periods=1).max()
adr_20_df = ((high_df / low_df) - 1).rolling(window=20).mean() * 100

base_rules = close_df > 40

# --- FAST VECTORIZED ATR & EMA/SMA DISTANCES ---
ema_9_df = close_df.ewm(span=9, adjust=False).mean()
ema_21_df = close_df.ewm(span=21, adjust=False).mean()
ema_50_df = close_df.ewm(span=50, adjust=False).mean()

tr1 = high_df - low_df
tr2 = (high_df - close_df.shift(1)).abs()
tr3 = (low_df - close_df.shift(1)).abs()
tr_df = pd.DataFrame(np.maximum(tr1.values, np.maximum(tr2.values, tr3.values)), index=close_df.index, columns=close_df.columns)
atr_14_df = tr_df.rolling(window=14).mean().replace(0, np.nan) # Prevent division by zero

weekly_close_df = close_df.resample('W-FRI').last()
daily_sma_4_df = weekly_close_df.rolling(4).mean().reindex(close_df.index, method='ffill')
daily_sma_10_df = weekly_close_df.rolling(10).mean().reindex(close_df.index, method='ffill')

dist_9_df = (close_df - ema_9_df) / atr_14_df
dist_21_df = (close_df - ema_21_df) / atr_14_df
dist_50_df = (close_df - ema_50_df) / atr_14_df
dist_w4_df = (close_df - daily_sma_4_df) / atr_14_df
dist_w10_df = (close_df - daily_sma_10_df) / atr_14_df

# --- WEEKLY POCKET PIVOT VECTORIZATION ---
pp_mask_df = pd.DataFrame(False, index=close_df.index, columns=close_df.columns)

for ticker in tickers:
    if ticker not in data.columns.levels[0]: continue
    df_tick = data[ticker].dropna(subset=['Close'])
    if len(df_tick) < 60: continue

    weekly_last_dates = df_tick.reset_index().groupby(pd.Grouper(key='Date', freq='W-FRI'))['Date'].last().dropna()

    weekly_df = df_tick.resample('W-FRI').agg({
        'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()

    weekly_df['ActualDate'] = weekly_last_dates.values

    ema_10 = weekly_df['Close'].ewm(span=10, adjust=False).mean()
    ema_30 = weekly_df['Close'].ewm(span=30, adjust=False).mean()
    cond1 = ema_10 > ema_30

    wk_range = weekly_df['High'] - weekly_df['Low']
    wcr = ((weekly_df['Close'] - weekly_df['Low']) / wk_range) * 100
    cond2 = wcr >= 40

    sma_vol = weekly_df['Volume'].rolling(10).mean()
    cond3 = weekly_df['Volume'] > sma_vol

    is_down_week = weekly_df['Close'].diff() < 0
    down_vols = weekly_df['Volume'].where(is_down_week, 0)
    max_down_vol_10w = down_vols.shift(1).rolling(10).max()
    cond4 = weekly_df['Volume'] > max_down_vol_10w

    cond5 = weekly_df['Close'] > weekly_df['Close'].shift(1)

    pp_weekly = cond1 & cond2 & cond3 & cond4 & cond5

    trigger_dates = weekly_df.loc[pp_weekly, 'ActualDate'].dropna()
    if not trigger_dates.empty:
        pp_mask_df.loc[trigger_dates, ticker] = True

# --- ASSIGN ALL SCANNERS ---
scanner_conditions = {
    "70% up from low": close_df >= (1.7 * low_52w),
    "1 week strength": (close_df.pct_change(periods=5, fill_method=None) * 100) > 15,
    "1 month strength": (close_df.pct_change(periods=21, fill_method=None) * 100) > 25,
    "3 month strength": (close_df.pct_change(periods=63, fill_method=None) * 100) > 35,
    "six month strength": (close_df.pct_change(periods=126, fill_method=None) * 100) > 50,
    "up on volume": ((close_df.pct_change(periods=1, fill_method=None) * 100) >= 4.5) & ((vol_df / avg_vol_50) > 2),
    "hvy": (vol_df == max_vol_252) & (vol_df > 0),
    "52 week high": close_df > prev_max_close_252,
    "Weekly Pocket Pivot": pp_mask_df
}

six_months_ago = pd.Timestamp.today().normalize() - pd.DateOffset(months=6)
time_mask = close_df.index >= six_months_ago

scanner_masks = {}
all_shortlisted = set()

for name, condition in scanner_conditions.items():
    combined_mask = (base_rules & condition).loc[time_mask]
    scanner_masks[name] = combined_mask
    passing_tickers = combined_mask.columns[combined_mask.any()].tolist()
    all_shortlisted.update(passing_tickers)

print(f"Total unique stocks passing Base Rules + ANY specific scan: {len(all_shortlisted)}")

# ==========================================
# 5. MARKET CAP CHECK & DATA FORMATTING
# ==========================================
print("Verifying Volume limits, Market Cap, and compiling final metrics...")
results = {name: [] for name in scanner_conditions.keys()}
adr_6m = adr_20_df.loc[time_mask]

mcap_cache = {}

for ticker in all_shortlisted:
    stock_name = ticker.replace('.NS', '')
    avg_vol_crores = volume_map.get(stock_name, 0)

    if avg_vol_crores < 1.0:
        continue

    absolute_rupee_vol = avg_vol_crores * 10000000

    try:
        if ticker not in mcap_cache:
            mcap_cache[ticker] = yf.Ticker(ticker).info.get('marketCap', 0)
        market_cap = mcap_cache[ticker]

        if market_cap > 0:
            for scan_name, mask in scanner_masks.items():
                if ticker in mask.columns and mask[ticker].any():
                    valid_dates = mask[ticker][mask[ticker]].index
                    ratio = absolute_rupee_vol / market_cap

                    if ratio >= 0.0025:
                        latest_date_obj = valid_dates.max()
                        latest_date_str = latest_date_obj.strftime('%Y-%m-%d')

                        trigger_adr = adr_6m.loc[latest_date_obj, ticker]
                        
                        # Fetch the 5 ATR Distances for this exact date and ticker
                        d9 = dist_9_df.loc[latest_date_obj, ticker]
                        d21 = dist_21_df.loc[latest_date_obj, ticker]
                        d50 = dist_50_df.loc[latest_date_obj, ticker]
                        dw4 = dist_w4_df.loc[latest_date_obj, ticker]
                        dw10 = dist_w10_df.loc[latest_date_obj, ticker]

                        industry = industry_map.get(stock_name, "Unclassified")

                        results[scan_name].append([
                            latest_date_str, stock_name, industry, avg_vol_crores,
                            round(trigger_adr, 2) if pd.notna(trigger_adr) else "N/A",
                            round(d9, 2) if pd.notna(d9) else "N/A",
                            round(d21, 2) if pd.notna(d21) else "N/A",
                            round(d50, 2) if pd.notna(d50) else "N/A",
                            round(dw4, 2) if pd.notna(dw4) else "N/A",
                            round(dw10, 2) if pd.notna(dw10) else "N/A"
                        ])
    except Exception as e:
        continue

# ==========================================
# 6. WRITE RESULTS TO TARGET SHEET
# ==========================================
print("\nExporting all formatted data to Unified Google Sheet...")

headers = [
    "Trigger Date", "Stock Symbol", "Industry Group", "Avg Rupee Vol (Cr)", "ADR %",
    "9 EMA (ATR)", "21 EMA (ATR)", "50 EMA (ATR)", "4W SMA (ATR)", "10W SMA (ATR)"
]
col_letter = chr(ord('A') + len(headers) - 1)

for tab_name, matched_stocks in results.items():
    try:
        worksheet = target_ss.worksheet(tab_name)
        worksheet.clear()
    except gspread.exceptions.WorksheetNotFound:
        worksheet = target_ss.add_worksheet(title=tab_name, rows="1000", cols="20")

    matched_stocks.sort(key=lambda x: x[0], reverse=True)
    sheet_output = [headers] + matched_stocks

    if len(sheet_output) > 1:
        worksheet.update(values=sheet_output, range_name='A1', value_input_option='USER_ENTERED')
        worksheet.format(f'A1:{col_letter}1', {
            "backgroundColor": {"red": 0.1, "green": 0.2, "blue": 0.4},
            "horizontalAlignment": "CENTER",
            "textFormat": {"foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}, "bold": True}
        })
        print(f" -> '{tab_name}' tab updated with {len(matched_stocks)} stocks.")
    else:
        worksheet.update(values=[["No stocks met criteria."]], range_name='A1', value_input_option='USER_ENTERED')
        print(f" -> '{tab_name}' tab updated (0 matches).")

print("\nMaster Scanner Complete! All 9 tabs updated in the Dashboard Sheet.")
