import os
import json
import warnings
import pandas as pd
import numpy as np
import gspread
from yahooquery import Ticker
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
# 3. VECTORIZED DATA DOWNLOAD (YAHOOQUERY API)
# ==========================================
print(f"Downloading 2 years of history for {len(tickers)} stocks instantly via Yahoo Backend...")

t = Ticker(tickers, asynchronous=True)
data = t.history(period="2y")

if not isinstance(data, pd.DataFrame) or data.empty:
    raise SystemExit("🛑 Error: Failed to retrieve bulk history data.")

# Transform the multi-index data into the TxN format the math engine expects
data = data.reset_index()

close_df = data.pivot(index='date', columns='symbol', values='close')
vol_df = data.pivot(index='date', columns='symbol', values='volume')
low_df = data.pivot(index='date', columns='symbol', values='low')
high_df = data.pivot(index='date', columns='symbol', values='high')

# Strip timezone formatting from dates to avoid comparison errors
close_df.index = pd.to_datetime(close_df.index).tz_localize(None)
vol_df.index = pd.to_datetime(vol_df.index).tz_localize(None)
low_df.index = pd.to_datetime(low_df.index).tz_localize(None)
high_df.index = pd.to_datetime(high_df.index).tz_localize(None)

# ==========================================
# 4. CRUNCH TECHNICALS & POCKET PIVOT
# ==========================================
print("Crunching technical criteria, ADR, and Weekly Pocket Pivots...")

low_52w = low_df.rolling(window=252, min_periods=200).min()
avg_vol_50 = vol_df.rolling(window=50).mean()
max_vol_252 = vol_df.rolling(window=252, min_periods=200).max()
prev_max_close_252 = close_df.shift(1).rolling(window=252, min_periods=1).max()
adr_20_df = ((high_df / low_df) - 1).rolling(window=20).mean() * 100

base_rules = close_df > 40

# --- WEEKLY POCKET PIVOT VECTORIZATION ---
pp_mask_df = pd.DataFrame(False, index=close_df.index, columns=close_df.columns)

for ticker in close_df.columns:
    df_tick = pd.DataFrame({
        'Close': close_df[ticker],
        'High': high_df[ticker],
        'Low': low_df[ticker],
        'Volume': vol_df[ticker]
    }).dropna(subset=['Close'])
    
    if len(df_tick) < 60: continue
    
    df_tick.index.name = 'Date'
    weekly_last_dates = df_tick.reset_index().groupby(pd.Grouper(key='Date', freq='W-FRI'))['Date'].last().dropna()

    weekly_df = df_tick.resample('W-FRI').agg({
        'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()

    weekly_df['ActualDate'] = weekly_last_dates.values

    ema_10 = weekly_df['Close'].ewm(span=10, adjust=False).mean()
    ema_30 = weekly_df['Close'].ewm(span=30, adjust=False).mean()
    cond1 = ema_10 > ema_30

    wk_range = weekly_df['High'] - weekly_df['Low']
    # Prevent division by zero
    wk_range = wk_range.replace(0, np.nan) 
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
    "1 week strength": (close_df.pct_change(periods=5) * 100) > 15,
    "1 month strength": (close_df.pct_change(periods=21) * 100) > 25,
    "3 month strength": (close_df.pct_change(periods=63) * 100) > 35,
    "six month strength": (close_df.pct_change(periods=126) * 100) > 50,
    "up on volume": ((close_df.pct_change(periods=1) * 100) >= 4.5) & ((vol_df / avg_vol_50) > 2),
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
# 5. ASYNC MARKET CAP CHECK & DATA FORMATTING
# ==========================================
print("Verifying Volume limits, bulk fetching Market Caps, and compiling results...")
results = {name: [] for name in scanner_conditions.keys()}
adr_6m = adr_20_df.loc[time_mask]

mcap_cache = {}

# Bulk fetch Market Caps to completely bypass the 10-minute bottleneck
if all_shortlisted:
    print(f"Fetching Market Caps for {len(all_shortlisted)} shortlisted stocks...")
    shortlisted_list = list(all_shortlisted)
    t_mcap = Ticker(shortlisted_list, asynchronous=True)
    try:
        summary_data = t_mcap.summary_detail
        for tkr, details in summary_data.items():
            if isinstance(details, dict):
                mcap_cache[tkr] = details.get('marketCap', 0)
            else:
                mcap_cache[tkr] = 0
    except Exception as e:
        print(f"Warning: Issue fetching market caps: {e}")

for ticker in all_shortlisted:
    stock_name = ticker.replace('.NS', '')
    avg_vol_crores = volume_map.get(stock_name, 0)

    if avg_vol_crores < 1.0:
        continue

    absolute_rupee_vol = avg_vol_crores * 10000000
    market_cap = mcap_cache.get(ticker, 0)

    if market_cap > 0:
        for scan_name, mask in scanner_masks.items():
            if ticker in mask.columns and mask[ticker].any():
                valid_dates = mask[ticker][mask[ticker]].index
                ratio = absolute_rupee_vol / market_cap

                if ratio >= 0.0025:
                    latest_date_obj = valid_dates.max()
                    latest_date_str = latest_date_obj.strftime('%Y-%m-%d')

                    trigger_adr = adr_6m.loc[latest_date_obj, ticker]
                    industry = industry_map.get(stock_name, "Unclassified")

                    results[scan_name].append([
                        latest_date_str,
                        stock_name,
                        industry,
                        avg_vol_crores,
                        round(trigger_adr, 2) if pd.notna(trigger_adr) else "N/A"
                    ])

# ==========================================
# 6. WRITE RESULTS TO TARGET SHEET
# ==========================================
print("\nExporting all formatted data to Unified Google Sheet...")
headers = ["Trigger Date", "Stock Symbol", "Industry Group", "Avg Rupee Vol (Cr)", "ADR %"]

for tab_name, matched_stocks in results.items():
    try:
        worksheet = target_ss.worksheet(tab_name)
        worksheet.clear()
    except gspread.exceptions.WorksheetNotFound:
        worksheet = target_ss.add_worksheet(title=tab_name, rows="1000", cols="5")

    matched_stocks.sort(key=lambda x: x[0], reverse=True)
    sheet_output = [headers] + matched_stocks

    if len(sheet_output) > 1:
        worksheet.update(values=sheet_output, range_name='A1', value_input_option='USER_ENTERED')
        worksheet.format('A1:E1', {
            "backgroundColor": {"red": 0.1, "green": 0.2, "blue": 0.4},
            "horizontalAlignment": "CENTER",
            "textFormat": {"foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}, "bold": True}
        })
        print(f" -> '{tab_name}' tab updated with {len(matched_stocks)} stocks.")
    else:
        worksheet.update(values=[["No stocks met criteria."]], range_name='A1', value_input_option='USER_ENTERED')
        print(f" -> '{tab_name}' tab updated (0 matches).")

print("\nMaster Scanner Complete! All 9 tabs updated in the Dashboard Sheet.")
