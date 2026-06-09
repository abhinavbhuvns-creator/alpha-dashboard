import os
import json
import warnings
import pandas as pd
import numpy as np
import gspread
from yahooquery import Ticker
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta

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

SINGLE_SHEET_URL = 'https://docs.google.com/spreadsheets/d/1A2fUfXGKXXQxzFnoR30cFVqtmb-28KTi4fR4N0e507g/edit?usp=sharing'
master_ss = gc.open_by_url(SINGLE_SHEET_URL)
target_ss = gc.open_by_url(SINGLE_SHEET_URL)

# ==========================================
# 2. GET INDUSTRY MAPPING FROM MASTER
# ==========================================
print("Mapping stocks to industries from Unified Sheet...")
master_ws = master_ss.worksheet("Avg_Rupee_Volume_Master")
master_data = master_ws.get_all_records()
df_master = pd.DataFrame(master_data)
stock_to_industry = df_master.set_index('Symbol')['Industry Group'].to_dict()

# ==========================================
# 3. FETCH DYNAMIC SETTINGS FROM TARGET SHEET
# ==========================================
print("Reading 'Settings' tab for dynamic scanners and timeframes...")
try:
    settings_ws = target_ss.worksheet("Settings")
    settings_data = settings_ws.get_all_values()
except gspread.exceptions.WorksheetNotFound:
    raise SystemExit("🛑 CRITICAL ERROR: Could not find the 'Settings' tab in your Target Sheet.")

scanners = []
rules_universe = {}

for row in settings_data[1:]:
    if len(row) >= 2:
        scan_name = row[0].strip()
        if not scan_name: continue
        try: days = int(row[1].strip())
        except ValueError: days = 15
        scanners.append(scan_name)
        rules_universe[scan_name] = days

all_tabs = target_ss.worksheets()
tab_mapping = {}
for sheet in all_tabs:
    normalized_name = sheet.title.replace(" ", "").upper()
    tab_mapping[normalized_name] = sheet

all_dates = []
raw_scanner_data = {}

for scan in scanners:
    target_name = scan.replace(" ", "").upper()
    if target_name == "SIXMONTHSTRENGTH" and "6MONTHSTRENGTH" in tab_mapping:
        target_name = "6MONTHSTRENGTH"
    if target_name not in tab_mapping:
        continue

    ws = tab_mapping[target_name]
    data = ws.get_all_values()
    raw_scanner_data[scan] = data

    for row in data[1:]:
        if len(row) >= 2 and row[0] != "Trigger Date":
            try: all_dates.append(pd.to_datetime(row[0]).normalize()) 
            except: pass

latest_trading_date = max(all_dates) if all_dates else pd.Timestamp.today().normalize()
print(f"Latest Market Date identified as: {latest_trading_date.strftime('%Y-%m-%d')}")

universe_tracker = {}
oneday_tracker = {}

for scan, data in raw_scanner_data.items():
    if len(data) <= 1: continue

    lookback_days = rules_universe.get(scan, 0)
    cutoff_date_univ = latest_trading_date - pd.Timedelta(days=lookback_days)

    for row in data[1:]:
        if len(row) < 2: continue
        stock = row[1].strip().upper() 
        if not stock or stock == "STOCK SYMBOL": continue

        if lookback_days >= 9000:
            if stock not in universe_tracker:
                universe_tracker[stock] = {s: "No" for s in scanners}
            universe_tracker[stock][scan] = "Yes"
            if stock not in oneday_tracker:
                oneday_tracker[stock] = {s: "No" for s in scanners}
            oneday_tracker[stock][scan] = "Yes"
            continue

        try: trigger_date = pd.to_datetime(row[0].strip()).normalize() 
        except: continue

        if trigger_date >= cutoff_date_univ:
            if stock not in universe_tracker:
                universe_tracker[stock] = {s: "No" for s in scanners}
            universe_tracker[stock][scan] = "Yes"

        if trigger_date == latest_trading_date:
            if stock not in oneday_tracker:
                oneday_tracker[stock] = {s: "No" for s in scanners}
            oneday_tracker[stock][scan] = "Yes"

all_unique_stocks = list(set(universe_tracker.keys()) | set(oneday_tracker.keys()))
print(f"Found {len(all_unique_stocks)} unique stocks matching timeframe rules.")

# ==========================================
# 4. DOWNLOAD TECHNICAL DATA (YAHOOQUERY BACKEND)
# ==========================================
print("Fetching data instantly via Yahoo Backend API...")
tickers = [s + '.NS' for s in all_unique_stocks]

# Pulls all data asynchronously in one massively parallel shot
t = Ticker(tickers, asynchronous=True)
data = t.history(period="1y")

tech_data = {}
print("Calculating EMAs, Returns, Squeeze Metric, and ADR...")

if isinstance(data, pd.DataFrame) and not data.empty:
    for stock in all_unique_stocks:
        ticker = stock + '.NS'
        try:
            if ticker not in data.index.levels[0]: 
                continue
                
            df = data.loc[ticker]
            df = df.dropna(subset=['close'])
            if df.empty or len(df) < 150: continue

            close = df['close'].iloc[-1]
            ema_21 = df['close'].ewm(span=21, adjust=False).mean().iloc[-1]
            ema_50 = df['close'].ewm(span=50, adjust=False).mean().iloc[-1]
            ema_150 = df['close'].ewm(span=150, adjust=False).mean().iloc[-1]

            rupee_vol = df['close'] * df['volume']
            avg_rupee_vol_20 = rupee_vol.rolling(window=20).mean().iloc[-1]

            ret_1d = df['close'].pct_change(periods=1).iloc[-1] * 100
            ret_1w = df['close'].pct_change(periods=5).iloc[-1] * 100
            ret_1m = df['close'].pct_change(periods=21).iloc[-1] * 100

            if len(df) >= 64:
                close_1m_ago = df['close'].iloc[-22]
                close_3m_ago = df['close'].iloc[-64]
                ret_prev_2m = ((close_1m_ago / close_3m_ago) - 1) * 100
            else:
                ret_prev_2m = np.nan

            daily_range = (df['high'] / df['low']) - 1
            adr_20 = daily_range.rolling(window=20).mean().iloc[-1] * 100

            tech_data[stock] = {
                "close": close, "ema_21": ema_21, "ema_50": ema_50, "ema_150": ema_150,
                "vol": avg_rupee_vol_20, "ret_1d": ret_1d, "ret_1w": ret_1w, "ret_1m": ret_1m,
                "ret_prev_2m": ret_prev_2m, "adr": adr_20
            }
        except Exception as e: 
            continue
else:
    print("Warning: Failed to retrieve bulk data.")

# ==========================================
# 5. BUILD THE THREE LISTS
# ==========================================
headers = [
    "Stock Symbol", "Industry Group", "ADR %", "20D Avg Rupee Vol",
    "1 Day Return %", "1 Week Return %", "1 Month Return %", "Prev 2M Return (Ending 1M Ago) %"
] + scanners

def format_row(stock, scan_dict):
    td = tech_data.get(stock)
    if not td: return None
    ind = stock_to_industry.get(stock, "Uncategorized")

    row = [
        stock, ind, round(td['adr'], 2), round(td['vol'], 0),
        round(td['ret_1d'], 2), round(td['ret_1w'], 2), round(td['ret_1m'], 2),
        round(td['ret_prev_2m'], 2) if pd.notna(td['ret_prev_2m']) else ""
    ]
    for scan in scanners:
        row.append(scan_dict.get(scan, "No"))
    return row

def passes_ema_rules(stock):
    td = tech_data.get(stock)
    if not td: return False
    return (td['ema_21'] > td['ema_50']) and (td['ema_50'] > td['ema_150']) and (td['close'] > td['ema_50'])

universe_output = [headers]
for stock, scans in universe_tracker.items():
    row = format_row(stock, scans)
    if row: universe_output.append(row)

watchlist_output = [headers]
for stock, scans in universe_tracker.items():
    if passes_ema_rules(stock):
        row = format_row(stock, scans)
        if row: watchlist_output.append(row)

oneday_output = [headers]
for stock, scans in oneday_tracker.items():
    if passes_ema_rules(stock):
        row = format_row(stock, scans)
        if row: oneday_output.append(row)

# ==========================================
# 6. EXPORT TO GOOGLE SHEETS
# ==========================================
def write_to_sheet(tab_name, data):
    print(f"Writing to '{tab_name}'...")
    try:
        ws = target_ss.worksheet(tab_name)
        ws.clear()
    except gspread.exceptions.WorksheetNotFound:
        ws = target_ss.add_worksheet(title=tab_name, rows="1000", cols="20")

    if len(data) > 1:
        ws.update(values=data, range_name='A1', value_input_option='USER_ENTERED')
        ws.freeze(rows=1)
        
        col_letter = chr(ord('H') + len(scanners))
        ws.format(f'A1:{col_letter}1', {
            "backgroundColor": {"red": 0.1, "green": 0.2, "blue": 0.4},
            "horizontalAlignment": "CENTER",
            "textFormat": {"foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}, "bold": True}
        })
        print(f" -> Generated {len(data)-1} stocks.")
    else:
        ws.update(values=[["No stocks met criteria."]], range_name='A1', value_input_option='USER_ENTERED')
        print(" -> 0 stocks matched.")

print("\nFinalizing outputs...")
write_to_sheet("universe", universe_output)
write_to_sheet("watchlist", watchlist_output)
write_to_sheet("watchlist one day", oneday_output)

print("\nDashboard Creation Complete! All 3 tabs are ready in your Target Sheet.")
