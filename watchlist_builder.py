import os
import json
import time
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
import requests
from google.oauth2.service_account import Credentials
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', message='.*Parsing dates.*')
pd.options.mode.chained_assignment = None

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

MS_COLS = [
    "EPS_Rating", "RS_Rating", "AD_Rating", "Group_Rank", "Composite_Strength", 
    "SMR_Strength", "RS_3_Mth", "RS_6_Mth", "Sales_Percentage_Chg_Last_Qtr", 
    "Sales_Percentage_Chg_Last_Qtr_-_1_Qtr_Ago", "EPS_Percentage_Chg_Last_Qtr", 
    "EPS_Percentage_Chg_Last_Qtr_-_1_Qtr_Ago", "Sales_Percentage_Chg_Last_Yr", 
    "EPS_Percentage_Chg_Last_Yr", "EBIT_Margin_Last_Yr", "Pre-Tax_Margin_Last_Yr", 
    "After_Tax_Margin_Last_Yr", "ROIC_Last_Yr", "ROE_Last_Yr", 
    "LT_Debt__Equity_Last_Yr", "IPO_Date", "Industry_Group", 
    "Percentage_from_Pivot_Weekly", "Days_from_Pivot", "Weeks_from_Pivot", 
    "Blue_Dot_Count"
]
master_ws = master_ss.worksheet("Avg_Rupee_Volume_Master")
master_data = master_ws.get_all_records()
df_master = pd.DataFrame(master_data)
stock_to_industry = df_master.set_index('Symbol')['Industry Group'].to_dict()

try:
    ms_ws = master_ss.worksheet("marketsmith database")
    df_ms = pd.DataFrame(ms_ws.get_all_records())
    if 'Symbol' in df_ms.columns:
        df_ms['Symbol'] = df_ms['Symbol'].astype(str).str.strip()
        ms_dict = df_ms.set_index('Symbol').to_dict('index')
    else:
        ms_dict = {}
except Exception as e:
    ms_dict = {}

print("Fetching Industry Ranks...")
industry_rank_map = {}
try:
    rank_ws = target_ss.worksheet("Industry Ranks")
    rank_data = rank_ws.get_all_values()
    if len(rank_data) > 1:
        headers = [h.lower() for h in rank_data[0]]
        df_ranks = pd.DataFrame(rank_data[1:], columns=headers)
        ind_col = next((c for c in headers if "industry" in c), None)
        rank_col = next((c for c in headers if "rank" in c and "history" not in c), None)
        if ind_col and rank_col:
            industry_rank_map = df_ranks.set_index(ind_col)[rank_col].to_dict()
except Exception as e:
    pass

print("Reading 'Settings' tab for dynamic scanners and timeframes...")
settings_ws = target_ss.worksheet("Settings")
settings_data = settings_ws.get_all_values()
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
tab_mapping = {sheet.title.replace(" ", "").upper(): sheet for sheet in all_tabs}
all_dates = []
raw_scanner_data = {}
for scan in scanners:
    target_name = scan.replace(" ", "").upper()
    if target_name == "SIXMONTHSTRENGTH" and "6MONTHSTRENGTH" in tab_mapping:
        target_name = "6MONTHSTRENGTH"
    if target_name not in tab_mapping: continue
    ws = tab_mapping[target_name]
    data = ws.get_all_values()
    raw_scanner_data[scan] = data
    for row in data[1:]:
        if len(row) >= 2 and row[0] != "Trigger Date":
            try: all_dates.append(pd.to_datetime(row[0].strip(), format='mixed', dayfirst=True).normalize()) 
            except: pass
latest_trading_date = max(all_dates) if all_dates else pd.Timestamp.today().normalize()

universe_tracker = {}
for scan, data in raw_scanner_data.items():
    if len(data) <= 1: continue
    lookback_days = rules_universe.get(scan, 0)
    cutoff_date_univ = latest_trading_date - pd.Timedelta(days=lookback_days)
    for row in data[1:]:
        if len(row) < 2: continue
        stock = row[1].strip().upper() 
        if not stock or stock == "STOCK SYMBOL": continue
        try: trigger_date = pd.to_datetime(row[0].strip(), format='mixed', dayfirst=True).normalize() 
        except: continue
        if trigger_date >= cutoff_date_univ:
            if stock not in universe_tracker: universe_tracker[stock] = {s: "No" for s in scanners}
            universe_tracker[stock][scan] = "Yes"

all_unique_stocks = list(universe_tracker.keys())
print(f"Found {len(all_unique_stocks)} unique stocks matching timeframe rules.")

tickers = [s + '.NS' for s in all_unique_stocks]
session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0'})
bench_close = None
for b in ['^CRSLDX', '^NSEI']:
    try:
        b_data = yf.download(b, period="250d", session=session, progress=False)
        if not b_data.empty and 'Close' in b_data:
            bench_close = b_data['Close']
            if isinstance(bench_close, pd.DataFrame): bench_close = bench_close.iloc[:, 0]
            bench_close = bench_close.ffill()
            bench_close.index = pd.to_datetime(bench_close.index).tz_localize(None).normalize()
            break
    except: continue

chunk_size = 20
ticker_chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
all_chunks = []
if len(tickers) > 0:
    for chunk in ticker_chunks:
        for _ in range(5):
            try:
                chunk_data = yf.download(chunk, period="250d", group_by="ticker", threads=False, progress=False, session=session)
                if not chunk_data.empty or len(chunk) == 0:
                    all_chunks.append(chunk_data)
                    break
            except: time.sleep(15)
        time.sleep(3)
    data = pd.concat(all_chunks, axis=1) if all_chunks else pd.DataFrame()
    if not data.empty:
        data.index = pd.to_datetime(data.index).tz_localize(None).normalize()
        data = data.groupby(data.index).max()
else:
    data = pd.DataFrame()

tech_data = {}
for stock in all_unique_stocks:
    ticker = stock + '.NS'
    if ticker == '^CNXSC': continue
    try:
        df = data[ticker].copy().dropna(subset=['Close']) if len(tickers)>1 else data.copy().dropna(subset=['Close'])
        if df.empty or len(df) < 5: continue
        
        df['tr1'] = df['High'] - df['Low']
        df['tr2'] = abs(df['High'] - df['Close'].shift(1))
        df['tr3'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        atr_14 = df['TR'].rolling(14, min_periods=1).mean().iloc[-1]
        close = df['Close'].iloc[-1]
        
        lookback_bars = min(252, len(df))
        high_52w = df['High'].tail(lookback_bars).max()
        safe_low_series = df['Low'].replace(0, np.nan)
        low_52w = safe_low_series.tail(lookback_bars).min()
        
        pct_off_high = ((close - high_52w) / high_52w) * 100 if pd.notna(high_52w) and high_52w != 0 else np.nan
        pct_above_low = ((close - low_52w) / low_52w) * 100 if pd.notna(low_52w) and low_52w != 0 else np.nan
        
        ema_9 = df['Close'].ewm(span=9, adjust=False, min_periods=1).mean().iloc[-1]
        ema_21 = df['Close'].ewm(span=21, adjust=False, min_periods=1).mean().iloc[-1]
        ema_50 = df['Close'].ewm(span=50, adjust=False, min_periods=1).mean().iloc[-1]
        
        weekly_df = df.resample('W-FRI').agg({'Open': 'first', 'Close': 'last'}).dropna()
        wk_sma_4 = weekly_df['Close'].rolling(4).mean().iloc[-1] if len(weekly_df) >= 10 else np.nan
        wk_sma_10 = weekly_df['Close'].rolling(10).mean().iloc[-1] if len(weekly_df) >= 10 else np.nan
        
        dist_9 = (close - ema_9) / atr_14 if pd.notna(atr_14) and atr_14 != 0 else np.nan
        dist_21 = (close - ema_21) / atr_14 if pd.notna(atr_14) and atr_14 != 0 else np.nan
        dist_50 = (close - ema_50) / atr_14 if pd.notna(atr_14) and atr_14 != 0 else np.nan
        dist_w4 = (close - wk_sma_4) / atr_14 if pd.notna(atr_14) and atr_14 != 0 else np.nan
        dist_w10 = (close - wk_sma_10) / atr_14 if pd.notna(atr_14) and atr_14 != 0 else np.nan
        
        c_vals = df['Close'].values
        l_vals = df['Low'].values
        active_gaps = []
        for j in range(1, len(c_vals)):
            if l_vals[j] > c_vals[j-1]: active_gaps.append([l_vals[j], c_vals[j-1]])
            for k in range(len(active_gaps)-1, -1, -1):
                if c_vals[j] < active_gaps[k][0]:
                    if c_vals[j] <= active_gaps[k][1]: active_gaps.pop(k)
                    else: active_gaps[k][0] = c_vals[j]
            if len(active_gaps) > 20: active_gaps.pop(0)
        nearest_sg_top = active_gaps[-1][0] if active_gaps else np.nan
        dist_sg = (close - nearest_sg_top) / atr_14 if pd.notna(atr_14) and atr_14 != 0 and pd.notna(nearest_sg_top) else np.nan
        
        rupee_vol = df['Close'] * df['Volume']
        avg_rupee_vol_20 = (rupee_vol.rolling(window=20, min_periods=1).mean().iloc[-1]) / 10000000
        
        ret_1d = df['Close'].pct_change(periods=1).iloc[-1] * 100
        ret_1w = df['Close'].pct_change(periods=5).iloc[-1] * 100 
        ret_curr_wk_body = ((weekly_df['Close'].iloc[-1] / weekly_df['Open'].iloc[-1]) - 1) * 100 if len(weekly_df) >= 2 else np.nan
        ret_prev_wk_body = ((weekly_df['Close'].iloc[-2] / weekly_df['Open'].iloc[-2]) - 1) * 100 if len(weekly_df) >= 2 else np.nan
        ret_1m = df['Close'].pct_change(periods=21).iloc[-1] * 100
        ret_prev_2m = ((df['Close'].iloc[-22] / df['Close'].iloc[-64]) - 1) * 100 if len(df) >= 64 else np.nan
        
        adr_20 = ((df['High'] / df['Low'].replace(0, np.nan)) - 1).rolling(window=20, min_periods=5).mean().iloc[-1] * 100
        
        if bench_close is not None:
            aligned_bench = bench_close.reindex(df.index).ffill()
            rs_raw = (df['Close'] / aligned_bench) * 100
            rs_off_high = ((rs_raw.iloc[-1] - rs_raw.tail(lookback_bars).max()) / rs_raw.tail(lookback_bars).max()) * 100
            rs_dist_21ema = ((rs_raw.iloc[-1] - rs_raw.ewm(span=21).mean().iloc[-1]) / rs_raw.ewm(span=21).mean().iloc[-1]) * 100
            rs_1w_chg = rs_raw.pct_change(periods=5).iloc[-1] * 100
            is_out = (df['Close'].pct_change() > aligned_bench.pct_change()).astype(int)
            rs_out_10d = (is_out.tail(10).sum() / 10.0) * 100 if len(is_out) >= 10 else np.nan
            rs_out_20d = (is_out.tail(20).sum() / 20.0) * 100 if len(is_out) >= 20 else np.nan
            rs_out_50d = (is_out.tail(50).sum() / 50.0) * 100 if len(is_out) >= 50 else np.nan
        else:
            rs_off_high = rs_dist_21ema = rs_1w_chg = rs_out_10d = rs_out_20d = rs_out_50d = np.nan
            
        tech_data[stock] = {
            "close": close, "vol": avg_rupee_vol_20, "ret_1d": ret_1d, "ret_1w": ret_1w, 
            "ret_curr_wk_body": ret_curr_wk_body, "ret_prev_wk_body": ret_prev_wk_body, 
            "ret_1m": ret_1m, "ret_prev_2m": ret_prev_2m, "adr": adr_20,
            "dist_9": dist_9, "dist_21": dist_21, "dist_50": dist_50,
            "dist_w4": dist_w4, "dist_w10": dist_w10, "dist_sg": dist_sg,
            "above_low": pct_above_low, "off_high": pct_off_high,
            "rs_off_high": rs_off_high, "rs_dist_21ema": rs_dist_21ema, "rs_1w_chg": rs_1w_chg,
            "rs_out_10d": rs_out_10d, "rs_out_20d": rs_out_20d, "rs_out_50d": rs_out_50d
        }
    except: continue
