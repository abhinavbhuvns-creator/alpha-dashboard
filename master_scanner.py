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
print("Authenticating with Google Service Account...", flush=True)
creds_json = os.environ.get('GOOGLE_CREDENTIALS')
if not creds_json:
    raise SystemExit("🛑 Error: GOOGLE_CREDENTIALS not found in GitHub Secrets.")
creds_dict = json.loads(creds_json)
scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
gc = gspread.authorize(creds)

# ==========================================
# 2. READ UNIFIED SHEET & MARKETSMITH
# ==========================================
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

print("Reading Tickers, Volumes, and Industries from Unified Sheet...", flush=True)
master_ws = master_ss.worksheet("Avg_Rupee_Volume_Master")
master_data = master_ws.get_all_records()
master_df = pd.DataFrame(master_data)
if 'Symbol' not in master_df.columns:
    raise SystemExit("🛑 Error: Could not find 'Symbol' column in the Master Sheet.")

master_df['Symbol'] = master_df['Symbol'].astype(str).str.strip().str.upper()

print("Reading Marketsmith Database...", flush=True)
try:
    ms_ws = master_ss.worksheet("marketsmith database")
    df_ms = pd.DataFrame(ms_ws.get_all_records())
    if 'Symbol' in df_ms.columns:
        df_ms['Symbol'] = df_ms['Symbol'].astype(str).str.strip().str.upper()
        ms_dict = df_ms.set_index('Symbol').to_dict('index')
    else:
        ms_dict = {}
except Exception:
    ms_dict = {}

tickers = (master_df['Symbol'] + '.NS').tolist()
industry_map = master_df.set_index('Symbol')['Industry Group'].to_dict()

volume_map = {}
for _, r in master_df.iterrows():
    sym = r['Symbol']
    try:
        volume_map[sym] = float(str(r.get('Avg_Rupee_Volume_Crores', 0)).replace(',', '').strip())
    except Exception:
        volume_map[sym] = 0.0

# ==========================================
# 3. BULLETPROOF BENCHMARK FETCHER
# ==========================================
print("\nFetching Benchmark Index separately to guarantee data integrity...", flush=True)
bench_close = None
for b in ['^NSEI', '^CRSLDX']:
    try:
        b_data = yf.download(b, period="2y", progress=False)
        if not b_data.empty and 'Close' in b_data:
            bench_close = b_data['Close']
            if isinstance(bench_close, pd.DataFrame):
                bench_close = bench_close.iloc[:, 0]
            bench_close = bench_close.ffill()
            bench_close.index = pd.to_datetime(bench_close.index).tz_localize(None).normalize()
            print(f" -> Successfully fetched benchmark: {b}", flush=True)
            break
    except Exception:
        continue

# ==========================================
# 4. VECTORIZED DATA DOWNLOAD (SAFE BATCHED)
# ==========================================
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

print(f"Downloading 2 years of history for {len(tickers)} stocks in safe batches...", flush=True)
all_chunks = []
ticker_chunks = list(chunker(tickers, 400))
for i, chunk in enumerate(ticker_chunks):
    print(f" -> Downloading price batch {i+1} of {len(ticker_chunks)}...", flush=True)
    chunk_data = yf.download(chunk, period="2y", group_by="ticker", threads=True, progress=False)
    all_chunks.append(chunk_data)
    if i < len(ticker_chunks) - 1:
        time.sleep(2)

data = pd.concat(all_chunks, axis=1)
if not data.empty:
    data.index = pd.to_datetime(data.index).tz_localize(None).normalize()
    data = data.groupby(data.index).max()

valid_cols = [t for t in tickers if t in data.columns.levels[0]]
close_df = pd.DataFrame({t: data[t]['Close'] for t in valid_cols}).ffill()
vol_df = pd.DataFrame({t: data[t]['Volume'] for t in valid_cols}).fillna(0)
low_df = pd.DataFrame({t: data[t]['Low'] for t in valid_cols}).ffill()
high_df = pd.DataFrame({t: data[t]['High'] for t in valid_cols}).ffill()

# Count actual trading bars per stock since listing (for 20-day IPO cooling rule)
bars_since_listing_df = data.xs('Close', axis=1, level=1).notna().cumsum()

# Align Benchmark and Calc Matrix for entire market (IPO-aware with min_periods=20)
if bench_close is not None:
    aligned_bench = bench_close.reindex(close_df.index).ffill()

    rs_raw_df = close_df.div(aligned_bench, axis=0) * 100.0
    rs_52w_max_df = rs_raw_df.rolling(window=252, min_periods=20).max()
    rs_off_high_df = ((rs_raw_df - rs_52w_max_df) / rs_52w_max_df) * 100.0

    rs_21ema_df = rs_raw_df.ewm(span=21, adjust=False, min_periods=1).mean()
    rs_dist_21ema_df = ((rs_raw_df - rs_21ema_df) / rs_21ema_df) * 100.0

    rs_1w_chg_df = rs_raw_df.pct_change(periods=5, fill_method=None) * 100.0

    stock_ret_df = close_df.pct_change(fill_method=None)
    bench_ret_df = aligned_bench.pct_change(fill_method=None)

    outperf_df = stock_ret_df.gt(bench_ret_df, axis=0).astype(int)
    rs_out_10d_df = (outperf_df.rolling(10, min_periods=1).sum() / 10.0) * 100.0
    rs_out_20d_df = (outperf_df.rolling(20, min_periods=1).sum() / 20.0) * 100.0
    rs_out_50d_df = (outperf_df.rolling(50, min_periods=1).sum() / 50.0) * 100.0

# ==========================================
# 5. CRUNCH TECHNICALS, POCKET PIVOT, & RS HIGH AFTER RSBP
# ==========================================
print("Crunching technical criteria, ATR Distances, Weekly Pocket Pivots, and RS High after RSBP...", flush=True)
safe_low = low_df.replace(0, np.nan)
low_52w = safe_low.rolling(window=252, min_periods=20).min()
max_high_252 = high_df.rolling(window=252, min_periods=20).max()
max_close_252 = close_df.rolling(window=252, min_periods=20).max()
avg_vol_50 = vol_df.rolling(window=50, min_periods=10).mean()
max_vol_252 = vol_df.rolling(window=252, min_periods=20).max()
prev_max_close_252 = close_df.shift(1).rolling(window=252, min_periods=20).max()

raw_daily_range = (high_df / safe_low) - 1.0
adr_20_df = raw_daily_range.replace([np.inf, -np.inf], np.nan).rolling(window=20, min_periods=5).mean() * 100.0
base_rules = close_df > 40

ema_9_df = close_df.ewm(span=9, adjust=False, min_periods=1).mean()
ema_21_df = close_df.ewm(span=21, adjust=False, min_periods=1).mean()
ema_50_df = close_df.ewm(span=50, adjust=False, min_periods=1).mean()

tr1 = high_df - low_df
tr2 = (high_df - close_df.shift(1)).abs()
tr3 = (low_df - close_df.shift(1)).abs()
tr_df = pd.DataFrame(np.maximum(tr1.values, np.maximum(tr2.values, tr3.values)), index=close_df.index, columns=close_df.columns)
atr_14_df = tr_df.rolling(window=14, min_periods=1).mean().replace(0, np.nan)

weekly_close_df = close_df.resample('W-FRI').last()
weekly_high_df = high_df.resample('W-FRI').max()
weekly_low_df = low_df.resample('W-FRI').min()
w_tr1 = weekly_high_df - weekly_low_df
w_tr2 = (weekly_high_df - weekly_close_df.shift(1)).abs()
w_tr3 = (weekly_low_df - weekly_close_df.shift(1)).abs()
w_tr_df = pd.DataFrame(np.maximum(w_tr1.values, np.maximum(w_tr2.values, w_tr3.values)), index=weekly_close_df.index, columns=weekly_close_df.columns)
w_atr_14_df = w_tr_df.rolling(window=14, min_periods=1).mean().replace(0, np.nan).reindex(close_df.index, method='ffill')

daily_sma_4_df = weekly_close_df.rolling(4, min_periods=1).mean().reindex(close_df.index, method='ffill')
daily_sma_10_df = weekly_close_df.rolling(10, min_periods=1).mean().reindex(close_df.index, method='ffill')

dist_9_df = (close_df - ema_9_df) / atr_14_df
dist_21_df = (close_df - ema_21_df) / atr_14_df
dist_50_df = (close_df - ema_50_df) / atr_14_df
dist_w4_df = (close_df - daily_sma_4_df) / atr_14_df
dist_w10_df = (close_df - daily_sma_10_df) / atr_14_df

pp_mask_df = pd.DataFrame(False, index=close_df.index, columns=close_df.columns)
rs_high_after_rsbp_mask_df = pd.DataFrame(False, index=close_df.index, columns=close_df.columns)

n_total_bars = len(close_df)

for ticker in close_df.columns:
    if ticker not in data.columns.levels[0]:
        continue

    # 1. RS High after RSBP (Blue Dot within 1 to 15 trading days after a Purple Dot, > 5 Cr Volume)
    stock_sym_clean = ticker.replace('.NS', '')
    if bench_close is not None and volume_map.get(stock_sym_clean, 0.0) >= 5.0:
        tick_rs = rs_raw_df[ticker]
        tick_high_rs = rs_52w_max_df[ticker]
        tick_high_p = max_high_252[ticker]
        tick_close_p = max_close_252[ticker]
        cooled_mask = (bars_since_listing_df[ticker] >= 20) if ticker in bars_since_listing_df.columns else True

        is_rs_nh = cooled_mask & tick_rs.notna() & tick_high_rs.notna() & (tick_rs >= tick_high_rs)
        is_p_nh = (high_df[ticker] >= tick_high_p) | (close_df[ticker] >= tick_close_p)

        rsbp_series = is_rs_nh & (~is_p_nh)       # Purple Dot (RS New High Before Price)
        rs_blue_series = is_rs_nh & is_p_nh       # Blue Dot (RS New High Confirmed With Price)

        purple_indices = np.where(rsbp_series.values)[0]
        blue_indices = np.where(rs_blue_series.values)[0]

        if len(purple_indices) > 0 and len(blue_indices) > 0:
            valid_after_rsbp = np.zeros(n_total_bars, dtype=bool)
            for p_idx in purple_indices:
                sub_blues = blue_indices[(blue_indices > p_idx) & (blue_indices <= p_idx + 15)]
                if len(sub_blues) > 0:
                    valid_after_rsbp[sub_blues] = True
            rs_high_after_rsbp_mask_df[ticker] = valid_after_rsbp

    # 2. Weekly Pocket Pivot
    df_tick = data[ticker].dropna(subset=['Close'])
    if len(df_tick) < 10:
        continue

    weekly_last_dates = df_tick.reset_index().groupby(pd.Grouper(key='Date', freq='W-FRI'))['Date'].last().dropna()
    weekly_df = df_tick.resample('W-FRI').agg({
        'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    if len(weekly_df) != len(weekly_last_dates):
        continue
    weekly_df['ActualDate'] = weekly_last_dates.values

    ema_10 = weekly_df['Close'].ewm(span=10, adjust=False).mean()
    ema_30 = weekly_df['Close'].ewm(span=30, adjust=False).mean()
    cond1 = ema_10 > ema_30

    wk_range = (weekly_df['High'] - weekly_df['Low']).replace(0, np.nan)
    wcr = ((weekly_df['Close'] - weekly_df['Low']) / wk_range) * 100.0
    cond2 = wcr >= 40

    sma_vol = weekly_df['Volume'].rolling(10, min_periods=2).mean()
    cond3 = weekly_df['Volume'] > sma_vol

    is_down_week = weekly_df['Close'].diff() < 0
    down_vols = weekly_df['Volume'].where(is_down_week, 0)
    max_down_vol_10w = down_vols.shift(1).rolling(10, min_periods=2).max()
    cond4 = weekly_df['Volume'] > max_down_vol_10w
    cond5 = weekly_df['Close'] > weekly_df['Close'].shift(1)

    pp_weekly = cond1 & cond2 & cond3 & cond4 & cond5
    trigger_dates = weekly_df.loc[pp_weekly, 'ActualDate'].dropna()
    if not trigger_dates.empty:
        valid_trig_dates = [d for d in trigger_dates if d in pp_mask_df.index]
        if valid_trig_dates:
            pp_mask_df.loc[valid_trig_dates, ticker] = True

scanner_conditions = {
    "70% up from low": close_df >= (1.7 * low_52w),
    "1 week strength": (close_df.pct_change(periods=5, fill_method=None) * 100.0) > 15,
    "1 month strength": (close_df.pct_change(periods=21, fill_method=None) * 100.0) > 25,
    "3 month strength": (close_df.pct_change(periods=63, fill_method=None) * 100.0) > 35,
    "six month strength": (close_df.pct_change(periods=126, fill_method=None) * 100.0) > 50,
    "up on volume": ((close_df.pct_change(periods=1, fill_method=None) * 100.0) >= 4.5) & ((vol_df / avg_vol_50) > 2),
    "hvy": (vol_df == max_vol_252) & (vol_df > 0),
    "52 week high": close_df > prev_max_close_252,
    "Weekly Pocket Pivot": pp_mask_df,
    "RS High after RSBP": rs_high_after_rsbp_mask_df
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

print(f"Total unique stocks passing Base Rules + ANY specific scan: {len(all_shortlisted)}", flush=True)

# ==========================================
# 6. MARKET CAP CHECK (FAST_INFO + FALLBACK) & DATA FORMATTING
# ==========================================
print("Verifying Volume limits, Market Cap, and compiling final metrics...", flush=True)
results = {name: [] for name in scanner_conditions.keys()}
adr_6m = adr_20_df.loc[time_mask]
mcap_cache = {}

for ticker in all_shortlisted:
    stock_name = ticker.replace('.NS', '')
    avg_vol_crores = volume_map.get(stock_name, 0.0)

    if avg_vol_crores < 1.0:
        continue

    absolute_rupee_vol = avg_vol_crores * 10000000.0

    try:
        if ticker not in mcap_cache:
            try:
                fi = yf.Ticker(ticker).fast_info
                mcap_cache[ticker] = getattr(fi, 'market_cap', 0) or 0
            except Exception:
                mcap_cache[ticker] = 0

        market_cap = mcap_cache[ticker]
        # Pass if Market Cap check succeeds (>= 0.25% turnover) OR if Yahoo blocks lookup (fallback to >= 5 Cr volume)
        passes_mcap = (market_cap > 0 and (absolute_rupee_vol / market_cap) >= 0.0025) or (market_cap == 0 and avg_vol_crores >= 5.0)

        if passes_mcap:
            for scan_name, mask in scanner_masks.items():
                # Strict > 5 Crore filter for RS High after RSBP
                if scan_name == "RS High after RSBP" and avg_vol_crores < 5.0:
                    continue

                if ticker in mask.columns and mask[ticker].any():
                    valid_dates = mask[ticker][mask[ticker]].index
                    latest_date_obj = valid_dates.max()
                    latest_date_str = latest_date_obj.strftime('%Y-%m-%d')

                    trigger_adr = adr_6m.loc[latest_date_obj, ticker]

                    d9 = dist_9_df.loc[latest_date_obj, ticker]
                    d21 = dist_21_df.loc[latest_date_obj, ticker]
                    d50 = dist_50_df.loc[latest_date_obj, ticker]
                    dw4 = dist_w4_df.loc[latest_date_obj, ticker]
                    dw10 = dist_w10_df.loc[latest_date_obj, ticker]

                    # Daily Support Gap (Daily 14-ATR)
                    c_series = close_df.loc[:latest_date_obj, ticker].dropna().values
                    l_series = low_df.loc[:latest_date_obj, ticker].dropna().values

                    active_gaps = []
                    for j in range(1, len(c_series)):
                        prev_c = c_series[j - 1]
                        curr_l = l_series[j]
                        curr_c = c_series[j]
                        if curr_l > prev_c:
                            active_gaps.append([curr_l, prev_c])
                        for k in range(len(active_gaps) - 1, -1, -1):
                            if curr_c < active_gaps[k][0]:
                                if curr_c <= active_gaps[k][1]:
                                    active_gaps.pop(k)
                                else:
                                    active_gaps[k][0] = curr_c
                        if len(active_gaps) > 20:
                            active_gaps.pop(0)

                    nearest_sg_top = np.nan
                    final_c = c_series[-1]
                    if active_gaps:
                        min_diff = float('inf')
                        for gap in active_gaps:
                            diff = final_c - gap[0]
                            if 0 <= diff < min_diff:
                                min_diff = diff
                                nearest_sg_top = gap[0]

                    atr_val = atr_14_df.loc[latest_date_obj, ticker]
                    dsg = (final_c - nearest_sg_top) / atr_val if pd.notna(atr_val) and atr_val != 0 and pd.notna(nearest_sg_top) else np.nan

                    # Weekly Support Gap (Weekly 14-ATR)
                    wc_series = close_df.loc[:latest_date_obj, ticker].resample('W-FRI').last().dropna().values
                    wl_series = low_df.loc[:latest_date_obj, ticker].resample('W-FRI').min().dropna().values
                    w_len = min(len(wc_series), len(wl_series))
                    w_active_gaps = []
                    for j in range(1, w_len):
                        if wl_series[j] > wc_series[j - 1]:
                            w_active_gaps.append([wl_series[j], wc_series[j - 1]])
                        for k in range(len(w_active_gaps) - 1, -1, -1):
                            if wc_series[j] < w_active_gaps[k][0]:
                                if wc_series[j] <= w_active_gaps[k][1]:
                                    w_active_gaps.pop(k)
                                else:
                                    w_active_gaps[k][0] = wc_series[j]
                        if len(w_active_gaps) > 20:
                            w_active_gaps.pop(0)

                    nearest_wsg_top = np.nan
                    if w_active_gaps:
                        min_diff = float('inf')
                        for gap in w_active_gaps:
                            diff = final_c - gap[0]
                            if 0 <= diff < min_diff:
                                min_diff = diff
                                nearest_wsg_top = gap[0]
                    w_atr_val = w_atr_14_df.loc[latest_date_obj, ticker]
                    dwsg = (final_c - nearest_wsg_top) / w_atr_val if pd.notna(w_atr_val) and w_atr_val != 0 and pd.notna(nearest_wsg_top) else np.nan

                    # Proper single-ticker IPO-aware ATH/ATL slice up to latest_date_obj
                    lookback_bars_dyn = min(252, len(c_series))
                    latest_high = high_df.loc[:latest_date_obj, ticker].tail(lookback_bars_dyn).max()
                    latest_low = safe_low.loc[:latest_date_obj, ticker].tail(lookback_bars_dyn).min()

                    pct_above_low = ((final_c - latest_low) / latest_low) * 100.0 if pd.notna(latest_low) and latest_low != 0 else np.nan
                    pct_off_high = ((final_c - latest_high) / latest_high) * 100.0 if pd.notna(latest_high) and latest_high != 0 else np.nan

                    if bench_close is not None:
                        d_rs_off_high = rs_off_high_df.loc[latest_date_obj, ticker]
                        d_rs_dist_21 = rs_dist_21ema_df.loc[latest_date_obj, ticker]
                        d_rs_1w = rs_1w_chg_df.loc[latest_date_obj, ticker]
                        d_rs_10d = rs_out_10d_df.loc[latest_date_obj, ticker]
                        d_rs_20d = rs_out_20d_df.loc[latest_date_obj, ticker]
                        d_rs_50d = rs_out_50d_df.loc[latest_date_obj, ticker]
                    else:
                        d_rs_off_high = d_rs_dist_21 = d_rs_1w = d_rs_10d = d_rs_20d = d_rs_50d = np.nan

                    industry = industry_map.get(stock_name, "Unclassified")

                    ms_data = ms_dict.get(stock_name, {})
                    ms_row = []
                    for col in MS_COLS:
                        val = ms_data.get(col, "N/A")
                        if val == "":
                            val = "N/A"
                        ms_row.append(val)

                    results[scan_name].append([
                        latest_date_str, stock_name, industry, round(avg_vol_crores, 2),
                        round(trigger_adr, 2) if pd.notna(trigger_adr) else "N/A",
                        round(d9, 2) if pd.notna(d9) else "N/A",
                        round(d21, 2) if pd.notna(d21) else "N/A",
                        round(d50, 2) if pd.notna(d50) else "N/A",
                        round(dw4, 2) if pd.notna(dw4) else "N/A",
                        round(dw10, 2) if pd.notna(dw10) else "N/A",
                        round(dsg, 2) if pd.notna(dsg) else "N/A",
                        round(dwsg, 2) if pd.notna(dwsg) else "N/A",
                        round(pct_above_low, 2) if pd.notna(pct_above_low) else "N/A",
                        round(pct_off_high, 2) if pd.notna(pct_off_high) else "N/A",
                        round(d_rs_off_high, 2) if pd.notna(d_rs_off_high) else "N/A",
                        round(d_rs_dist_21, 2) if pd.notna(d_rs_dist_21) else "N/A",
                        round(d_rs_1w, 2) if pd.notna(d_rs_1w) else "N/A",
                        round(d_rs_10d, 2) if pd.notna(d_rs_10d) else "N/A",
                        round(d_rs_20d, 2) if pd.notna(d_rs_20d) else "N/A",
                        round(d_rs_50d, 2) if pd.notna(d_rs_50d) else "N/A"
                    ] + ms_row)
    except Exception:
        continue

# ==========================================
# 7. WRITE RESULTS TO TARGET SHEET
# ==========================================
print("\nExporting all formatted data to Unified Google Sheet...", flush=True)
headers = [
    "Trigger Date", "Stock Symbol", "Industry Group", "Avg Rupee Vol (Cr)", "ADR %",
    "9 EMA (ATR)", "21 EMA (ATR)", "50 EMA (ATR)", "4W SMA (ATR)", "10W SMA (ATR)",
    "Supp Gap (ATR)", "Wk Supp Gap (ATR)", "% above 52W Low", "% off 52W High",
    "RS Line % off 52W High", "% Dist from RS 21 EMA", "RS Line 1W Change %",
    "RS Outperform (10D) %", "RS Outperform (20D) %", "RS Outperform (50D) %"
] + MS_COLS

def get_column_letter(col_idx):
    col_str = ""
    while col_idx > 0:
        col_idx, remainder = divmod(col_idx - 1, 26)
        col_str = chr(65 + remainder) + col_str
    return col_str

dynamic_col_letter = get_column_letter(len(headers))
for tab_name, matched_stocks in results.items():
    try:
        worksheet = target_ss.worksheet(tab_name)
        worksheet.clear()
    except gspread.exceptions.WorksheetNotFound:
        worksheet = target_ss.add_worksheet(title=tab_name, rows="1000", cols="50")

    matched_stocks.sort(key=lambda x: x[0], reverse=True)
    sheet_output = [headers] + matched_stocks

    if len(sheet_output) > 1:
        safe_data = []
        for row in sheet_output:
            clean_row = []
            for val in row:
                if pd.isna(val):
                    clean_row.append("")
                elif isinstance(val, (int, float, np.number)) and np.isinf(val):
                    clean_row.append("")
                else:
                    clean_row.append(val)
            safe_data.append(clean_row)
        worksheet.update(values=safe_data, range_name='A1', value_input_option='USER_ENTERED')
        worksheet.freeze(rows=1)
        worksheet.format(f'A1:{dynamic_col_letter}1', {
            "backgroundColor": {"red": 0.1, "green": 0.2, "blue": 0.4},
            "horizontalAlignment": "CENTER",
            "textFormat": {"foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}, "bold": True}
        })
        print(f" -> '{tab_name}' tab updated with {len(matched_stocks)} stocks.", flush=True)
    else:
        worksheet.update(values=[["No stocks met criteria."]], range_name='A1', value_input_option='USER_ENTERED')
        print(f" -> '{tab_name}' tab updated (0 matches).", flush=True)

print("\n✅ Master Scanner Complete! All tabs updated in the Dashboard Sheet.", flush=True)
