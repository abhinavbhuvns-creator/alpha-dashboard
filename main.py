import os
import json
import time
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
import shutil
from google.oauth2.service_account import Credentials
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore')
pd.options.mode.chained_assignment = None

os.makedirs('charts', exist_ok=True)

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
# 2. READ UNIFIED MASTER SHEET & MARKETSMITH
# ==========================================
SINGLE_SHEET_URL = 'https://docs.google.com/spreadsheets/d/1A2fUfXGKXXQxzFnoR30cFVqtmb-28KTi4fR4N0e507g/edit?usp=sharing'
spreadsheet = gc.open_by_url(SINGLE_SHEET_URL)

MASTER_TAB_NAME = "Avg_Rupee_Volume_Master"
TARGET_TAB_NAME = "Daily_Technicals"

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

print(f"Reading existing data from '{MASTER_TAB_NAME}'...")
try:
    master_ws = spreadsheet.worksheet(MASTER_TAB_NAME)
    df_master = pd.DataFrame(master_ws.get_all_records())
except Exception as e:
    raise SystemExit(f"🛑 Error: Could not read Master tab. {e}")

print("Reading Marketsmith Database...")
try:
    ms_ws = spreadsheet.worksheet("marketsmith database")
    df_ms = pd.DataFrame(ms_ws.get_all_records())
    if 'Symbol' in df_ms.columns:
        df_ms['Symbol'] = df_ms['Symbol'].astype(str).str.strip()
        valid_ms_cols = ['Symbol'] + [c for c in MS_COLS if c in df_ms.columns]
        df_ms = df_ms[valid_ms_cols]
    else:
        df_ms = pd.DataFrame(columns=['Symbol'])
except Exception as e:
    print(f"Warning: Could not read Marketsmith tab: {e}")
    df_ms = pd.DataFrame(columns=['Symbol'])

tickers = (df_master['Symbol'].str.strip() + '.NS').tolist()

# 🟢 INJECT BENCHMARKS FOR RS RATING DOWNLOAD
if '^NSEI' not in tickers: tickers.append('^NSEI')
if '^CNXSC' not in tickers: tickers.append('^CNXSC')

# ==========================================
# 3. BATCH DOWNLOAD (2 YEARS OF DATA)
# ==========================================
print(f"Downloading price history for {len(tickers)} stocks...")

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
# 3.5 VECTORIZED RELATIVE STRENGTH RANKING
# ==========================================
print("\nCalculating Cross-Sectional RS Ratings (Classic & Webster)...")
close_data = pd.DataFrame()
for t in tickers:
    if t in data.columns.levels[0] and 'Close' in data[t]:
        close_data[t] = data[t]['Close']

# Separate the benchmark
nifty = close_data['^NSEI'] if '^NSEI' in close_data.columns else None
close_data.drop(columns=['^NSEI', '^CNXSC'], inplace=True, errors='ignore')

if nifty is not None and not close_data.empty:
    daily_returns = close_data.pct_change()
    webster_base = daily_returns.ewm(span=63, adjust=False).mean()
    
    roc_63 = close_data / close_data.shift(63)
    roc_126 = close_data / close_data.shift(126)
    roc_189 = close_data / close_data.shift(189)
    roc_252 = close_data / close_data.shift(252)
    classic_base = (roc_63 * 2) + roc_126 + roc_189 + roc_252

    fridays_idx = close_data.index[close_data.index.dayofweek == 4]
    
    def calc_1_to_99_rank(row):
        return (row.rank(pct=True) * 98) + 1

    print(" -> Ranking all stocks historically for every Friday...")
    webster_friday_ranks = webster_base.loc[fridays_idx].apply(calc_1_to_99_rank, axis=1)
    classic_friday_ranks = classic_base.loc[fridays_idx].apply(calc_1_to_99_rank, axis=1)
else:
    print(" -> Skipping RS Calculation: NIFTY 50 benchmark data missing.")
    webster_friday_ranks, classic_friday_ranks = pd.DataFrame(), pd.DataFrame()

# Extract Benchmark for custom RS Lines
if '^CNXSC' in data.columns.levels[0]:
    bench_close = data['^CNXSC']['Close'].ffill()
elif '^NSEI' in data.columns.levels[0]:
    bench_close = data['^NSEI']['Close'].ffill()
else:
    bench_close = None

# ==========================================
# 4. TECH METRICS & CHART GENERATION
# ==========================================
print("\nGenerating Sheet Analytics & Custom Visual Chart PNGs...")
tech_metrics = {}

plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': '#131722', 'axes.facecolor': '#131722',
    'axes.edgecolor': '#2a2e39', 'grid.color': '#2a2e39',
    'xtick.color': '#8a93a6', 'ytick.color': '#8a93a6', 'font.size': 8
})

for ticker in tickers:
    if ticker in ['^NSEI', '^CNXSC']: continue # Skip generating metrics/charts for indexes
    
    stock_sym = ticker.replace('.NS', '')
    try:
        if len(tickers) == 1: df = data.copy()
        else:
            if ticker not in data.columns.levels[0]: continue
            df = data[ticker].copy()

        df = df.dropna(subset=['Close'])
        if len(df) < 60: continue 

        close = df['Close'].iloc[-1]
        high_52w = df['High'].tail(252).max() 
        
        safe_low_series = df['Low'].replace(0, np.nan)
        low_52w = safe_low_series.tail(252).min() 
        
        df['tr1'] = df['High'] - df['Low']
        df['tr2'] = abs(df['High'] - df['Close'].shift(1))
        df['tr3'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        atr_14 = df['TR'].rolling(14).mean().iloc[-1]
        
        df['Safe_Low'] = df['Low'].replace(0, np.nan)
        daily_range = (df['High'] / df['Safe_Low']) - 1
        adr_20 = daily_range.replace([np.inf, -np.inf], np.nan).rolling(window=20, min_periods=5).mean().iloc[-1] * 100
        
        avg_vol_20 = df['Volume'].rolling(window=20).mean().iloc[-1]
        rvol_20 = (df['Volume'].iloc[-1] / avg_vol_20) if avg_vol_20 > 0 else np.nan

        is_down_day = df['Close'] < df['Close'].shift(1)
        daily_down_vols = df['Volume'].where(is_down_day, 0)
    
        max_down_vol_10d = daily_down_vols.shift(1).rolling(10).max()
        is_up_day = df['Close'] > df['Close'].shift(1)
        daily_pp = is_up_day & (df['Volume'] > max_down_vol_10d)
        ppv_10d_count = daily_pp.tail(10).sum()

        df['ema4'] = df['Close'].ewm(span=4, adjust=False).mean()
        df['ema6'] = df['Close'].ewm(span=6, adjust=False).mean()
        df['ema9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['ema21'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['ema50'] = df['Close'].ewm(span=50, adjust=False).mean()
   
        # WEEKLY CANDLES
        weekly_df = df.resample('W-FRI').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
        
        if len(weekly_df) >= 10:
            wk_sma_4 = weekly_df['Close'].rolling(4).mean().iloc[-1]
            wk_sma_10 = weekly_df['Close'].rolling(10).mean().iloc[-1]
            is_down_week = weekly_df['Close'] < weekly_df['Close'].shift(1)
            weekly_down_vols = weekly_df['Volume'].where(is_down_week, 0)
            max_down_vol_10w = weekly_down_vols.shift(1).rolling(10).max()
            is_up_week = weekly_df['Close'] > weekly_df['Close'].shift(1)
            weekly_pp = is_up_week & (weekly_df['Volume'] > max_down_vol_10w)
            ppv_4w_count = weekly_pp.tail(4).sum()
        else:
            wk_sma_4 = np.nan; wk_sma_10 = np.nan; ppv_4w_count = 0

        dist_4 = (close - df['ema4'].iloc[-1]) / atr_14 if pd.notna(atr_14) and atr_14 != 0 else np.nan
        dist_6 = (close - df['ema6'].iloc[-1]) / atr_14 if pd.notna(atr_14) and atr_14 != 0 else np.nan
        dist_9 = (close - df['ema9'].iloc[-1]) / atr_14 if pd.notna(atr_14) and atr_14 != 0 else np.nan
        dist_21 = (close - df['ema21'].iloc[-1]) / atr_14 if pd.notna(atr_14) and atr_14 != 0 else np.nan
        dist_50 = (close - df['ema50'].iloc[-1]) / atr_14 if pd.notna(atr_14) and atr_14 != 0 else np.nan
        dist_w4 = (close - wk_sma_4) / atr_14 if pd.notna(atr_14) and atr_14 != 0 else np.nan
        dist_w10 = (close - wk_sma_10) / atr_14 if pd.notna(atr_14) and atr_14 != 0 else np.nan

        # SUPPORT GAPS
        c_vals = df['Close'].values
        l_vals = df['Low'].values
        active_gaps = []
        for j in range(1, len(c_vals)):
            prev_c = c_vals[j-1]
            curr_l = l_vals[j]
            curr_c = c_vals[j]
            if curr_l > prev_c:
                active_gaps.append([curr_l, prev_c])
            for k in range(len(active_gaps)-1, -1, -1):
                if curr_c < active_gaps[k][0]:
                    if curr_c <= active_gaps[k][1]:
                        active_gaps.pop(k) 
                    else:
                        active_gaps[k][0] = curr_c 
            if len(active_gaps) > 20:
                active_gaps.pop(0)
                
        nearest_sg_top = np.nan
        if active_gaps:
            min_diff = float('inf')
            for gap in active_gaps:
                diff = close - gap[0]
                if diff >= 0 and diff < min_diff:
                    min_diff = diff
                    nearest_sg_top = gap[0]
                    
        dist_sg = (close - nearest_sg_top) / atr_14 if pd.notna(atr_14) and atr_14 != 0 and pd.notna(nearest_sg_top) else np.nan

        ret_1d = df['Close'].pct_change(periods=1).iloc[-1] * 100
        ret_1w = df['Close'].pct_change(periods=5).iloc[-1] * 100 
        
        if len(weekly_df) >= 2:
            ret_curr_wk_body = ((weekly_df['Close'].iloc[-1] / weekly_df['Open'].iloc[-1]) - 1) * 100
            ret_prev_wk_body = ((weekly_df['Close'].iloc[-2] / weekly_df['Open'].iloc[-2]) - 1) * 100
        else:
            ret_curr_wk_body = np.nan
            ret_prev_wk_body = np.nan

        ret_1m = df['Close'].pct_change(periods=21).iloc[-1] * 100 if len(df) >= 22 else None
        ret_3m = df['Close'].pct_change(periods=63).iloc[-1] * 100 if len(df) >= 64 else None
        ret_6m = df['Close'].pct_change(periods=126).iloc[-1] * 100 if len(df) >= 127 else None
        dist_ema21 = ((close - df['ema21'].iloc[-1]) / df['ema21'].iloc[-1]) * 100
        
        dist_high = ((close - high_52w) / high_52w) * 100
        dist_low = ((close - low_52w) / low_52w) * 100

        if len(df) >= 64:
            close_1m_ago = df['Close'].iloc[-22]
            close_3m_ago = df['Close'].iloc[-64]
            ret_past_2m_till_last_month = ((close_1m_ago / close_3m_ago) - 1) * 100
        else:
            ret_past_2m_till_last_month = None
            
        # 🟢 RS LINE DYNAMIC METRICS
        if bench_close is not None:
            aligned_bench = bench_close.reindex(df.index).ffill()
            rs_raw = (df['Close'] / aligned_bench) * 100
            
            rs_52w_high = rs_raw.tail(252).max()
            rs_off_high = ((rs_raw.iloc[-1] - rs_52w_high) / rs_52w_high) * 100 if pd.notna(rs_52w_high) and rs_52w_high > 0 else np.nan
            
            rs_21ema = rs_raw.ewm(span=21, adjust=False).mean()
            rs_dist_21ema = ((rs_raw.iloc[-1] - rs_21ema.iloc[-1]) / rs_21ema.iloc[-1]) * 100 if rs_21ema.iloc[-1] > 0 else np.nan
            
            rs_1w_chg = rs_raw.pct_change(periods=5).iloc[-1] * 100
            
            stock_ret = df['Close'].pct_change()
            bench_ret = aligned_bench.pct_change()
            is_outperforming = (stock_ret > bench_ret).astype(int)
            
            rs_out_10d = (is_outperforming.tail(10).sum() / 10.0) * 100 if len(is_outperforming) >= 10 else np.nan
            rs_out_20d = (is_outperforming.tail(20).sum() / 20.0) * 100 if len(is_outperforming) >= 20 else np.nan
            rs_out_50d = (is_outperforming.tail(50).sum() / 50.0) * 100 if len(is_outperforming) >= 50 else np.nan
        else:
            rs_off_high = rs_dist_21ema = rs_1w_chg = rs_out_10d = rs_out_20d = rs_out_50d = np.nan

        tech_metrics[stock_sym] = {
            "ADR %": round(adr_20, 2) if pd.notna(adr_20) else "",
            "RVOL (20D)": round(rvol_20, 2) if pd.notna(rvol_20) else "",
            "PPV (10D)": int(ppv_10d_count) if pd.notna(ppv_10d_count) else 0,
            "PPV (4W)": int(ppv_4w_count) if pd.notna(ppv_4w_count) else 0,
            "1 Day Return %": round(ret_1d, 2) if pd.notna(ret_1d) else "",
            "1 Week Return %": round(ret_1w, 2) if pd.notna(ret_1w) else "",
            "Curr Wk Body %": round(ret_curr_wk_body, 2) if pd.notna(ret_curr_wk_body) else "",
            "Prev Wk Body %": round(ret_prev_wk_body, 2) if pd.notna(ret_prev_wk_body) else "",
            "1 Month Return %": round(ret_1m, 2) if pd.notna(ret_1m) else "",
            "Prev 2M Return (Ending 1M Ago) %": round(ret_past_2m_till_last_month, 2) if pd.notna(ret_past_2m_till_last_month) else "",
            "3 Month Return %": round(ret_3m, 2) if pd.notna(ret_3m) else "",
            "6 Month Return %": round(ret_6m, 2) if pd.notna(ret_6m) else "",
            "% Dist from 21 EMA": round(dist_ema21, 2) if pd.notna(dist_ema21) else "",
            "% above 52W Low": round(dist_low, 2) if pd.notna(dist_low) else "",
            "% off 52W High": round(dist_high, 2) if pd.notna(dist_high) else "",
            "4 EMA": round(df['ema4'].iloc[-1], 2) if pd.notna(df['ema4'].iloc[-1]) else "",
            "6 EMA": round(df['ema6'].iloc[-1], 2) if pd.notna(df['ema6'].iloc[-1]) else "",
            "9 EMA": round(df['ema9'].iloc[-1], 2) if pd.notna(df['ema9'].iloc[-1]) else "",
            "21 EMA": round(df['ema21'].iloc[-1], 2) if pd.notna(df['ema21'].iloc[-1]) else "",
            "50 EMA": round(df['ema50'].iloc[-1], 2) if pd.notna(df['ema50'].iloc[-1]) else "",
            "4 EMA (ATR)": round(dist_4, 2) if pd.notna(dist_4) else "",
            "6 EMA (ATR)": round(dist_6, 2) if pd.notna(dist_6) else "",
            "9 EMA (ATR)": round(dist_9, 2) if pd.notna(dist_9) else "",
            "21 EMA (ATR)": round(dist_21, 2) if pd.notna(dist_21) else "",
            "50 EMA (ATR)": round(dist_50, 2) if pd.notna(dist_50) else "",
            "4W SMA (ATR)": round(dist_w4, 2) if pd.notna(dist_w4) else "",
            "10W SMA (ATR)": round(dist_w10, 2) if pd.notna(dist_w10) else "",
            "Supp Gap (ATR)": round(dist_sg, 2) if pd.notna(dist_sg) else "",
            "RS Line % off 52W High": round(rs_off_high, 2) if pd.notna(rs_off_high) else "",
            "% Dist from RS 21 EMA": round(rs_dist_21ema, 2) if pd.notna(rs_dist_21ema) else "",
            "RS Line 1W Change %": round(rs_1w_chg, 2) if pd.notna(rs_1w_chg) else "",
            "RS Outperform (10D) %": round(rs_out_10d, 2) if pd.notna(rs_out_10d) else "",
            "RS Outperform (20D) %": round(rs_out_20d, 2) if pd.notna(rs_out_20d) else "",
            "RS Outperform (50D) %": round(rs_out_50d, 2) if pd.notna(rs_out_50d) else "",
            "Webster RS (Current)": int(webster_friday_ranks[ticker].iloc[-1]) if not webster_friday_ranks.empty and ticker in webster_friday_ranks.columns and not pd.isna(webster_friday_ranks[ticker].iloc[-1]) else "",
            "Classic RS (Current)": int(classic_friday_ranks[ticker].iloc[-1]) if not classic_friday_ranks.empty and ticker in classic_friday_ranks.columns and not pd.isna(classic_friday_ranks[ticker].iloc[-1]) else "",
        }

        # STATIC CHART GEN
        df_chart = df.tail(65)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 2.5), gridspec_kw={'height_ratios': [3, 1]})
        fig.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.1, hspace=0.05)
        
        idx = np.arange(len(df_chart))
        up = df_chart['Close'] >= df_chart['Open']
        down = ~up
        
        ax1.vlines(idx[up], df_chart['Low'][up], df_chart['High'][up], color='#26a69a', linewidth=1)
        ax1.vlines(idx[down], df_chart['Low'][down], df_chart['High'][down], color='#ef5350', linewidth=1)
        ax1.bar(idx[up], df_chart['Close'][up] - df_chart['Open'][up], bottom=df_chart['Open'][up], color='#26a69a', width=0.6)
        ax1.bar(idx[down], df_chart['Open'][down] - df_chart['Close'][down], bottom=df_chart['Close'][down], color='#ef5350', width=0.6)
        
        ax1.plot(idx, df_chart['ema4'], color='#87CEFA', linewidth=0.8, label='4 EMA')
        ax1.plot(idx, df_chart['ema9'], color='#800080', linewidth=0.8, label='9 EMA')
        ax1.plot(idx, df_chart['ema21'], color='#CC9900', linewidth=0.8, label='21 EMA')
        ax1.plot(idx, df_chart['ema50'], color='#FF0000', linewidth=0.8, label='50 EMA')
        
        ax1.set_title(f"{stock_sym}", color='white', fontsize=10, fontweight='bold', loc='left', pad=2)
        ax1.grid(True, alpha=0.2)
        ax1.set_xticklabels([])
 
        ax2.bar(idx[up], df_chart['Volume'][up], color='#26a69a', alpha=0.4, width=0.6)
        ax2.bar(idx[down], df_chart['Volume'][down], color='#ef5350', alpha=0.4, width=0.6)
        ax2.grid(True, alpha=0.2)
        ax2.set_yticklabels([])
        
        step = max(1, len(df_chart) // 3)
        ax2.set_xticks(idx[::step])
        ax2.set_xticklabels(df_chart.index.strftime('%b %d')[::step], rotation=0)

        plt.savefig(f"charts/{stock_sym}.png", dpi=120, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)

    except Exception as e:
        print(f"Error processing {stock_sym}: {e}")
        continue

# ==========================================
# 4.5 GENERATE INDEX CHARTS FOR BREADTH VIEW
# ==========================================
print("\nGenerating Nifty 50 and Smallcap Index Charts...")
try:
    idx_data = yf.download(['^NSEI', '^CNXSC'], period="1y", group_by="ticker", progress=False)
    for idx_sym, file_name in [('^NSEI', 'NIFTY50'), ('^CNXSC', 'SMALLCAP100')]:
        if idx_sym in idx_data.columns.levels[0]:
            df_idx = idx_data[idx_sym].dropna(subset=['Close']).tail(65)
       
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 2.5), gridspec_kw={'height_ratios': [3, 1]})
            fig.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.1, hspace=0.05)
            
            idx = np.arange(len(df_idx))
            up = df_idx['Close'] >= df_idx['Open']
            down = ~up
            
            ax1.vlines(idx[up], df_idx['Low'][up], df_idx['High'][up], color='#26a69a', linewidth=1)
            ax1.vlines(idx[down], df_idx['Low'][down], df_idx['High'][down], color='#ef5350', linewidth=1)
            ax1.bar(idx[up], df_idx['Close'][up] - df_idx['Open'][up], bottom=df_idx['Open'][up], color='#26a69a', width=0.6)
            ax1.bar(idx[down], df_idx['Open'][down] - df_idx['Close'][down], bottom=df_idx['Close'][down], color='#ef5350', width=0.6)
            
            ax1.set_title(file_name, color='white', fontsize=10, fontweight='bold', loc='left', pad=2)
    
            ax1.grid(True, alpha=0.2)
            ax1.set_xticklabels([])
            
            if 'Volume' in df_idx.columns and df_idx['Volume'].sum() > 0:
                ax2.bar(idx[up], df_idx['Volume'][up], color='#26a69a', alpha=0.4, width=0.6)
                ax2.bar(idx[down], df_idx['Volume'][down], color='#ef5350', alpha=0.4, width=0.6)
        
            ax2.grid(True, alpha=0.2)
            ax2.set_yticklabels([])
            
            step = max(1, len(df_idx) // 3)
            ax2.set_xticks(idx[::step])
            ax2.set_xticklabels(df_idx.index.strftime('%b %d')[::step], rotation=0)

            plt.savefig(f"charts/{file_name}.png", dpi=120, facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close(fig)
except Exception as e:
    print(f"Error generating index charts: {e}")

# ==========================================
# 5. PUSH IMAGES TO PUBLIC GITHUB REPO
# ==========================================
print("\nPushing generated charts to public CDN repository...")
GITHUB_USER = "abhinavbhuvns-creator"
PUBLIC_REPO = "alpha-charts"
GH_PAT = os.environ.get('GH_PAT')

if GH_PAT:
    try:
        repo_url = f"https://{GH_PAT}@github.com/{GITHUB_USER}/{PUBLIC_REPO}.git"
        os.system(f"git clone {repo_url}")
        
        for file in os.listdir("charts"):
            shutil.copy(os.path.join("charts", file), os.path.join(PUBLIC_REPO, file))
          
        os.chdir(PUBLIC_REPO)
        os.system("git config user.name 'github-actions[bot]'")
        os.system("git config user.email 'github-actions[bot]@users.noreply.github.com'")
        os.system("git add .")
        os.system("git commit -m 'Auto-update daily static charts'")
        os.system("git push")
        os.chdir("..")
        print("✅ Charts successfully synced to public CDN!")
    except Exception as e:
        print(f"⚠️ Error pushing charts: {e}")
else:
    print("⚠️ GH_PAT secret not found. Skipping chart push to public repo.")

# ==========================================
# 6. MERGE DATA & WRITE TO GOOGLE SHEET
# ==========================================
print("Updating Google Sheets Master Framework...")
df_tech = pd.DataFrame.from_dict(tech_metrics, orient='index').reset_index()
df_tech.rename(columns={'index': 'Symbol'}, inplace=True)
df_final = pd.merge(df_master, df_tech, on='Symbol', how='left').fillna("")

# 🟢 MERGE MARKETSMITH DATA
if not df_ms.empty:
    df_final = pd.merge(df_final, df_ms, on='Symbol', how='left')
    for col in [c for c in df_ms.columns if c != 'Symbol']:
        df_final[col] = df_final[col].replace(r'^\s*$', np.nan, regex=True).fillna("N/A")

try:
    target_ws = spreadsheet.worksheet(TARGET_TAB_NAME)
    target_ws.clear()
except gspread.exceptions.WorksheetNotFound:
    target_ws = spreadsheet.add_worksheet(title=TARGET_TAB_NAME, rows="2500", cols="40")

sheet_output = [df_final.columns.values.tolist()] + df_final.values.tolist()
target_ws.update(values=sheet_output, range_name='A1', value_input_option='USER_ENTERED')
target_ws.freeze(rows=1)
target_ws.format('A1:Z1', {'textFormat': {'bold': True}, "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9}})

# ==========================================
# 6.5 WRITE HISTORICAL RS RATINGS TO SHEETS
# ==========================================
if not webster_friday_ranks.empty and not classic_friday_ranks.empty:
    print("\nExporting last 6 Months of Friday RS Ratings to Google Sheets...")
    
    recent_fridays = fridays_idx[-26:]
    webster_hist = webster_friday_ranks.loc[recent_fridays].round(0).fillna("")
    classic_hist = classic_friday_ranks.loc[recent_fridays].round(0).fillna("")
    
    webster_hist.columns = [c.replace('.NS', '') for c in webster_hist.columns]
    webster_hist.index = webster_hist.index.strftime('%Y-%m-%d')
    webster_hist = webster_hist.reset_index().rename(columns={'Date': 'Friday_Date'})
    
    classic_hist.columns = [c.replace('.NS', '') for c in classic_hist.columns]
    classic_hist.index = classic_hist.index.strftime('%Y-%m-%d')
    classic_hist = classic_hist.reset_index().rename(columns={'Date': 'Friday_Date'})
    
    try:
        w_ws = spreadsheet.worksheet("Webster_3M_History")
        w_ws.clear()
    except gspread.exceptions.WorksheetNotFound:
        w_ws = spreadsheet.add_worksheet(title="Webster_3M_History", rows="100", cols="2100")
    w_ws.update(values=[webster_hist.columns.values.tolist()] + webster_hist.values.tolist(), range_name='A1')
    w_ws.freeze(rows=1, cols=1)
    
    try:
        c_ws = spreadsheet.worksheet("Classic_12M_History")
        c_ws.clear()
    except gspread.exceptions.WorksheetNotFound:
        c_ws = spreadsheet.add_worksheet(title="Classic_12M_History", rows="100", cols="2100")
    c_ws.update(values=[classic_hist.columns.values.tolist()] + classic_hist.values.tolist(), range_name='A1')
    c_ws.freeze(rows=1, cols=1)

# ==========================================
# 7. CALCULATE 6-MONTH MARKET BREADTH
# ==========================================
print("\nCalculating 6-Month Market Breadth History...")
try:
    valid_tickers = [t for t in tickers if t in data.columns.levels[0] and t not in ['^NSEI', '^CNXSC']]
    
    close_df = pd.DataFrame({t: data[t]['Close'] for t in valid_tickers})
    vol_df = pd.DataFrame({t: data[t]['Volume'] for t in valid_tickers})
    high_df = pd.DataFrame({t: data[t]['High'] for t in valid_tickers})
    low_df = pd.DataFrame({t: data[t]['Low'] for t in valid_tickers})

    prev_close = close_df.shift(1)
    up_counts = (close_df > prev_close).sum(axis=1)
    down_counts = (close_df < prev_close).sum(axis=1)
    
    rolling_up_5 = up_counts.rolling(5).sum()
    rolling_down_5 = down_counts.rolling(5).sum()
    ratio_5d = round(rolling_up_5 / rolling_down_5.replace(0, np.nan), 2)

    sma50 = close_df.rolling(50).mean()
    sma200 = close_df.rolling(200).mean()
    above_50 = (close_df > sma50).sum(axis=1)
    below_50 = (close_df < sma50).sum(axis=1)
    above_200 = (close_df > sma200).sum(axis=1)
    below_200 = (close_df < sma200).sum(axis=1)

    high252 = high_df.rolling(252).max()
    low252 = low_df.rolling(252).min()
    new_highs = (high_df >= high252).sum(axis=1)
    new_lows = (low_df <= low252).sum(axis=1)

    avg_vol_50 = vol_df.rolling(50).mean()
    up_on_vol = ((close_df.pct_change() >= 0.045) & (vol_df > 2 * avg_vol_50) & (close_df > 40)).sum(axis=1)
    
    prev_max_252 = prev_close.rolling(252).max()
    high_52w_scan = ((close_df > prev_max_252) & (close_df > 40)).sum(axis=1)

    ema21 = close_df.ewm(span=21, adjust=False).mean()
    ema50_df = close_df.ewm(span=50, adjust=False).mean()
    ema150 = close_df.ewm(span=150, adjust=False).mean()
    wl_one_day = ((ema21 > ema50_df) & (ema50_df > ema150) & (close_df > ema50_df) & (close_df > 40)).sum(axis=1)

    breadth_df = pd.DataFrame({
        'Date': close_df.index.strftime('%Y-%m-%d'),
        'Stock Up': up_counts,
        'Stock Down': down_counts,
        '5 Day Ratio': ratio_5d,
        'New High': new_highs,
        'New Low': new_lows,
        'Above 50 SMA': above_50,
        'Below 50 SMA': below_50,
        'Above 200 SMA': above_200,
        'Below 200 SMA': below_200,
        'Up on Volume': up_on_vol,
        '52 Week High': high_52w_scan,
        'Watchlist One Day': wl_one_day
    }).dropna().tail(130)

    breadth_tab = "Market_Breadth_History"
    try:
        breadth_ws = spreadsheet.worksheet(breadth_tab)
        breadth_ws.clear()
    except gspread.exceptions.WorksheetNotFound:
        breadth_ws = spreadsheet.add_worksheet(title=breadth_tab, rows="200", cols="15")
    
    breadth_out = [breadth_df.columns.values.tolist()] + breadth_df.values.tolist()
    breadth_ws.update(values=breadth_out, range_name='A1', value_input_option='USER_ENTERED')
    breadth_ws.format('A1:M1', {'textFormat': {'bold': True}, "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9}})
    print(" -> Successfully overwrote 6-month breadth history.")
except Exception as e:
    print(f" -> Could not calculate market breadth: {e}")

# ==========================================
# 8. BUILD SYNTHETIC ETF CHARTS (GROWTH50)
# ==========================================
print("\nBuilding Synthetic Equal-Weight ETF for GROWTH50...")
try:
    ws_growth = spreadsheet.worksheet("GROWTH50")
    records = ws_growth.get_all_records()
    df_etf_stocks = pd.DataFrame(records)
    
    sym_col = next((c for c in df_etf_stocks.columns if 'symbol' in c.lower() or 'ticker' in c.lower()), None)
    if sym_col:
        etf_tickers = (df_etf_stocks[sym_col].astype(str).str.strip().str.upper() + '.NS').tolist()
        valid_tickers = [t for t in etf_tickers if t in data.columns.levels[0] and t not in ['^NSEI', '^CNXSC']]
        
        if valid_tickers:
            df_hist = data[valid_tickers].tail(252).ffill().bfill()
            index_open = pd.Series(0.0, index=df_hist.index)
            index_high = pd.Series(0.0, index=df_hist.index)
            index_low = pd.Series(0.0, index=df_hist.index)
            index_close = pd.Series(0.0, index=df_hist.index)
            
            valid_count = 0
            for t in valid_tickers:
                c_series = df_hist[t]['Close']
                if c_series.empty or pd.isna(c_series.iloc[0]) or c_series.iloc[0] == 0: continue
                factor = 100.0 / c_series.iloc[0]
                index_open += df_hist[t]['Open'] * factor
                index_high += df_hist[t]['High'] * factor
                index_low += df_hist[t]['Low'] * factor
                index_close += df_hist[t]['Close'] * factor
                valid_count += 1
            
            if valid_count > 0:
                df_index = pd.DataFrame({
                    'Open': round(index_open / valid_count, 2),
                    'High': round(index_high / valid_count, 2),
                    'Low': round(index_low / valid_count, 2),
                    'Close': round(index_close / valid_count, 2)
                })
                
                df_index['9 EMA'] = round(df_index['Close'].ewm(span=9, adjust=False).mean(), 2)
                df_index['21 EMA'] = round(df_index['Close'].ewm(span=21, adjust=False).mean(), 2)
                df_index['50 EMA'] = round(df_index['Close'].ewm(span=50, adjust=False).mean(), 2)
                
                df_idx_chart = df_index.tail(65).copy()
                fig, ax1 = plt.subplots(1, 1, figsize=(4, 2.5))
                fig.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.15)
                
                idx = np.arange(len(df_idx_chart))
                up = df_idx_chart['Close'] >= df_idx_chart['Open']
                down = ~up
                
                ax1.vlines(idx[up], df_idx_chart['Low'][up], df_idx_chart['High'][up], color='#26a69a', linewidth=1)
                ax1.vlines(idx[down], df_idx_chart['Low'][down], df_idx_chart['High'][down], color='#ef5350', linewidth=1)
                ax1.bar(idx[up], df_idx_chart['Close'][up] - df_idx_chart['Open'][up], bottom=df_idx_chart['Open'][up], color='#26a69a', width=0.6)
                ax1.bar(idx[down], df_idx_chart['Open'][down] - df_idx_chart['Close'][down], bottom=df_idx_chart['Close'][down], color='#ef5350', width=0.6)
                
                ax1.plot(idx, df_idx_chart['9 EMA'], color='#87CEFA', linewidth=0.8, label='9 EMA')
                ax1.plot(idx, df_idx_chart['21 EMA'], color='#26a69a', linewidth=0.8, label='21 EMA')
                ax1.plot(idx, df_idx_chart['50 EMA'], color='#f5b041', linewidth=0.8, label='50 EMA')
                
                ax1.set_title("GROWTH50 Index", color='white', fontsize=10, fontweight='bold', loc='left', pad=2)
                ax1.grid(True, alpha=0.2)
                
                step = max(1, len(df_idx_chart) // 3)
                ax1.set_xticks(idx[::step])
                
                df_idx_chart.reset_index(drop=True, inplace=True)
                ax1.set_xticklabels(pd.to_datetime(df_idx_chart['Date']).dt.strftime('%b %d')[::step], rotation=0)

                plt.savefig(f"charts/Chart_GROWTH50.png", dpi=120, facecolor=fig.get_facecolor(), edgecolor='none')
                plt.close(fig)

                df_index.reset_index(inplace=True)
                df_index['Date'] = df_index['Date'].dt.strftime('%Y-%m-%d')
                
                out_tab = "Chart_GROWTH50"
                try:
                    out_ws = spreadsheet.worksheet(out_tab)
                    out_ws.clear()
                except:
                    out_ws = spreadsheet.add_worksheet(title=out_tab, rows="300", cols="10")
                    
                out_data = [df_index.columns.values.tolist()] + df_index.values.tolist()
                out_ws.update(values=out_data, range_name='A1')
except Exception as e:
    print(f" -> Could not build ETF chart: {e}")

print(f"\n✅ Success! Systems completely aligned.")
