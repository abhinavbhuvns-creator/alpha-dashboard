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
import matplotlib.subplots as plt_sub
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore')
pd.options.mode.chained_assignment = None
os.makedirs('charts', exist_ok=True)

# ==========================================
# 1. AUTHENTICATE USING GITHUB SECRETS
# ==========================================
print("Authenticating with Google Service Account...")
creds_json = os.environ.get('GOOGLE_CREDENTIALS')
if not creds_json: raise SystemExit("🛑 Error: GOOGLE_CREDENTIALS not found in GitHub Secrets.")

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

try:
    master_ws = spreadsheet.worksheet(MASTER_TAB_NAME)
    df_master = pd.DataFrame(master_ws.get_all_records())
except Exception as e:
    raise SystemExit(f"🛑 Error: Could not read Master tab. {e}")

try:
    ms_ws = spreadsheet.worksheet("marketsmith database")
    df_ms = pd.DataFrame(ms_ws.get_all_records())
    if 'Symbol' in df_ms.columns:
        df_ms['Symbol'] = df_ms['Symbol'].astype(str).str.strip()
        df_ms = df_ms[['Symbol'] + [c for c in MS_COLS if c in df_ms.columns]]
    else: df_ms = pd.DataFrame(columns=['Symbol'])
except:
    df_ms = pd.DataFrame(columns=['Symbol'])

tickers = (df_master['Symbol'].str.strip() + '.NS').tolist()
if '^NSEI' not in tickers: tickers.append('^NSEI')

# ==========================================
# 3. BATCH DOWNLOAD (2 YEARS FOR SPEED)
# ==========================================
print(f"Downloading 2 years of price history for {len(tickers)} stocks...")
def chunker(seq, size): return (seq[pos:pos + size] for pos in range(0, len(seq), size))
ticker_chunks = list(chunker(tickers, 250))
all_chunks = []
for i, chunk in enumerate(ticker_chunks):
    chunk_data = yf.download(chunk, period="2y", group_by="ticker", threads=True, progress=False)
    all_chunks.append(chunk_data)
    time.sleep(2) 

data = pd.concat(all_chunks, axis=1)
if not data.empty:
    data.index = pd.to_datetime(data.index).tz_localize(None).normalize()
    data = data.groupby(data.index).max() 

# ==========================================
# 4. VECTORIZED RELATIVE STRENGTH RANKING
# ==========================================
print("\nCalculating Cross-Sectional RS Ratings...")
close_data = pd.DataFrame()
for t in tickers:
    if t in data.columns.levels[0] and 'Close' in data[t]: close_data[t] = data[t]['Close']
nifty = close_data['^NSEI'] if '^NSEI' in close_data.columns else None
close_data.drop(columns=['^NSEI'], inplace=True, errors='ignore')

if nifty is not None and not close_data.empty:
    rs_line_base = close_data.div(nifty, axis=0)
    webster_base = rs_line_base.ewm(span=63, adjust=False).mean()
    roc_63 = close_data / close_data.shift(63)
    roc_126 = close_data / close_data.shift(126)
    roc_189 = close_data / close_data.shift(189)
    roc_252 = close_data / close_data.shift(252)
    classic_base = (roc_63 * 2) + roc_126 + roc_189 + roc_252
    fridays_idx = close_data.index[close_data.index.dayofweek == 4]
    
    def calc_1_to_99_rank(row): return (row.rank(pct=True) * 98) + 1
    webster_friday_ranks = webster_base.loc[fridays_idx].apply(calc_1_to_99_rank, axis=1)
    classic_friday_ranks = classic_base.loc[fridays_idx].apply(calc_1_to_99_rank, axis=1)
else:
    webster_friday_ranks, classic_friday_ranks = pd.DataFrame(), pd.DataFrame()

# ==========================================
# 5. TECH METRICS & CHART GENERATION
# ==========================================
print("\nGenerating Sheet Analytics & Custom Visual Chart PNGs...")
tech_metrics = {}
plt.style.use('dark_background')
plt.rcParams.update({'figure.facecolor': '#131722', 'axes.facecolor': '#131722', 'axes.edgecolor': '#2a2e39', 'grid.color': '#2a2e39', 'xtick.color': '#8a93a6', 'ytick.color': '#8a93a6', 'font.size': 8})

for ticker in tickers:
    if ticker == '^NSEI': continue 
    stock_sym = ticker.replace('.NS', '')
    try:
        df = data[ticker].copy().dropna(subset=['Close']) if len(tickers)>1 else data.copy().dropna(subset=['Close'])
        if len(df) < 5: continue 

        close = df['Close'].iloc[-1]
        high_52w = df['High'].tail(252).max() 
        safe_low_series = df['Low'].replace(0, np.nan)
        low_52w = safe_low_series.tail(252).min() 
        
        df['tr1'] = df['High'] - df['Low']
        df['tr2'] = abs(df['High'] - df['Close'].shift(1))
        df['tr3'] = abs(df['Low'] - df['Close'].shift(1))
        atr_14 = df[['tr1', 'tr2', 'tr3']].max(axis=1).rolling(14, min_periods=1).mean().iloc[-1]
        
        df['Safe_Low'] = df['Low'].replace(0, np.nan)
        adr_20 = ((df['High'] / df['Safe_Low']) - 1).replace([np.inf, -np.inf], np.nan).rolling(window=20, min_periods=5).mean().iloc[-1] * 100
        
        avg_vol_20 = df['Volume'].rolling(window=20, min_periods=1).mean().iloc[-1]
        rvol_20 = (df['Volume'].iloc[-1] / avg_vol_20) if avg_vol_20 > 0 else np.nan

        daily_down_vols = df['Volume'].where(df['Close'] < df['Close'].shift(1), 0)
        max_down_vol_10d = daily_down_vols.shift(1).rolling(10, min_periods=1).max()
        ppv_10d_count = ((df['Close'] > df['Close'].shift(1)) & (df['Volume'] > max_down_vol_10d)).tail(10).sum()

        df['ema4'] = df['Close'].ewm(span=4, adjust=False, min_periods=1).mean()
        df['ema6'] = df['Close'].ewm(span=6, adjust=False, min_periods=1).mean()
        df['ema9'] = df['Close'].ewm(span=9, adjust=False, min_periods=1).mean()
        df['ema21'] = df['Close'].ewm(span=21, adjust=False, min_periods=1).mean()
        df['ema50'] = df['Close'].ewm(span=50, adjust=False, min_periods=1).mean()
   
        weekly_df = df.resample('W-FRI').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
        if len(weekly_df) >= 10:
            wk_sma_4 = weekly_df['Close'].rolling(4).mean().iloc[-1]
            wk_sma_10 = weekly_df['Close'].rolling(10).mean().iloc[-1]
            weekly_down_vols = weekly_df['Volume'].where(weekly_df['Close'] < weekly_df['Close'].shift(1), 0)
            max_down_vol_10w = weekly_down_vols.shift(1).rolling(10).max()
            ppv_4w_count = ((weekly_df['Close'] > weekly_df['Close'].shift(1)) & (weekly_df['Volume'] > max_down_vol_10w)).tail(4).sum()
        else: wk_sma_4 = wk_sma_10 = np.nan; ppv_4w_count = 0

        dist_4 = (close - df['ema4'].iloc[-1]) / atr_14 if pd.notna(atr_14) and atr_14 != 0 else np.nan
        dist_6 = (close - df['ema6'].iloc[-1]) / atr_14 if pd.notna(atr_14) and atr_14 != 0 else np.nan
        dist_9 = (close - df['ema9'].iloc[-1]) / atr_14 if pd.notna(atr_14) and atr_14 != 0 else np.nan
        dist_21 = (close - df['ema21'].iloc[-1]) / atr_14 if pd.notna(atr_14) and atr_14 != 0 else np.nan
        dist_50 = (close - df['ema50'].iloc[-1]) / atr_14 if pd.notna(atr_14) and atr_14 != 0 else np.nan
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
                
        nearest_sg_top = np.nan
        if active_gaps:
            min_diff = float('inf')
            for gap in active_gaps:
                diff = close - gap[0]
                if 0 <= diff < min_diff: min_diff = diff; nearest_sg_top = gap[0]
        dist_sg = (close - nearest_sg_top) / atr_14 if pd.notna(atr_14) and atr_14 != 0 and pd.notna(nearest_sg_top) else np.nan

        ret_1d = df['Close'].pct_change(periods=1).iloc[-1] * 100
        ret_1w = df['Close'].pct_change(periods=5).iloc[-1] * 100 
        if len(weekly_df) >= 2:
            ret_curr_wk_body = ((weekly_df['Close'].iloc[-1] / weekly_df['Open'].iloc[-1]) - 1) * 100
            ret_prev_wk_body = ((weekly_df['Close'].iloc[-2] / weekly_df['Open'].iloc[-2]) - 1) * 100
        else: ret_curr_wk_body = ret_prev_wk_body = np.nan

        ret_1m = df['Close'].pct_change(periods=21).iloc[-1] * 100 if len(df) >= 22 else None
        ret_3m = df['Close'].pct_change(periods=63).iloc[-1] * 100 if len(df) >= 64 else None
        ret_6m = df['Close'].pct_change(periods=126).iloc[-1] * 100 if len(df) >= 127 else None
        dist_ema21 = ((close - df['ema21'].iloc[-1]) / df['ema21'].iloc[-1]) * 100
        dist_high = ((close - high_52w) / high_52w) * 100
        dist_low = ((close - low_52w) / low_52w) * 100

        ret_past_2m_till_last_month = ((df['Close'].iloc[-22] / df['Close'].iloc[-64]) - 1) * 100 if len(df) >= 64 else None

        if '^NSEI' in data.columns.levels[0]:
            aligned_nifty = data['^NSEI']['Close'].reindex(df.index).ffill()
            df['RS_Line'] = df['Close'] / aligned_nifty
            df['RS_21_EMA'] = df['RS_Line'].ewm(span=21, adjust=False, min_periods=1).mean()
            rs_52w_high = df['RS_Line'].tail(252).max()
            
            current_rs = df['RS_Line'].iloc[-1]
            current_rs_21ema = df['RS_21_EMA'].iloc[-1]
            
            rs_off_high = ((current_rs - rs_52w_high) / rs_52w_high) * 100 if pd.notna(rs_52w_high) and rs_52w_high > 0 else np.nan
            dist_rs_21ema = ((current_rs - current_rs_21ema) / current_rs_21ema) * 100 if pd.notna(current_rs_21ema) else np.nan
            rs_1w_chg = df['RS_Line'].pct_change(periods=5, fill_method=None).iloc[-1] * 100
            
            is_out = (df['Close'].pct_change(fill_method=None) > aligned_nifty.pct_change(fill_method=None)).astype(int)
            len_10, len_20, len_50 = float(len(is_out.tail(10))), float(len(is_out.tail(20))), float(len(is_out.tail(50)))
            
            rs_10d_outperf = (is_out.tail(10).sum() / len_10) * 100 if len_10 > 0 else np.nan
            rs_20d_outperf = (is_out.tail(20).sum() / len_20) * 100 if len_20 > 0 else np.nan
            rs_50d_outperf = (is_out.tail(50).sum() / len_50) * 100 if len_50 > 0 else np.nan
        else: rs_off_high = dist_rs_21ema = rs_1w_chg = rs_10d_outperf = rs_20d_outperf = rs_50d_outperf = np.nan

        tech_metrics[stock_sym] = {
            "RS Line % off 52W High": round(rs_off_high, 2) if pd.notna(rs_off_high) else "",
            "% Dist from RS 21 EMA": round(dist_rs_21ema, 2) if pd.notna(dist_rs_21ema) else "",
            "RS Line 1W Change %": round(rs_1w_chg, 2) if pd.notna(rs_1w_chg) else "",
            "RS Outperform (10D) %": round(rs_10d_outperf, 2) if pd.notna(rs_10d_outperf) else "",
            "RS Outperform (20D) %": round(rs_20d_outperf, 2) if pd.notna(rs_20d_outperf) else "",
            "RS Outperform (50D) %": round(rs_50d_outperf, 2) if pd.notna(rs_50d_outperf) else "",
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
            "Supp Gap (ATR)": round(dist_sg, 2) if pd.notna(dist_sg) else ""
        }

        # STATIC CHART GEN
        df_chart = df.tail(65)
        fig, (ax1, ax2) = plt_sub.subplots(2, 1, figsize=(4, 2.5), gridspec_kw={'height_ratios': [3, 1]})
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
# 6. GENERATE INDEX CHARTS FOR BREADTH VIEW
# ==========================================
print("\nGenerating Nifty 50 and Smallcap Index Charts...")
try:
    idx_data = yf.download(['^NSEI', '^CRSLDX'], period="1y", group_by="ticker", progress=False)
    for idx_sym, file_name in [('^NSEI', 'NIFTY50'), ('^CRSLDX', 'SMALLCAP250')]:
        if idx_sym in idx_data.columns.levels[0]:
            df_idx = idx_data[idx_sym].dropna(subset=['Close']).tail(65)
            fig, (ax1, ax2) = plt_sub.subplots(2, 1, figsize=(4, 2.5), gridspec_kw={'height_ratios': [3, 1]})
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
except Exception as e: print(f"Error generating index charts: {e}")

# ==========================================
# 7. PUSH IMAGES TO PUBLIC GITHUB REPO
# ==========================================
print("\nPushing generated charts to public CDN repository...")
GITHUB_USER = "abhinavbhuvns-creator"
PUBLIC_REPO = "alpha-charts"
GH_PAT = os.environ.get('GH_PAT')
if GH_PAT:
    try:
        repo_url = f"https://{GH_PAT}@github.com/{GITHUB_USER}/{PUBLIC_REPO}.git"
        os.system(f"git clone {repo_url}")
        for file in os.listdir("charts"): shutil.copy(os.path.join("charts", file), os.path.join(PUBLIC_REPO, file))
        os.chdir(PUBLIC_REPO)
        os.system("git config user.name 'github-actions[bot]'")
        os.system("git config user.email 'github-actions[bot]@users.noreply.github.com'")
        os.system("git add .")
        os.system("git commit -m 'Auto-update daily static charts'")
        os.system("git push")
        os.chdir("..")
        print("✅ Charts successfully synced to public CDN!")
    except Exception as e: print(f"⚠️ Error pushing charts: {e}")
else: print("⚠️ GH_PAT secret not found. Skipping chart push to public repo.")

# ==========================================
# 8. MERGE DATA & WRITE TO GOOGLE SHEET
# ==========================================
print("Updating Google Sheets Master Framework...")
df_tech = pd.DataFrame.from_dict(tech_metrics, orient='index').reset_index()
df_tech.rename(columns={'index': 'Symbol'}, inplace=True)
df_final = pd.merge(df_master, df_tech, on='Symbol', how='left').fillna("")

if not df_ms.empty:
    df_final = pd.merge(df_final, df_ms, on='Symbol', how='left')
    for col in [c for c in df_ms.columns if c != 'Symbol']: df_final[col] = df_final[col].replace(r'^\s*$', np.nan, regex=True).fillna("N/A")
df_final = df_final.replace([np.inf, -np.inf], np.nan).fillna("")

try: target_ws = spreadsheet.worksheet(TARGET_TAB_NAME); target_ws.clear()
except: target_ws = spreadsheet.add_worksheet(title=TARGET_TAB_NAME, rows="2500", cols="40")

sheet_output = [df_final.columns.values.tolist()] + df_final.values.tolist()
target_ws.update(values=sheet_output, range_name='A1', value_input_option='USER_ENTERED')
target_ws.freeze(rows=1)
target_ws.format('A1:Z1', {'textFormat': {'bold': True}, "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9}})

if not webster_friday_ranks.empty and not classic_friday_ranks.empty:
    recent_fridays = fridays_idx[-26:]
    webster_hist = webster_friday_ranks.loc[recent_fridays].round(0).fillna("")
    classic_hist = classic_friday_ranks.loc[recent_fridays].round(0).fillna("")
    
    webster_hist.columns = [c.replace('.NS', '') for c in webster_hist.columns]
    webster_hist.index = webster_hist.index.strftime('%Y-%m-%d')
    webster_hist = webster_hist.reset_index().rename(columns={'Date': 'Friday_Date'}).replace([np.inf, -np.inf], np.nan).fillna("")
    
    classic_hist.columns = [c.replace('.NS', '') for c in classic_hist.columns]
    classic_hist.index = classic_hist.index.strftime('%Y-%m-%d')
    classic_hist = classic_hist.reset_index().rename(columns={'Date': 'Friday_Date'}).replace([np.inf, -np.inf], np.nan).fillna("")
    
    try: w_ws = spreadsheet.worksheet("Webster_3M_History"); w_ws.clear()
    except: w_ws = spreadsheet.add_worksheet(title="Webster_3M_History", rows="100", cols="2100")
    w_ws.update(values=[webster_hist.columns.values.tolist()] + webster_hist.values.tolist(), range_name='A1')
    w_ws.freeze(rows=1, cols=1)
    
    try: c_ws = spreadsheet.worksheet("Classic_12M_History"); c_ws.clear()
    except: c_ws = spreadsheet.add_worksheet(title="Classic_12M_History", rows="100", cols="2100")
    c_ws.update(values=[classic_hist.columns.values.tolist()] + classic_hist.values.tolist(), range_name='A1')
    c_ws.freeze(rows=1, cols=1)

# ==========================================
# 9. MARKET BREADTH (DAILY APPEND LOGIC)
# ==========================================
print("\nUpdating Market Breadth History (Daily Append)...")
try:
    try:
        breadth_ws = spreadsheet.worksheet("Market_Breadth_History")
        existing_breadth_data = breadth_ws.get_all_records()
        df_existing_breadth = pd.DataFrame(existing_breadth_data)
    except: df_existing_breadth = pd.DataFrame()

    valid_tickers = [t for t in tickers if t in data.columns.levels[0] and t != '^NSEI']
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

    ema9_df_b = close_df.ewm(span=9, adjust=False, min_periods=1).mean()
    ema21_df_b = close_df.ewm(span=21, adjust=False, min_periods=1).mean()
    above_9 = (close_df > ema9_df_b).sum(axis=1)
    below_9 = (close_df < ema9_df_b).sum(axis=1)
    above_21 = (close_df > ema21_df_b).sum(axis=1)
    below_21 = (close_df < ema21_df_b).sum(axis=1)

    sma50 = close_df.rolling(50, min_periods=1).mean()
    sma200 = close_df.rolling(200, min_periods=1).mean()
    above_50 = (close_df > sma50).sum(axis=1)
    below_50 = (close_df < sma50).sum(axis=1)
    above_200 = (close_df > sma200).sum(axis=1)
    below_200 = (close_df < sma200).sum(axis=1)

    high252 = high_df.rolling(252, min_periods=1).max()
    low252 = low_df.rolling(252, min_periods=1).min()
    new_highs = (high_df >= high252).sum(axis=1)
    new_lows = (low_df <= low252).sum(axis=1)

    avg_vol_50 = vol_df.rolling(50, min_periods=1).mean()
    up_on_vol = ((close_df.pct_change() >= 0.045) & (vol_df > 2 * avg_vol_50) & (close_df > 40)).sum(axis=1)
    
    prev_max_252 = prev_close.rolling(252, min_periods=1).max()
    high_52w_scan = ((close_df > prev_max_252) & (close_df > 40)).sum(axis=1)

    ema150 = close_df.ewm(span=150, adjust=False, min_periods=1).mean()
    wl_one_day = ((ema21_df_b > sma50) & (sma50 > ema150) & (close_df > sma50) & (close_df > 40)).sum(axis=1)

    # 🟢 GRAB ONLY THE LAST 5 DAYS (To ensure overlap captures latest market changes safely)
    breadth_recent_df = pd.DataFrame({
        'Date': close_df.index.strftime('%Y-%m-%d'),
        'Stock Up': up_counts,
        'Stock Down': down_counts,
        '5 Day Ratio': ratio_5d,
        'New High': new_highs,
        'New Low': new_lows,
        'Above 9 EMA': above_9,
        'Below 9 EMA': below_9,
        'Above 21 EMA': above_21,
        'Below 21 EMA': below_21,
        'Above 50 SMA': above_50,
        'Below 50 SMA': below_50,
        'Above 200 SMA': above_200,
        'Below 200 SMA': below_200,
        'Up on Volume': up_on_vol,
        '52 Week High': high_52w_scan,
        'Watchlist One Day': wl_one_day
    }).dropna().tail(5)

    # 🟢 MERGE & TRIM TO MAINTAIN 5-YEAR (1260 days) LIMIT
    combined_breadth = pd.concat([breadth_recent_df, df_existing_breadth])
    combined_breadth = combined_breadth.drop_duplicates(subset=['Date'], keep='first')
    combined_breadth = combined_breadth.sort_values('Date', ascending=False).head(1260)
    combined_breadth = combined_breadth.replace([np.inf, -np.inf], np.nan).fillna("")

    try: breadth_ws.clear()
    except: breadth_ws = spreadsheet.add_worksheet(title="Market_Breadth_History", rows="1500", cols="20")
    
    breadth_out = [combined_breadth.columns.values.tolist()] + combined_breadth.values.tolist()
    breadth_ws.update(values=breadth_out, range_name='A1', value_input_option='USER_ENTERED')
    breadth_ws.format('A1:Q1', {'textFormat': {'bold': True}, "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9}})
    print(" -> Successfully appended daily breadth data.")
except Exception as e: print(f" -> Could not calculate market breadth: {e}")

# ==========================================
# 10. BUILD SYNTHETIC ETF CHARTS (GROWTH50)
# ==========================================
print("\nBuilding Synthetic Equal-Weight ETF for GROWTH50...")
try:
    ws_growth = spreadsheet.worksheet("GROWTH50")
    df_etf_stocks = pd.DataFrame(ws_growth.get_all_records())
    sym_col = next((c for c in df_etf_stocks.columns if 'symbol' in c.lower() or 'ticker' in c.lower()), None)
    if sym_col:
        etf_tickers = (df_etf_stocks[sym_col].astype(str).str.strip().str.upper() + '.NS').tolist()
        valid_tickers = [t for t in etf_tickers if t in data.columns.levels[0] and t != '^NSEI']
        
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
                fig, ax1 = plt_sub.subplots(1, 1, figsize=(4, 2.5))
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
                try: out_ws = spreadsheet.worksheet("Chart_GROWTH50"); out_ws.clear()
                except: out_ws = spreadsheet.add_worksheet(title="Chart_GROWTH50", rows="300", cols="10")
                    
                df_index = df_index.replace([np.inf, -np.inf], np.nan).fillna("")
                out_data = [df_index.columns.values.tolist()] + df_index.values.tolist()
                out_ws.update(values=out_data, range_name='A1')
except Exception as e: print(f" -> Could not build ETF chart: {e}")

print(f"\n✅ Success! Systems completely aligned.")
