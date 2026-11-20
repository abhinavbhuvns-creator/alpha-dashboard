import os
import json
import time
import warnings
from datetime import datetime
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
# 2. READ UNIFIED SHEET (> 5 CRORE FILTER FOR INDUSTRY RANKING)
# ==========================================
SINGLE_SHEET_URL = 'https://docs.google.com/spreadsheets/d/1A2fUfXGKXXQxzFnoR30cFVqtmb-28KTi4fR4N0e507g/edit?usp=sharing'
spreadsheet = gc.open_by_url(SINGLE_SHEET_URL)

print("Opening Master spreadsheet and filtering for > 5 Crore Average Rupee Volume...", flush=True)
try:
    master_ws = spreadsheet.worksheet('Avg_Rupee_Volume_Master')
    master_data = master_ws.get_all_records()
    df_master = pd.DataFrame(master_data)
except Exception as e:
    raise SystemExit(f"Error opening sheet. Ensure you have edit access. {e}")

unique_tickers = []
industry_map = {}
stock_to_ind = {}
stock_to_vol = {}

for _, row in df_master.iterrows():
    symbol = str(row.get('Symbol', '')).replace('.NS', '').strip().upper()
    ind = str(row.get('Industry Group', '')).strip()
    try:
        vol_cr = float(str(row.get('Avg_Rupee_Volume_Crores', 0)).replace(',', '').strip())
    except Exception:
        vol_cr = 0.0

    # Strict > 5 Crore Average Rupee Volume Filter
    if symbol and ind and ind != 'Unclassified' and vol_cr >= 5.0:
        t_ns = symbol + '.NS'
        unique_tickers.append(t_ns)
        stock_to_ind[t_ns] = ind
        stock_to_vol[t_ns] = vol_cr
        if ind not in industry_map:
            industry_map[ind] = []
        industry_map[ind].append(t_ns)

unique_tickers = list(set(unique_tickers))
print(f"Found {len(unique_tickers)} liquid NSE tickers (> 5 Cr Rupee Volume) across {len(industry_map)} Industry Groups.", flush=True)

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

# ==========================================
# 3. FETCH 1 YEAR OF DAILY DATA
# ==========================================
print("Fetching 1 Year of daily data to calculate Returns and IPO/52W Highs...", flush=True)
market_data_daily = {}
chunk_size = 200
ticker_chunks = list(chunker(unique_tickers, chunk_size))
for i, chunk in enumerate(ticker_chunks):
    print(f"  -> Downloading Daily Batch {i+1} of {len(ticker_chunks)}...", flush=True)
    df = yf.download(chunk, period="1y", group_by="ticker", progress=False)
    if len(chunk) == 1:
        market_data_daily[chunk[0]] = df
    else:
        for t in chunk:
            try:
                market_data_daily[t] = df[t]
            except Exception:
                pass
    time.sleep(1)

# ==========================================
# 4. CALCULATE DAILY METRICS (IPO-SAFE)
# ==========================================
stock_base_metrics = {}
breakout_candidates = []
print("\nCalculating metrics and finding prime 30m breakout candidates...", flush=True)

for ticker in unique_tickers:
    try:
        ticker_data = market_data_daily.get(ticker)
        if ticker_data is None or ticker_data['Close'].dropna().empty:
            continue
        closes = ticker_data['Close'].dropna()
        highs = ticker_data['High'].dropna()
        if len(closes) < 6:
            continue

        current_price = closes.iloc[-1]
        ema_21 = closes.ewm(span=21, adjust=False).mean()
        macro_high = highs.max()  # Automatically uses ATH for recent IPOs < 252D
        pct_dist_high = (current_price - macro_high) / macro_high if pd.notna(macro_high) and macro_high > 0 else np.nan
        historical_daily = ticker_data.iloc[:-5]
        base_high = historical_daily['High'].dropna().max() if not historical_daily.empty else macro_high

        prev_2m_val = (closes.iloc[-22] / closes.iloc[-64] - 1) if len(closes) >= 64 else np.nan
        stock_base_metrics[ticker] = {
            '1d': (closes.iloc[-1] / closes.iloc[-2] - 1) if len(closes) >= 2 else np.nan,
            '1w': (closes.iloc[-1] / closes.iloc[-6] - 1) if len(closes) >= 6 else np.nan,
            '1m': (closes.iloc[-1] / closes.iloc[-22] - 1) if len(closes) >= 22 else np.nan,
            'prev_2m': prev_2m_val,
            '3m': (closes.iloc[-1] / closes.iloc[-64] - 1) if len(closes) >= 64 else np.nan,
            '6m': (closes.iloc[-1] / closes.iloc[-127] - 1) if len(closes) >= 127 else np.nan,
            'pct_dist_ema': (current_price - ema_21.iloc[-1]) / ema_21.iloc[-1],
            'macro_high': macro_high,
            'pct_dist_high': pct_dist_high,
            'base_high': base_high,
            'crosses_1d': 0,
            'crosses_1w': 0
        }
        if pd.notna(macro_high) and current_price >= (macro_high * 0.95):
            breakout_candidates.append(ticker)
    except Exception:
        pass

# ==========================================
# 5. INTRADAY 30M BREAKOUT CROSSOVERS
# ==========================================
market_data_30m = {}
if len(breakout_candidates) > 0:
    candidate_chunks = list(chunker(breakout_candidates, 200))
    all_30m_chunks = []
    for i, chunk in enumerate(candidate_chunks):
        df_30m_chunk = yf.download(chunk, period="5d", interval="30m", group_by="ticker", progress=False)
        all_30m_chunks.append(df_30m_chunk)
        if i < len(candidate_chunks) - 1:
            time.sleep(1)
    if all_30m_chunks:
        df_30m = pd.concat(all_30m_chunks, axis=1)
        if len(breakout_candidates) == 1:
            market_data_30m[breakout_candidates[0]] = df_30m
        else:
            for t in breakout_candidates:
                try:
                    if t in df_30m.columns.levels[0]:
                        market_data_30m[t] = df_30m[t]
                except Exception:
                    pass

for ticker in breakout_candidates:
    try:
        ticker_30m = market_data_30m.get(ticker)
        if ticker_30m is None or ticker_30m['Close'].dropna().empty:
            continue
        df_30m = ticker_30m.dropna(subset=['Close', 'High']).copy()
        metrics = stock_base_metrics[ticker]
        prev_highs = df_30m['High'].shift(1).copy()
        if pd.notna(metrics['base_high']) and len(prev_highs) > 0:
            prev_highs.iloc[0] = metrics['base_high']
        df_30m['Resistance_Line'] = prev_highs.cummax().fillna(metrics['base_high'])
        prev_closes = df_30m['Close'].shift(1).copy()
        daily_data = market_data_daily.get(ticker)
        if daily_data is not None:
            hist_daily = daily_data.iloc[:-5]
            if not hist_daily.empty and len(prev_closes) > 0:
                prev_closes.iloc[0] = hist_daily['Close'].dropna().iloc[-1]
        crossovers = (prev_closes < df_30m['Resistance_Line']) & (df_30m['Close'] > df_30m['Resistance_Line'])
        stock_base_metrics[ticker]['crosses_1w'] = int(crossovers.sum())
        last_date = df_30m.index[-1].date()
        stock_base_metrics[ticker]['crosses_1d'] = int(crossovers[df_30m.index.date == last_date].sum())
    except Exception:
        pass

df_stocks = pd.DataFrame.from_dict(stock_base_metrics, orient='index')

# ==========================================
# 6. MODERN INDUSTRY ETF RETURNS & INDUSTRY RANKS
#    (Bayesian Shrinkage k=4 + Smoothed Breadth + Flat-Base Emerging Score)
# ==========================================
print("\nRolling up >5 Cr stocks into ETF Groups, Bayesian Industry Ranks, and Top 2 Weekly/Monthly Leaders...", flush=True)
etf_results = []
rank_rows = []

mkt = {k: (df_stocks[k].mean() if not df_stocks[k].dropna().empty else 0.0) for k in ['1w', '1m', '3m', '6m', 'prev_2m']}

for industry, stocks in industry_map.items():
    valid_group_tickers = [t for t in stocks if t in df_stocks.index]
    if not valid_group_tickers:
        continue
    ind_data = df_stocks.loc[valid_group_tickers]
    sorted_1w = ind_data['1w'].dropna().sort_values(ascending=False)
    sorted_1m = ind_data['1m'].dropna().sort_values(ascending=False)

    top1_1w = f"{sorted_1w.index[0].replace('.NS', '')} ({sorted_1w.iloc[0]*100:+.1f}%)" if len(sorted_1w) > 0 else ""
    top2_1w = f"{sorted_1w.index[1].replace('.NS', '')} ({sorted_1w.iloc[1]*100:+.1f}%)" if len(sorted_1w) > 1 else ""
    top1_1m = f"{sorted_1m.index[0].replace('.NS', '')} ({sorted_1m.iloc[0]*100:+.1f}%)" if len(sorted_1m) > 0 else ""
    top2_1m = f"{sorted_1m.index[1].replace('.NS', '')} ({sorted_1m.iloc[1]*100:+.1f}%)" if len(sorted_1m) > 1 else ""

    # Bayesian Shrinkage (k=4) + 50/50 Mean-Median Blend
    n = len(valid_group_tickers)
    w = n / (n + 4.0)
    shrink = lambda col: w * (0.5 * ind_data[col].mean() + 0.5 * ind_data[col].median()) + (1.0 - w) * mkt[col] if not ind_data[col].dropna().empty else mkt[col]

    ret_1w_grp, ret_1m_grp, ret_3m_grp, ret_6m_grp, prev_2m_grp = shrink('1w'), shrink('1m'), shrink('3m'), shrink('6m'), shrink('prev_2m')

    # Laplace-Smoothed Breadth (prevents 100%/0% distortion on 1-2 stock groups)
    pct_pos_1m = ((ind_data['1m'] > 0).sum() + 1.5) / (n + 3.0)

    # Composite & Flat-Base Emerging Scores
    composite_score = (ret_1w_grp * 0.25) + (ret_1m_grp * 0.35) + (ret_3m_grp * 0.25) + (ret_6m_grp * 0.15) + (pct_pos_1m * 0.02)
    emerging_score = (ret_1m_grp * 0.50) + (ret_1w_grp * 0.30) - (abs(prev_2m_grp) * 0.20)

    etf_results.append({
        'Industry Name': industry,
        'Stocks (>5Cr)': n,
        '1 Day Return': ind_data['1d'].mean(),
        '1 Week Return': ind_data['1w'].mean(),
        '1 Month Return': ind_data['1m'].mean(),
        'Prev 2M Return': ind_data['prev_2m'].mean(),
        '3 Month Return': ind_data['3m'].mean(),
        '6 Month Return': ind_data['6m'].mean(),
        '% Distance from 21 EMA': ind_data['pct_dist_ema'].mean(),
        '% Distance from 52W/IPO High': ind_data['pct_dist_high'].mean(),
        'Sector 30m 52W Crosses (1D)': int(ind_data['crosses_1d'].sum()),
        'Sector 30m 52W Crosses (1W)': int(ind_data['crosses_1w'].sum()),
        'Top Leader (1W)': top1_1w,
        '2nd Best (1W)': top2_1w,
        'Top Leader (1M)': top1_1m,
        '2nd Best (1M)': top2_1m
    })

    rank_rows.append({
        'Industry Name': industry,
        'Stocks (>5Cr)': n,
        '_score': composite_score,
        '_emerging': emerging_score,
        '_1w': ret_1w_grp,
        '_1m': ret_1m_grp,
        '_3m': ret_3m_grp,
        '_6m': ret_6m_grp,
        'Top Leader (1W)': top1_1w,
        '2nd Best (1W)': top2_1w,
        'Top Leader (1M)': top1_1m,
        '2nd Best (1M)': top2_1m
    })

df_ranks = pd.DataFrame(rank_rows)
if not df_ranks.empty:
    df_ranks['Master Trend Rank'] = df_ranks['_score'].rank(ascending=False, method='min').astype(int)
    df_ranks['Emerging Sector Rank'] = df_ranks['_emerging'].rank(ascending=False, method='min').astype(int)
    df_ranks['1 Week Rank'] = df_ranks['_1w'].rank(ascending=False, method='min').astype(int)
    df_ranks['1 Month Rank'] = df_ranks['_1m'].rank(ascending=False, method='min').astype(int)
    df_ranks['3 Month Rank'] = df_ranks['_3m'].rank(ascending=False, method='min').astype(int)
    df_ranks['6 Month Rank'] = df_ranks['_6m'].rank(ascending=False, method='min').astype(int)
    df_ranks = df_ranks.sort_values('Master Trend Rank', ascending=True)
    rank_cols_order = [
        'Industry Name', 'Stocks (>5Cr)', 'Master Trend Rank', 'Emerging Sector Rank',
        '1 Week Rank', '1 Month Rank', '3 Month Rank', '6 Month Rank',
        'Top Leader (1W)', '2nd Best (1W)', 'Top Leader (1M)', '2nd Best (1M)'
    ]
    df_ranks_out = df_ranks[rank_cols_order].fillna('')

individual_stocks_db = []
for ticker in df_stocks.index:
    stock_row = df_stocks.loc[ticker]
    individual_stocks_db.append({
        'Ticker': ticker.replace('.NS', ''),
        'Industry Name': stock_to_ind.get(ticker, 'Unclassified'),
        '1 Day Return': stock_row['1d'],
        '1 Week Return': stock_row['1w'],
        '1 Month Return': stock_row['1m'],
        'Prev 2M Squeeze': stock_row['prev_2m'],
        '3 Month Return': stock_row['3m'],
        '6 Month Return': stock_row['6m'],
        '% Distance from 21 EMA': stock_row['pct_dist_ema'],
        '52W / IPO High': stock_row['macro_high'],
        '% Dist from 52W/IPO High': stock_row['pct_dist_high'],
        '30m 52W Crosses (1D)': stock_row['crosses_1d'],
        '30m 52W Crosses (1W)': stock_row['crosses_1w'],
        'Avg Rupee Vol (Cr)': stock_to_vol.get(ticker, 0.0)
    })

df_etf = pd.DataFrame(etf_results).replace([np.inf, -np.inf], np.nan).fillna('')
df_all_stocks = pd.DataFrame(individual_stocks_db).replace([np.inf, -np.inf], np.nan).fillna('')

# ==========================================
# 7. WRITE TO GOOGLE SHEETS (ETF RETURNS, STOCK DB, & INDUSTRY RANKS)
# ==========================================
print("Writing ETF Returns, Stock Database, and Industry Ranks to Google Sheet...", flush=True)
etf_tab_name = 'Industry ETF Returns'
try:
    etf_sheet = spreadsheet.worksheet(etf_tab_name)
    etf_sheet.clear()
except gspread.exceptions.WorksheetNotFound:
    etf_sheet = spreadsheet.add_worksheet(title=etf_tab_name, rows="200", cols="20")
etf_data_out = [df_etf.columns.values.tolist()] + df_etf.values.tolist()
etf_sheet.update(values=etf_data_out, range_name='A1', value_input_option='USER_ENTERED')
etf_sheet.format('A1:P1', {'textFormat': {'bold': True}})
etf_sheet.format('C2:J200', {"numberFormat": {"type": "PERCENT", "pattern": "0.00%"}})

db_tab_name = 'Stock Database'
try:
    db_sheet = spreadsheet.worksheet(db_tab_name)
    db_sheet.clear()
except gspread.exceptions.WorksheetNotFound:
    db_sheet = spreadsheet.add_worksheet(title=db_tab_name, rows="2000", cols="16")
db_data_out = [df_all_stocks.columns.values.tolist()] + df_all_stocks.values.tolist()
db_sheet.update(values=db_data_out, range_name='A1', value_input_option='USER_ENTERED')
db_sheet.format('A1:N1', {'textFormat': {'bold': True}})
db_sheet.format('C2:I2000', {"numberFormat": {"type": "PERCENT", "pattern": "0.00%"}})
db_sheet.format('K2:K2000', {"numberFormat": {"type": "PERCENT", "pattern": "0.00%"}})

if not df_ranks.empty:
    rank_tab_name = 'Industry Ranks'
    try:
        rank_sheet = spreadsheet.worksheet(rank_tab_name)
        rank_sheet.clear()
    except gspread.exceptions.WorksheetNotFound:
        rank_sheet = spreadsheet.add_worksheet(title=rank_tab_name, rows="200", cols="15")
    rank_data_out = [df_ranks_out.columns.values.tolist()] + df_ranks_out.values.tolist()
    rank_sheet.update(values=rank_data_out, range_name='A1', value_input_option='USER_ENTERED')
    rank_sheet.freeze(rows=1)
    rank_sheet.format('A1:L1', {'textFormat': {'bold': True, 'foregroundColor': {'red': 1, 'green': 1, 'blue': 1}}, 'backgroundColor': {'red': 0.1, 'green': 0.45, 'blue': 0.91}})

print("✅ Success! ETF Returns, Stock Database (>5Cr), and Bayesian Industry Ranks fully updated.", flush=True)
