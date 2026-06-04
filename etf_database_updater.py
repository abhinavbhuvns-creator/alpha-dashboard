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
# UNIFIED URL: Connects to the single dashboard sheet
SINGLE_SHEET_URL = 'https://docs.google.com/spreadsheets/d/1A2fUfXGKXXQxzFnoR30cFVqtmb-28KTi4fR4N0e507g/edit?usp=sharing'
spreadsheet = gc.open_by_url(SINGLE_SHEET_URL)

print("Opening Master spreadsheet and reading Industry Groups...")
try:
    master_ws = spreadsheet.worksheet('Avg_Rupee_Volume_Master')
    master_data = master_ws.get_all_records()
    df_master = pd.DataFrame(master_data)
except Exception as e:
    raise SystemExit(f"Error opening sheet. Ensure you have edit access. {e}")

# 3. Extract unique tickers and build an Industry Map directly from the Master Sheet
unique_tickers = []
industry_map = {} 
stock_to_ind = {} 

for _, row in df_master.iterrows():
    symbol = str(row.get('Symbol', '')).strip().upper()
    ind = str(row.get('Industry Group', '')).strip()

    if symbol and ind and ind != 'Unclassified':
        t_ns = symbol + '.NS' if not symbol.endswith('.NS') else symbol
        unique_tickers.append(t_ns)
        stock_to_ind[t_ns] = ind

        if ind not in industry_map:
            industry_map[ind] = []
        industry_map[ind].append(t_ns)

unique_tickers = list(set(unique_tickers))
print(f"Found {len(unique_tickers)} valid NSE tickers with assigned Industry Groups.")

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

# 4. CHUNKING: Fetch 1 Year of Daily Data safely
print("Fetching 1 Year of daily data to calculate Returns and IPO/52W Highs...")
market_data_daily = {}
chunk_size = 200
ticker_chunks = list(chunker(unique_tickers, chunk_size))

for i, chunk in enumerate(ticker_chunks):
    print(f"  -> Downloading Daily Batch {i+1} of {len(ticker_chunks)}...")
    df = yf.download(chunk, period="1y", group_by="ticker", progress=False)
    if len(chunk) == 1:
        market_data_daily[chunk[0]] = df
    else:
        for t in chunk:
            try: market_data_daily[t] = df[t]
            except: pass
    time.sleep(1)

# 5. SMART FILTER: Calculate daily metrics & isolate true breakout candidates
stock_base_metrics = {}
breakout_candidates = []

print("\nCalculating metrics and finding prime 30m breakout candidates...")
for ticker in unique_tickers:
    try:
        ticker_data = market_data_daily.get(ticker)
        if ticker_data is None or ticker_data['Close'].dropna().empty: continue

        closes = ticker_data['Close'].dropna()
        highs = ticker_data['High'].dropna()
        if len(closes) < 2: continue

        current_price = closes.iloc[-1]
        ema_21 = closes.ewm(span=21, adjust=False).mean()

        macro_high = highs.max()
        pct_dist_high = (current_price - macro_high) / macro_high if pd.notna(macro_high) else np.nan

        historical_daily = ticker_data.iloc[:-5]
        base_high = historical_daily['High'].dropna().max() if not historical_daily.empty else macro_high

        # === SQUEEZE METRIC: prev_2m ===
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
    except:
        pass

print(f"Filtered down to {len(breakout_candidates)} prime stocks within 5% of their Highs.")

# 6. INTRADAY FETCH: Pull 30m data ONLY for the breakout candidates
market_data_30m = {}
if len(breakout_candidates) > 0:
    print(f"Fetching 5 days of 30m Intraday Data for {len(breakout_candidates)} candidates in batches...")
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
                except: pass

# 7. Calculate 30m Rolling Crossovers
for ticker in breakout_candidates:
    try:
        ticker_30m = market_data_30m.get(ticker)
        if ticker_30m is None or ticker_30m['Close'].dropna().empty: continue

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
    except:
        pass

df_stocks = pd.DataFrame.from_dict(stock_base_metrics, orient='index')

# 8A. Calculate Industry ETF Returns
print("\nRolling up data into ETF Groups and Stock Database...")
etf_results = []
for industry, stocks in industry_map.items():
    valid_group_tickers = [t for t in stocks if t in df_stocks.index]

    if not valid_group_tickers: continue

    ind_data = df_stocks.loc[valid_group_tickers]
    sorted_1w = ind_data['1w'].sort_values(ascending=False).dropna()
    sorted_1m = ind_data['1m'].sort_values(ascending=False).dropna()

    etf_results.append({
        'Industry Name': industry,
        '1 Day Return': ind_data['1d'].mean(),
        '1 Week Return': ind_data['1w'].mean(),
        '1 Month Return': ind_data['1m'].mean(),
        'Prev 2M Return': ind_data['prev_2m'].mean(),
        '3 Month Return': ind_data['3m'].mean(),
        '6 Month Return': ind_data['6m'].mean(),
        '% Distance from 21 EMA': ind_data['pct_dist_ema'].mean(),
        '% Distance from 52W/IPO High': ind_data['pct_dist_high'].mean(),
        'Sector 30m 52W Crosses (1D)': ind_data['crosses_1d'].sum(),
        'Sector 30m 52W Crosses (1W)': ind_data['crosses_1w'].sum(),
        'Top Leader (1W)': sorted_1w.index[0].replace('.NS', '') if len(sorted_1w) > 0 else '',
        '2nd Best (1W)': sorted_1w.index[1].replace('.NS', '') if len(sorted_1w) > 1 else '',
        'Top Leader (1M)': sorted_1m.index[0].replace('.NS', '') if len(sorted_1m) > 0 else '',
        '2nd Best (1M)': sorted_1m.index[1].replace('.NS', '') if len(sorted_1m) > 1 else ''
    })

# 8B. Prepare Stock Database
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
        '30m 52W Crosses (1W)': stock_row['crosses_1w']
    })

df_etf = pd.DataFrame(etf_results).fillna('')
df_all_stocks = pd.DataFrame(individual_stocks_db).fillna('')

# 9. Write back to Target Google Sheet
print("Writing data back to your unified Dashboard Google Sheet...")

etf_tab_name = 'Industry ETF Returns'
try:
    etf_sheet = spreadsheet.worksheet(etf_tab_name)
    etf_sheet.clear()
except gspread.exceptions.WorksheetNotFound:
    etf_sheet = spreadsheet.add_worksheet(title=etf_tab_name, rows="150", cols="20")

etf_data_out = [df_etf.columns.values.tolist()] + df_etf.values.tolist()
etf_sheet.update(values=etf_data_out, range_name='A1', value_input_option='USER_ENTERED')
etf_sheet.format('A1:O1', {'textFormat': {'bold': True}})
etf_sheet.format('B2:I150', {"numberFormat": {"type": "PERCENT", "pattern": "0.00%"}})

db_tab_name = 'Stock Database'
try:
    db_sheet = spreadsheet.worksheet(db_tab_name)
    db_sheet.clear()
except gspread.exceptions.WorksheetNotFound:
    db_sheet = spreadsheet.add_worksheet(title=db_tab_name, rows="2000", cols="15")

db_data_out = [df_all_stocks.columns.values.tolist()] + df_all_stocks.values.tolist()
db_sheet.update(values=db_data_out, range_name='A1', value_input_option='USER_ENTERED')
db_sheet.format('A1:M1', {'textFormat': {'bold': True}})
db_sheet.format('C2:I2000', {"numberFormat": {"type": "PERCENT", "pattern": "0.00%"}})
db_sheet.format('K2:K2000', {"numberFormat": {"type": "PERCENT", "pattern": "0.00%"}})

print("Success! ETF Returns and Stock Database fully updated.")
