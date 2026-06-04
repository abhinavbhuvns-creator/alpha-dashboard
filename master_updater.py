import os
import json
import time
import warnings
import concurrent.futures
import urllib.request
import io
import pandas as pd
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials

# Suppress pandas fragmentation warnings
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

# Use the MASTER SHEET URL
SHEET_URL = 'https://docs.google.com/spreadsheets/d/1Z9TgE-znOIPoh1dlrTG5tBHFOvgYij_0EiGYWMbPGzE/edit?usp=sharing'
spreadsheet = gc.open_by_url(SHEET_URL)
TAB_NAME = "Avg_Rupee_Volume_Master"

# ==========================================
# 2. FETCH OFFICIAL TICKER LIST
# ==========================================
print("Fetching official NSE master stock list...")
nse_url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

try:
    req = urllib.request.Request(nse_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        csv_data = response.read().decode('utf-8')
    nse_df = pd.read_csv(io.StringIO(csv_data))
except Exception as e:
    print("Primary NSE URL failed. Attempting backup...")
    nse_df = pd.read_csv("https://raw.githubusercontent.com/anandor/nse-ticker-list/main/EQUITY_L.csv")

nse_df.columns = nse_df.columns.str.strip()
if 'SERIES' in nse_df.columns:
    nse_df = nse_df[nse_df['SERIES'].isin(['EQ', 'BE'])]

base_map = nse_df[['SYMBOL', 'NAME OF COMPANY']].copy()
base_map.columns = ['Symbol', 'Company Name']
tickers = (base_map['Symbol'].str.strip() + '.NS').tolist()

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

# ==========================================
# 3. VECTORIZED VOLUME CALCULATIONS (BATCHED)
# ==========================================
print(f"Downloading 1 month of price history for {len(tickers)} stocks in safe batches...")
all_data = []

ticker_chunks = list(chunker(tickers, 200))
for i, chunk in enumerate(ticker_chunks):
    print(f" -> Downloading price batch {i+1} of {len(ticker_chunks)}...")
    chunk_data = yf.download(chunk, period="1mo", group_by="ticker", threads=True, progress=False)
    all_data.append(chunk_data)
    if i < len(ticker_chunks) - 1:
        time.sleep(3)

data = pd.concat(all_data, axis=1)

print("Calculating 20-day Average Rupee Volume (in Crores)...")
close_df = pd.DataFrame({t: data[t]['Close'] for t in tickers if t in data.columns.levels[0]})
vol_df = pd.DataFrame({t: data[t]['Volume'] for t in tickers if t in data.columns.levels[0]})

rupee_vol_matrix = close_df * vol_df
avg_rupee_vol_20 = rupee_vol_matrix.tail(20).mean()

vol_df_clean = avg_rupee_vol_20.reset_index()
vol_df_clean.columns = ['Symbol', 'Avg_Rupee_Volume']
vol_df_clean['Symbol'] = vol_df_clean['Symbol'].str.replace('.NS', '', regex=False)

final_df = pd.merge(base_map, vol_df_clean, on='Symbol', how='inner')

final_df['Avg_Rupee_Volume_Crores'] = (final_df['Avg_Rupee_Volume'] / 10000000).round(2)
final_df = final_df.drop(columns=['Avg_Rupee_Volume']).dropna()
final_df = final_df.sort_values(by='Avg_Rupee_Volume_Crores', ascending=False)

# ==========================================
# 4. BATCHED MULTITHREADING WITH SMART RESUME
# ==========================================
print("\nChecking Google Sheet for already classified stocks...")
existing_sectors = {}
try:
    worksheet = spreadsheet.worksheet(TAB_NAME)
    existing_data = worksheet.get_all_records()
    for row in existing_data:
        sym = row.get('Symbol', '')
        sec = row.get('Sector', 'Unclassified')
        ind = row.get('Industry Group', 'Unclassified')

        if sym and sec != 'Unclassified' and sec != '':
            existing_sectors[sym] = {'Sector': sec, 'Industry Group': ind}

    print(f" -> Found {len(existing_sectors)} stocks already classified. Bypassing Yahoo for these...")
except Exception:
    print(" -> No existing data found in sheet. Starting fresh...")

def get_sector_info(symbol):
    if symbol in existing_sectors:
        return {
            'Symbol': symbol,
            'Sector': existing_sectors[symbol]['Sector'],
            'Industry Group': existing_sectors[symbol]['Industry Group']
        }
    try:
        info = yf.Ticker(symbol + ".NS").info
        return {
            'Symbol': symbol,
            'Sector': info.get('sector', 'Unclassified'),
            'Industry Group': info.get('industry', 'Unclassified')
        }
    except:
        return {'Symbol': symbol, 'Sector': 'Unclassified', 'Industry Group': 'Unclassified'}

symbols_list = final_df['Symbol'].tolist()
chunk_size = 250
chunks = list(chunker(symbols_list, chunk_size))
sector_data = []

print(f"\nFetching Sectors for {len(symbols_list)} stocks in {len(chunks)} batches.")
print("This will take a few minutes...")

for i, chunk in enumerate(chunks):
    print(f" -> Processing sector batch {i+1} of {len(chunks)}...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(get_sector_info, chunk)
        for res in results:
            sector_data.append(res)

    if i < len(chunks) - 1:
        time.sleep(5)

sector_df = pd.DataFrame(sector_data)
final_df = pd.merge(final_df, sector_df, on='Symbol', how='left')
final_df = final_df[['Symbol', 'Company Name', 'Sector', 'Industry Group', 'Avg_Rupee_Volume_Crores']]

# ==========================================
# 5. WRITE RESULTS TO GOOGLE SHEETS
# ==========================================
print("\nExporting sorted metrics directly to your Google Sheet...")

try:
    worksheet = spreadsheet.worksheet(TAB_NAME)
    worksheet.clear()
except gspread.exceptions.WorksheetNotFound:
    worksheet = spreadsheet.add_worksheet(title=TAB_NAME, rows="2500", cols="5")

sheet_output = [final_df.columns.values.tolist()] + final_df.values.tolist()
worksheet.update(values=sheet_output, range_name='A1', value_input_option='USER_ENTERED')

print(f"\nMaster Sheet Update Complete! Successfully mapped industries for {len(final_df)} records.")
