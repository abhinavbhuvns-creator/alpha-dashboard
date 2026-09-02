import os
import json
import time
import urllib.request
import io
import warnings
import concurrent.futures
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

SHEET_URL = 'https://docs.google.com/spreadsheets/d/1A2fUfXGKXXQxzFnoR30cFVqtmb-28KTi4fR4N0e507g/edit?usp=sharing'
spreadsheet = gc.open_by_url(SHEET_URL)
TAB_NAME = "Avg_Rupee_Volume_Master"

# ==========================================
# 2. FETCH OFFICIAL NSE TICKER LIST (NEW IPOS)
# ==========================================
print("Fetching official NSE master stock list...")
nse_url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

try:
    req = urllib.request.Request(nse_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        csv_data = response.read().decode('utf-8')
    nse_df = pd.read_csv(io.StringIO(csv_data))
except Exception as e:
    print(f"Primary NSE URL failed ({e}). Attempting backup repository...")
    nse_df = pd.read_csv("https://raw.githubusercontent.com/anandor/nse-ticker-list/main/EQUITY_L.csv")

nse_df.columns = nse_df.columns.str.strip()
if 'SERIES' in nse_df.columns:
    nse_df = nse_df[nse_df['SERIES'].isin(['EQ', 'BE'])]

base_map = nse_df[['SYMBOL', 'NAME OF COMPANY']].copy()
base_map.columns = ['Symbol', 'Company Name']
base_map['Symbol'] = base_map['Symbol'].str.strip()
tickers = (base_map['Symbol'] + '.NS').tolist()

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

# ==========================================
# 3. DOWNLOAD 1-MONTH DATA & CALCULATE VOLUME
# ==========================================
print(f"Downloading price history for {len(tickers)} stocks to calculate 20D Rupee Volume...")
all_data = []
ticker_chunks = list(chunker(tickers, 200))

for i, chunk in enumerate(ticker_chunks):
    print(f" -> Downloading price batch {i+1} of {len(ticker_chunks)}...")
    chunk_data = yf.download(chunk, period="1mo", group_by="ticker", threads=True, progress=False)
    all_data.append(chunk_data)
    if i < len(ticker_chunks) - 1:
        time.sleep(2)

data = pd.concat(all_data, axis=1)

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
# 4. SMART RESUME SECTOR & INDUSTRY MAPPING
# ==========================================
print("\nReading existing sheet to preserve known sectors...")
existing_sectors = {}
try:
    worksheet = spreadsheet.worksheet(TAB_NAME)
    existing_data = worksheet.get_all_records()
    for row in existing_data:
        sym = str(row.get('Symbol', '')).strip()
        sec = str(row.get('Sector', 'Unclassified')).strip()
        ind = str(row.get('Industry Group', 'Unclassified')).strip()
        if sym and sec not in ['Unclassified', ''] and ind not in ['Unclassified', '']:
            existing_sectors[sym] = {'Sector': sec, 'Industry Group': ind}
    print(f" -> Found {len(existing_sectors)} stocks already classified. Skipping Yahoo fetch for these.")
except Exception:
    print(" -> No existing classification found. Proceeding with fresh fetch.")

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
    except Exception:
        return {'Symbol': symbol, 'Sector': 'Unclassified', 'Industry Group': 'Unclassified'}

symbols_list = final_df['Symbol'].tolist()
chunks = list(chunker(symbols_list, 200))
sector_data = []

print(f"Fetching sectors for remaining/new stocks in {len(chunks)} batches...")
for i, chunk in enumerate(chunks):
    print(f" -> Processing sector batch {i+1} of {len(chunks)}...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(get_sector_info, chunk)
        for res in results:
            sector_data.append(res)
    if i < len(chunks) - 1:
        time.sleep(3)

sector_df = pd.DataFrame(sector_data).drop_duplicates(subset=['Symbol'])
final_df = pd.merge(final_df, sector_df, on='Symbol', how='left')
final_df = final_df[['Symbol', 'Company Name', 'Sector', 'Industry Group', 'Avg_Rupee_Volume_Crores']]

# ==========================================
# 5. OVERWRITE MASTER GOOGLE SHEET TAB
# ==========================================
print("\nExporting updated Master Universe to Google Sheets...")
try:
    worksheet = spreadsheet.worksheet(TAB_NAME)
    worksheet.clear()
except gspread.exceptions.WorksheetNotFound:
    worksheet = spreadsheet.add_worksheet(title=TAB_NAME, rows="2500", cols="6")

sheet_output = [final_df.columns.values.tolist()] + final_df.values.tolist()
worksheet.update(values=sheet_output, range_name='A1', value_input_option='USER_ENTERED')
worksheet.freeze(rows=1)
worksheet.format('A1:E1', {
    "backgroundColor": {"red": 0.1, "green": 0.2, "blue": 0.4},
    "horizontalAlignment": "CENTER",
    "textFormat": {"foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}, "bold": True}
})

print(f"✅ Universe updated successfully with {len(final_df)} stocks (including new IPOs)!")
