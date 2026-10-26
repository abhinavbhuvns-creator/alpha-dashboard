import os
import json
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
import concurrent.futures
from datetime import datetime, timedelta
import warnings

warnings.simplefilter(action='ignore')
pd.options.mode.chained_assignment = None

# ==========================================
# 1. AUTHENTICATE GOOGLE SHEETS
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
TAB_NAME = "Fundamental_Data"

# ==========================================
# 2. GET USER STOCK UNIVERSE
# ==========================================
print("Fetching approved stock universe from Master Sheet...")
try:
    master_ws = spreadsheet.worksheet("Avg_Rupee_Volume_Master")
    master_df = pd.DataFrame(master_ws.get_all_records())
    universe_dict = {
        str(row.get('Symbol', '')).strip().upper(): {
            'Company Name': row.get('Company Name', ''),
            'Industry': row.get('Industry Group', 'Unclassified')
        } for _, row in master_df.iterrows() if str(row.get('Symbol', '')).strip()
    }
except Exception as e:
    raise SystemExit(f"🛑 Error reading Master Data: {e}")

approved_symbols = list(universe_dict.keys())

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def get_fy_quarter(date_str):
    try:
        dt = pd.to_datetime(date_str)
        if dt.month in [1, 2, 3]: return f"Q4 FY{dt.year}"
        elif dt.month in [4, 5, 6]: return f"Q1 FY{dt.year+1}"
        elif dt.month in [7, 8, 9]: return f"Q2 FY{dt.year+1}"
        else: return f"Q3 FY{dt.year+1}"
    except: return "Unknown"

def calc_yoy(c, p):
    try: return ((float(c) - float(p)) / abs(float(p))) * 100 if pd.notna(c) and pd.notna(p) and float(p) != 0 else np.nan
    except: return np.nan

def calc_cagr(c, p, yrs=3):
    try: return ((float(c) / float(p)) ** (1 / yrs) - 1) * 100 if pd.notna(c) and pd.notna(p) and float(p) > 0 and float(c) > 0 else np.nan
    except: return np.nan

# ==========================================
# 4. MULTITHREADED YAHOO FINANCE FETCH
# ==========================================
print(f"Executing Multithreaded Yahoo Finance Fetch for {len(approved_symbols)} stocks...")

def fetch_yf_data(sym):
    data = {"Stock Symbol": sym, "Industry": universe_dict[sym]['Industry'], "Company Name": universe_dict[sym]['Company Name']}
    try:
        ticker = yf.Ticker(f"{sym}.NS")
        inc_stmt = ticker.financials
        bal_sheet = ticker.balance_sheet
        
        if not inc_stmt.empty and len(inc_stmt.columns) >= 4:
            inc_stmt = inc_stmt[sorted(inc_stmt.columns)]
            data["Annual Sales 3Y Growth (%)"] = round(calc_cagr(inc_stmt.loc['Total Revenue'].iloc[-1], inc_stmt.loc['Total Revenue'].iloc[-4]), 2) if 'Total Revenue' in inc_stmt.index else "N/A"
            data["Annual PAT 3Y Growth (%)"] = round(calc_yoy(inc_stmt.loc['Net Income'].iloc[-1], inc_stmt.loc['Net Income'].iloc[-4]), 2) if 'Net Income' in inc_stmt.index else "N/A"
            data["Annual EPS 3Y Growth (%)"] = round(calc_yoy(inc_stmt.loc['Basic EPS'].iloc[-1], inc_stmt.loc['Basic EPS'].iloc[-4]), 2) if 'Basic EPS' in inc_stmt.index else "N/A"
        else:
            data.update({"Annual Sales 3Y Growth (%)": "N/A", "Annual PAT 3Y Growth (%)": "N/A", "Annual EPS 3Y Growth (%)": "N/A"})
            
        if not inc_stmt.empty and not bal_sheet.empty and len(bal_sheet.columns) >= 4:
            bal_sheet = bal_sheet[sorted(bal_sheet.columns)]
            ce_cur = bal_sheet.loc['Total Assets'].iloc[-1] - bal_sheet.loc['Current Liabilities'].iloc[-1] if 'Total Assets' in bal_sheet.index and 'Current Liabilities' in bal_sheet.index else 0
            ce_pst = bal_sheet.loc['Total Assets'].iloc[-4] - bal_sheet.loc['Current Liabilities'].iloc[-4] if 'Total Assets' in bal_sheet.index and 'Current Liabilities' in bal_sheet.index else 0
            roce_cur = (inc_stmt.loc['EBIT'].iloc[-1] / ce_cur) * 100 if ce_cur > 0 and 'EBIT' in inc_stmt.index else np.nan
            roce_pst = (inc_stmt.loc['EBIT'].iloc[-4] / ce_pst) * 100 if ce_pst > 0 and 'EBIT' in inc_stmt.index else np.nan
            data["ROCE 3Y Expansion (%)"] = round(roce_cur - roce_pst, 2) if pd.notna(roce_cur) and pd.notna(roce_pst) else "N/A"
        else:
            data["ROCE 3Y Expansion (%)"] = "N/A"
    except Exception:
        data.update({"Annual Sales 3Y Growth (%)": "N/A", "Annual PAT 3Y Growth (%)": "N/A", "Annual EPS 3Y Growth (%)": "N/A", "ROCE 3Y Expansion (%)": "N/A"})
    return data

with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    yf_results = list(executor.map(fetch_yf_data, approved_symbols))

yf_df = pd.DataFrame(yf_results)
yf_df.set_index('Stock Symbol', inplace=True)

# ==========================================
# 5. NSE BULK FINANCIALS FETCH (14 QUARTERS)
# ==========================================
print("Connecting to NSE for Bulk Financial Data (14 Quarters)...")
session = requests.Session()
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': '*/*'
}
session.get("https://www.nseindia.com", headers=headers, timeout=10)
time.sleep(2)

nse_raw_data = []
end_dt = datetime.today()

# 14 chunks of 90 days to cover roughly 3.5 years
for i in range(14):
    start_dt = end_dt - timedelta(days=90)
    start_str = start_dt.strftime('%d-%m-%Y')
    end_str = end_dt.strftime('%d-%m-%Y')
    api_url = f"https://www.nseindia.com/api/corporates-financial-results?index=equities&from_date={start_str}&to_date={end_str}"
    
    print(f" -> Fetching market-wide financials: {start_str} to {end_str}")
    success = False
    for attempt in range(3):
        try:
            res = session.get(api_url, headers=headers, timeout=15)
            if res.status_code == 200:
                data = res.json()
                if data:
                    nse_raw_data.extend(data)
                print(f"    Found {len(data)} results in chunk.")
                success = True
                break
        except Exception:
            pass
        time.sleep(2)
        
    if not success:
        print(f"    ⚠️ NSE rejected chunk or timed out.")
        
    end_dt = start_dt - timedelta(days=1)
    time.sleep(1)

# ==========================================
# 6. PROCESS BULK NSE DATA & MERGE
# ==========================================
print("Processing NSE Bulk Data & Calculating YoY Growth...")
final_master_data = []

# Process NSE Bulk Data if it worked
if nse_raw_data:
    df_nse = pd.DataFrame(nse_raw_data).rename(columns=lambda x: str(x).strip().lower())
    date_col = 'reviwperiodenddate' if 'reviwperiodenddate' in df_nse.columns else 'todate'
    
    if date_col in df_nse.columns and 'symbol' in df_nse.columns:
        df_nse['symbol'] = df_nse['symbol'].str.strip().str.upper()
        df_nse[date_col] = pd.to_datetime(df_nse[date_col], format='mixed', dayfirst=True, errors='coerce')
        df_nse = df_nse.dropna(subset=[date_col]).sort_values(by=date_col, ascending=False)
    else:
        df_nse = pd.DataFrame() # Fallback if API returned weird structure
else:
    df_nse = pd.DataFrame()

# Merge per symbol
for sym in approved_symbols:
    stock_dict = yf_df.loc[sym].to_dict() if sym in yf_df.index else {}
    stock_dict["Stock Symbol"] = sym
    
    if not df_nse.empty and sym in df_nse['symbol'].values:
        sym_df = df_nse[df_nse['symbol'] == sym].drop_duplicates(subset=[date_col]).head(14).reset_index(drop=True)
        
        for i in range(10):
            lbl = f"Q_Minus_{i}" if i > 0 else "Q_Current"
            if i < len(sym_df):
                row = sym_df.iloc[i]
                s = pd.to_numeric(row.get('totincomefromoperations', row.get('incomefromoperations')), errors='coerce') / 100
                p = pd.to_numeric(row.get('netprofitlossforperiod'), errors='coerce') / 100
                e = pd.to_numeric(row.get('dilutedepsbeforeextraordinaryitems'), errors='coerce')
                
                s_yoy, p_yoy, e_yoy = np.nan, np.nan, np.nan
                if (i + 4) < len(sym_df):
                    p_row = sym_df.iloc[i + 4]
                    ps = pd.to_numeric(p_row.get('totincomefromoperations', p_row.get('incomefromoperations')), errors='coerce') / 100
                    pp = pd.to_numeric(p_row.get('netprofitlossforperiod'), errors='coerce') / 100
                    pe = pd.to_numeric(p_row.get('dilutedepsbeforeextraordinaryitems'), errors='coerce')
                    s_yoy = calc_yoy(s, ps)
                    p_yoy = calc_yoy(p, pp)
                    e_yoy = calc_yoy(e, pe)
                    
                stock_dict.update({
                    f"Quarter ({lbl})": get_fy_quarter(row[date_col]), 
                    f"Sales Cr ({lbl})": round(s,2) if pd.notna(s) else "N/A", 
                    f"Sales YoY % ({lbl})": f"{round(s_yoy,1)}%" if pd.notna(s_yoy) else "N/A", 
                    f"PAT Cr ({lbl})": round(p,2) if pd.notna(p) else "N/A", 
                    f"PAT YoY % ({lbl})": f"{round(p_yoy,1)}%" if pd.notna(p_yoy) else "N/A", 
                    f"EPS ({lbl})": round(e,2) if pd.notna(e) else "N/A", 
                    f"EPS YoY % ({lbl})": f"{round(e_yoy,1)}%" if pd.notna(e_yoy) else "N/A"
                })
            else:
                stock_dict.update({f"Quarter ({lbl})": "N/A", f"Sales Cr ({lbl})": "N/A", f"Sales YoY % ({lbl})": "N/A", f"PAT Cr ({lbl})": "N/A", f"PAT YoY % ({lbl})": "N/A", f"EPS ({lbl})": "N/A", f"EPS YoY % ({lbl})": "N/A"})
    else:
        # No NSE data found for this symbol
        for i in range(10):
            lbl = f"Q_Minus_{i}" if i > 0 else "Q_Current"
            stock_dict.update({f"Quarter ({lbl})": "N/A", f"Sales Cr ({lbl})": "N/A", f"Sales YoY % ({lbl})": "N/A", f"PAT Cr ({lbl})": "N/A", f"PAT YoY % ({lbl})": "N/A", f"EPS ({lbl})": "N/A", f"EPS YoY % ({lbl})": "N/A"})

    final_master_data.append(stock_dict)

# ==========================================
# 7. PUSH TO GOOGLE SHEETS
# ==========================================
print("Pushing formatted data to Google Sheets...")
df_final = pd.DataFrame(final_master_data).fillna("N/A")

# Reorder columns to ensure Stock Symbol and Industry are first
cols = df_final.columns.tolist()
cols.insert(0, cols.pop(cols.index('Stock Symbol')))
if 'Industry' in cols: cols.insert(1, cols.pop(cols.index('Industry')))
if 'Company Name' in cols: cols.insert(2, cols.pop(cols.index('Company Name')))
df_final = df_final[cols]

try:
    worksheet = spreadsheet.worksheet(TAB_NAME)
    worksheet.clear()
except gspread.exceptions.WorksheetNotFound:
    worksheet = spreadsheet.add_worksheet(title=TAB_NAME, rows="2500", cols="80")

sheet_output = [df_final.columns.values.tolist()] + df_final.values.tolist()
worksheet.update(values=sheet_output, range_name='A1', value_input_option='USER_ENTERED')
worksheet.freeze(rows=1, cols=3)
worksheet.format('A1:BZ1', {
    "backgroundColor": {"red": 0.1, "green": 0.2, "blue": 0.4},
    "textFormat": {"foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}, "bold": True},
    "horizontalAlignment": "CENTER"
})

print(f"✅ Fundamental Data fully updated for {len(final_master_data)} stocks!")
