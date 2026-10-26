import os
import json
import time
import requests
import urllib.parse
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
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
# 2. MASTER UNIVERSE & SMART RESUME LOGIC
# ==========================================
print("Fetching approved stock universe...")
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

try:
    worksheet = spreadsheet.worksheet(TAB_NAME)
    existing_data = worksheet.get_all_records()
    completed_symbols = [str(row.get('Stock Symbol', '')).strip().upper() for row in existing_data]
    sheet_headers = worksheet.row_values(1)
    is_empty = False
except gspread.exceptions.WorksheetNotFound:
    worksheet = spreadsheet.add_worksheet(title=TAB_NAME, rows="2500", cols="80")
    completed_symbols = []
    sheet_headers = []
    is_empty = True

symbols_to_process = [sym for sym in approved_symbols if sym not in completed_symbols]

print(f"Total Universe: {len(approved_symbols)} | Completed: {len(completed_symbols)} | Remaining: {len(symbols_to_process)}")

if not symbols_to_process:
    print("✅ All stocks fully processed!")
    raise SystemExit(0)

BATCH_LIMIT = 20
symbols_to_process = symbols_to_process[:BATCH_LIMIT]
print(f"Processing next batch of {len(symbols_to_process)} stocks...")

# ==========================================
# 3. HELPER FUNCTIONS & FALLBACK LOGIC
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

def get_metric(row, keys):
    # Hunts through the dictionary of possible NSE column names and STRIPS COMMAS
    for k in keys:
        if k in row and pd.notna(row[k]) and str(row[k]).strip() not in ["", "-"]:
            clean_val = str(row[k]).replace(',', '').strip()
            return pd.to_numeric(clean_val, errors='coerce')
    return np.nan

# Define the exact variations the NSE uses across different sectors
SALES_KEYS = ['totincomefromoperations', 'incomefromoperations', 'totalincome', 'interestearned', 'revenuefromoperations']
PAT_KEYS = ['netprofitlossforperiod', 'netprofitlossforperiodfromcontinuingoperations', 'profitlossaftertax']
EPS_KEYS = ['dilutedepsbeforeextraordinaryitems', 'dilutedepsafterextraordinaryitems', 'dilutedepsbeforeextraordinaryitemsnotannualised', 'dilutedeps']

# ==========================================
# 4. INITIALIZE SESSION
# ==========================================
session = requests.Session()
req_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)', 'Accept': '*/*'}
session.get("https://www.nseindia.com", headers=req_headers, timeout=10)
time.sleep(2)

batch_data = []

# ==========================================
# 5. SYMBOL-BY-SYMBOL FETCH
# ==========================================
for idx, sym in enumerate(symbols_to_process):
    print(f"Fetching {sym} ({idx+1}/{len(symbols_to_process)})...")
    stock_data = {"Stock Symbol": sym, "Industry": universe_dict[sym]['Industry'], "Company Name": universe_dict[sym]['Company Name']}

    # --- YAHOO FINANCE (Annual) ---
    try:
        ticker = yf.Ticker(f"{sym}.NS")
        inc_stmt = ticker.financials
        bal_sheet = ticker.balance_sheet
        
        if not inc_stmt.empty and len(inc_stmt.columns) >= 4:
            inc_stmt = inc_stmt[sorted(inc_stmt.columns)]
            stock_data["Annual Sales 3Y Growth (%)"] = round(calc_cagr(inc_stmt.loc['Total Revenue'].iloc[-1], inc_stmt.loc['Total Revenue'].iloc[-4]), 2) if 'Total Revenue' in inc_stmt.index else "N/A"
            stock_data["Annual PAT 3Y Growth (%)"] = round(calc_yoy(inc_stmt.loc['Net Income'].iloc[-1], inc_stmt.loc['Net Income'].iloc[-4]), 2) if 'Net Income' in inc_stmt.index else "N/A"
            stock_data["Annual EPS 3Y Growth (%)"] = round(calc_yoy(inc_stmt.loc['Basic EPS'].iloc[-1], inc_stmt.loc['Basic EPS'].iloc[-4]), 2) if 'Basic EPS' in inc_stmt.index else "N/A"
        else:
            stock_data.update({"Annual Sales 3Y Growth (%)": "N/A", "Annual PAT 3Y Growth (%)": "N/A", "Annual EPS 3Y Growth (%)": "N/A"})
            
        if not inc_stmt.empty and not bal_sheet.empty and len(bal_sheet.columns) >= 4:
            bal_sheet = bal_sheet[sorted(bal_sheet.columns)]
            ce_cur = bal_sheet.loc['Total Assets'].iloc[-1] - bal_sheet.loc['Current Liabilities'].iloc[-1] if 'Total Assets' in bal_sheet.index and 'Current Liabilities' in bal_sheet.index else 0
            ce_pst = bal_sheet.loc['Total Assets'].iloc[-4] - bal_sheet.loc['Current Liabilities'].iloc[-4] if 'Total Assets' in bal_sheet.index and 'Current Liabilities' in bal_sheet.index else 0
            roce_cur = (inc_stmt.loc['EBIT'].iloc[-1] / ce_cur) * 100 if ce_cur > 0 and 'EBIT' in inc_stmt.index else np.nan
            roce_pst = (inc_stmt.loc['EBIT'].iloc[-4] / ce_pst) * 100 if ce_pst > 0 and 'EBIT' in inc_stmt.index else np.nan
            stock_data["ROCE 3Y Expansion (%)"] = round(roce_cur - roce_pst, 2) if pd.notna(roce_cur) and pd.notna(roce_pst) else "N/A"
        else:
            stock_data["ROCE 3Y Expansion (%)"] = "N/A"
    except Exception:
        stock_data.update({"Annual Sales 3Y Growth (%)": "N/A", "Annual PAT 3Y Growth (%)": "N/A", "Annual EPS 3Y Growth (%)": "N/A", "ROCE 3Y Expansion (%)": "N/A"})

    # --- NSE API (Quarterly) ---
    q_data = []
    safe_sym = urllib.parse.quote(sym)
    api_url = f"https://www.nseindia.com/api/corporates-financial-results?index=equities&symbol={safe_sym}&period=Quarterly"
    
    for _ in range(3):
        try:
            res = session.get(api_url, headers=req_headers, timeout=10)
            if res.status_code == 200: 
                q_data = res.json()
                break
        except Exception:
            pass
        time.sleep(2)
        
    if q_data:
        df_q = pd.DataFrame(q_data).rename(columns=lambda x: str(x).strip().lower())
        date_col = 'reviwperiodenddate' if 'reviwperiodenddate' in df_q.columns else 'todate'
        if date_col in df_q.columns:
            df_q[date_col] = pd.to_datetime(df_q[date_col], format='mixed', dayfirst=True, errors='coerce')
            df_q = df_q.dropna(subset=[date_col]).sort_values(by=date_col, ascending=False).drop_duplicates(subset=[date_col]).head(14).reset_index(drop=True)
            
            for i in range(10):
                lbl = f"Q_Minus_{i}" if i > 0 else "Q_Current"
                if i < len(df_q):
                    row = df_q.iloc[i]
                    
                    # Convert to Crores (NSE gives data in Lakhs, so divide by 100)
                    s = get_metric(row, SALES_KEYS) / 100 if pd.notna(get_metric(row, SALES_KEYS)) else np.nan
                    p = get_metric(row, PAT_KEYS) / 100 if pd.notna(get_metric(row, PAT_KEYS)) else np.nan
                    e = get_metric(row, EPS_KEYS)
                    
                    s_yoy, p_yoy, e_yoy = np.nan, np.nan, np.nan
                    if (i + 4) < len(df_q):
                        p_row = df_q.iloc[i + 4]
                        ps = get_metric(p_row, SALES_KEYS) / 100 if pd.notna(get_metric(p_row, SALES_KEYS)) else np.nan
                        pp = get_metric(p_row, PAT_KEYS) / 100 if pd.notna(get_metric(p_row, PAT_KEYS)) else np.nan
                        pe = get_metric(p_row, EPS_KEYS)
                        
                        s_yoy = calc_yoy(s, ps); p_yoy = calc_yoy(p, pp); e_yoy = calc_yoy(e, pe)
                        
                    stock_data.update({
                        f"Quarter ({lbl})": get_fy_quarter(row[date_col]), 
                        f"Sales Cr ({lbl})": round(s,2) if pd.notna(s) else "N/A", 
                        f"Sales YoY % ({lbl})": f"{round(s_yoy,1)}%" if pd.notna(s_yoy) else "N/A", 
                        f"PAT Cr ({lbl})": round(p,2) if pd.notna(p) else "N/A", 
                        f"PAT YoY % ({lbl})": f"{round(p_yoy,1)}%" if pd.notna(p_yoy) else "N/A", 
                        f"EPS ({lbl})": round(e,2) if pd.notna(e) else "N/A", 
                        f"EPS YoY % ({lbl})": f"{round(e_yoy,1)}%" if pd.notna(e_yoy) else "N/A"
                    })
                else:
                    stock_data.update({f"Quarter ({lbl})": "N/A", f"Sales Cr ({lbl})": "N/A", f"Sales YoY % ({lbl})": "N/A", f"PAT Cr ({lbl})": "N/A", f"PAT YoY % ({lbl})": "N/A", f"EPS ({lbl})": "N/A", f"EPS YoY % ({lbl})": "N/A"})
        else:
            for i in range(10):
                lbl = f"Q_Minus_{i}" if i > 0 else "Q_Current"
                stock_data.update({f"Quarter ({lbl})": "N/A", f"Sales Cr ({lbl})": "N/A", f"Sales YoY % ({lbl})": "N/A", f"PAT Cr ({lbl})": "N/A", f"PAT YoY % ({lbl})": "N/A", f"EPS ({lbl})": "N/A", f"EPS YoY % ({lbl})": "N/A"})
    else:
        for i in range(10):
            lbl = f"Q_Minus_{i}" if i > 0 else "Q_Current"
            stock_data.update({f"Quarter ({lbl})": "N/A", f"Sales Cr ({lbl})": "N/A", f"Sales YoY % ({lbl})": "N/A", f"PAT Cr ({lbl})": "N/A", f"PAT YoY % ({lbl})": "N/A", f"EPS ({lbl})": "N/A", f"EPS YoY % ({lbl})": "N/A"})

    batch_data.append(stock_data)
    time.sleep(1) 

# ==========================================
# 6. PUSH BATCH TO GOOGLE SHEETS
# ==========================================
print(f"Appending batch of {len(batch_data)} stocks to Google Sheets...")
df_batch = pd.DataFrame(batch_data).fillna("N/A")

if is_empty and not sheet_headers:
    sheet_headers = df_batch.columns.tolist()
    worksheet.update(values=[sheet_headers], range_name='A1')
    worksheet.freeze(rows=1, cols=3)
    worksheet.format('A1:BZ1', {
        "backgroundColor": {"red": 0.1, "green": 0.2, "blue": 0.4},
        "textFormat": {"foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}, "bold": True},
        "horizontalAlignment": "CENTER"
    })
    is_empty = False

for col in sheet_headers:
    if col not in df_batch.columns: df_batch[col] = "N/A"
df_batch = df_batch[sheet_headers]

worksheet.append_rows(df_batch.values.tolist(), value_input_option='USER_ENTERED')
print(f"✅ Batch saved! Re-run the script to process the next {BATCH_LIMIT} stocks.")
