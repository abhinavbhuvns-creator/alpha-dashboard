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
# 2. GET USER STOCK UNIVERSE
# ==========================================
print("Fetching approved stock universe from Master Sheet...")
try:
    master_ws = spreadsheet.worksheet("Avg_Rupee_Volume_Master")
    master_df = pd.DataFrame(master_ws.get_all_records())
    
    universe_dict = {}
    for _, row in master_df.iterrows():
        sym = str(row.get('Symbol', '')).strip().upper()
        if sym:
            universe_dict[sym] = {
                'Company Name': row.get('Company Name', sym),
                'Industry': row.get('Industry Group', 'Unclassified')
            }
except Exception as e:
    raise SystemExit(f"🛑 Error reading Master Data: {e}")

approved_symbols = list(universe_dict.keys())

# ==========================================
# 3. HELPER FUNCTIONS FOR MATH & QUARTERS
# ==========================================
def get_fy_quarter(date_str):
    try:
        dt = pd.to_datetime(date_str)
        month = dt.month
        year = dt.year
        if month in [1, 2, 3]: return f"Q4 FY{year}"
        elif month in [4, 5, 6]: return f"Q1 FY{year+1}"
        elif month in [7, 8, 9]: return f"Q2 FY{year+1}"
        else: return f"Q3 FY{year+1}"
    except:
        return "Unknown Qtr"

def calc_yoy_growth(current, past):
    try:
        c = float(current)
        p = float(past)
        if pd.isna(c) or pd.isna(p) or p == 0:
            return np.nan
        # Absolute value on denominator handles negative to positive turnarounds mathematically
        return ((c - p) / abs(p)) * 100
    except:
        return np.nan

def calc_cagr(current, past, years=3):
    try:
        c = float(current)
        p = float(past)
        if pd.isna(c) or pd.isna(p) or p <= 0 or c <= 0:
            return np.nan
        return ((c / p) ** (1 / years) - 1) * 100
    except:
        return np.nan

# ==========================================
# 4. INITIALIZE SESSIONS
# ==========================================
print("Initializing NSE Session & Fetching Fundamentals...")
session = requests.Session()
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.9'
}
session.get("https://www.nseindia.com", headers=headers, timeout=10)
time.sleep(2)

final_fundamental_data = []

# ==========================================
# 5. FETCH DATA SYMBOL BY SYMBOL
# ==========================================
for idx, sym in enumerate(approved_symbols):
    if idx > 0 and idx % 40 == 0:
        print(f" -> Resting session at {idx} stocks to prevent IP ban...")
        session.close()
        time.sleep(5)
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        time.sleep(2)

    print(f"Processing {sym} ({idx+1}/{len(approved_symbols)})...")
    stock_data = {
        "Stock Symbol": sym,
        "Industry": universe_dict[sym]['Industry'],
        "Company Name": universe_dict[sym]['Company Name']
    }

    # --- PART A: YFINANCE FOR ANNUAL & ROCE ---
    yf_sym = f"{sym}.NS"
    ticker = yf.Ticker(yf_sym)
    
    annual_sales_cagr = np.nan
    annual_pat_cagr = np.nan
    annual_eps_cagr = np.nan
    roce_growth = np.nan

    try:
        # Fetch Income Statement and Balance Sheet
        inc_stmt = ticker.financials
        bal_sheet = ticker.balance_sheet
        
        if not inc_stmt.empty and len(inc_stmt.columns) >= 4:
            # Sort columns chronologically (oldest to newest)
            inc_stmt = inc_stmt[sorted(inc_stmt.columns)]
            
            sales_row = inc_stmt.loc['Total Revenue'] if 'Total Revenue' in inc_stmt.index else None
            pat_row = inc_stmt.loc['Net Income'] if 'Net Income' in inc_stmt.index else None
            eps_row = inc_stmt.loc['Basic EPS'] if 'Basic EPS' in inc_stmt.index else None
            
            if sales_row is not None:
                annual_sales_cagr = calc_cagr(sales_row.iloc[-1], sales_row.iloc[-4], 3)
            if pat_row is not None:
                annual_pat_cagr = calc_yoy_growth(pat_row.iloc[-1], pat_row.iloc[-4]) # Simple 3Y absolute growth for profit
            if eps_row is not None:
                annual_eps_cagr = calc_yoy_growth(eps_row.iloc[-1], eps_row.iloc[-4])

        # ROCE Calculation: EBIT / (Total Assets - Current Liabilities)
        if not inc_stmt.empty and not bal_sheet.empty and len(bal_sheet.columns) >= 4:
            bal_sheet = bal_sheet[sorted(bal_sheet.columns)]
            
            ebit_row = inc_stmt.loc['EBIT'] if 'EBIT' in inc_stmt.index else None
            assets_row = bal_sheet.loc['Total Assets'] if 'Total Assets' in bal_sheet.index else None
            curr_liab_row = bal_sheet.loc['Current Liabilities'] if 'Current Liabilities' in bal_sheet.index else None
            
            if ebit_row is not None and assets_row is not None and curr_liab_row is not None:
                try:
                    ce_current = assets_row.iloc[-1] - curr_liab_row.iloc[-1]
                    roce_current = (ebit_row.iloc[-1] / ce_current) * 100 if ce_current > 0 else np.nan
                    
                    ce_past = assets_row.iloc[-4] - curr_liab_row.iloc[-4]
                    roce_past = (ebit_row.iloc[-4] / ce_past) * 100 if ce_past > 0 else np.nan
                    
                    roce_growth = roce_current - roce_past # Basis point/percentage expansion
                except:
                    pass

    except Exception:
        pass

    stock_data["Annual Sales 3Y Growth (%)"] = round(annual_sales_cagr, 2) if pd.notna(annual_sales_cagr) else "N/A"
    stock_data["Annual PAT 3Y Growth (%)"] = round(annual_pat_cagr, 2) if pd.notna(annual_pat_cagr) else "N/A"
    stock_data["Annual EPS 3Y Growth (%)"] = round(annual_eps_cagr, 2) if pd.notna(annual_eps_cagr) else "N/A"
    stock_data["ROCE 3Y Expansion (%)"] = round(roce_growth, 2) if pd.notna(roce_growth) else "N/A"


    # --- PART B: NSE API FOR QUARTERLY RESULTS ---
    safe_sym = urllib.parse.quote(sym)
    api_url = f"https://www.nseindia.com/api/corporates-financial-results?index=equities&symbol={safe_sym}&period=Quarterly"
    
    q_data = []
    for attempt in range(3):
        try:
            res = session.get(api_url, headers=headers, timeout=10)
            if res.status_code == 200:
                q_data = res.json()
                break
        except Exception:
            pass
        time.sleep(2)
        
    if not q_data:
        final_fundamental_data.append(stock_data)
        continue

    # Extract and clean quarterly JSON
    df_q = pd.DataFrame(q_data)
    if 'fromDate' not in df_q.columns or 'reviwPeriodEndDate' not in df_q.columns:
        final_fundamental_data.append(stock_data)
        continue

    # Standardize columns (handling NSE's inconsistent naming)
    df_q = df_q.rename(columns=lambda x: x.strip().lower())
    
    # We need End Date, Income, Profit, and EPS. We force them to numeric.
    date_col = 'reviwperiodenddate' if 'reviwperiodenddate' in df_q.columns else 'todate'
    sales_col = 'totincomefromoperations' if 'totincomefromoperations' in df_q.columns else 'incomefromoperations'
    pat_col = 'netprofitlossforperiod'
    eps_col = 'dilutedepsbeforeextraordinaryitems'
    
    if date_col not in df_q.columns:
        final_fundamental_data.append(stock_data)
        continue

    df_q[date_col] = pd.to_datetime(df_q[date_col], format='mixed', dayfirst=True, errors='coerce')
    df_q = df_q.dropna(subset=[date_col]).sort_values(by=date_col, ascending=False).drop_duplicates(subset=[date_col])
    
    for col in [sales_col, pat_col, eps_col]:
        if col in df_q.columns:
            df_q[col] = pd.to_numeric(df_q[col], errors='coerce')
        else:
            df_q[col] = np.nan

    # Limit to the required 14 quarters (to calculate YoY for 10 quarters)
    df_q = df_q.head(14).reset_index(drop=True)
    
    # Generate the 10-Quarter layout
    for i in range(10):
        q_label = f"Q_Minus_{i}" if i > 0 else "Q_Current"
        
        if i < len(df_q):
            current_row = df_q.iloc[i]
            q_name = get_fy_quarter(current_row[date_col])
            sales = current_row[sales_col] / 100 if pd.notna(current_row[sales_col]) else np.nan # Convert to Crores usually reported in Lakhs
            pat = current_row[pat_col] / 100 if pd.notna(current_row[pat_col]) else np.nan
            eps = current_row[eps_col] if pd.notna(current_row[eps_col]) else np.nan
            
            # YoY calculations (Compare against index i + 4)
            sales_yoy = np.nan
            pat_yoy = np.nan
            eps_yoy = np.nan
            
            if (i + 4) < len(df_q):
                past_row = df_q.iloc[i + 4]
                past_sales = past_row[sales_col] / 100 if pd.notna(past_row[sales_col]) else np.nan
                past_pat = past_row[pat_col] / 100 if pd.notna(past_row[pat_col]) else np.nan
                past_eps = past_row[eps_col] if pd.notna(past_row[eps_col]) else np.nan
                
                sales_yoy = calc_yoy_growth(sales, past_sales)
                pat_yoy = calc_yoy_growth(pat, past_pat)
                eps_yoy = calc_yoy_growth(eps, past_eps)
            
            stock_data[f"Quarter ({q_label})"] = q_name
            stock_data[f"Sales Cr ({q_label})"] = round(sales, 2) if pd.notna(sales) else "N/A"
            stock_data[f"Sales YoY % ({q_label})"] = f"{round(sales_yoy, 1)}%" if pd.notna(sales_yoy) else "N/A"
            
            stock_data[f"PAT Cr ({q_label})"] = round(pat, 2) if pd.notna(pat) else "N/A"
            stock_data[f"PAT YoY % ({q_label})"] = f"{round(pat_yoy, 1)}%" if pd.notna(pat_yoy) else "N/A"
            
            stock_data[f"EPS ({q_label})"] = round(eps, 2) if pd.notna(eps) else "N/A"
            stock_data[f"EPS YoY % ({q_label})"] = f"{round(eps_yoy, 1)}%" if pd.notna(eps_yoy) else "N/A"
        else:
            stock_data[f"Quarter ({q_label})"] = "N/A"
            stock_data[f"Sales Cr ({q_label})"] = "N/A"
            stock_data[f"Sales YoY % ({q_label})"] = "N/A"
            stock_data[f"PAT Cr ({q_label})"] = "N/A"
            stock_data[f"PAT YoY % ({q_label})"] = "N/A"
            stock_data[f"EPS ({q_label})"] = "N/A"
            stock_data[f"EPS YoY % ({q_label})"] = "N/A"

    final_fundamental_data.append(stock_data)

# ==========================================
# 6. EXPORT TO GOOGLE SHEETS
# ==========================================
print("\nFormatting and writing to Google Sheets...")
df_final = pd.DataFrame(final_fundamental_data)
df_final = df_final.fillna("N/A")

try:
    worksheet = spreadsheet.worksheet(TAB_NAME)
    worksheet.clear()
except gspread.exceptions.WorksheetNotFound:
    # 75 columns width handles 10 quarters of 7 metrics plus base data
    worksheet = spreadsheet.add_worksheet(title=TAB_NAME, rows="2000", cols="80")

sheet_output = [df_final.columns.values.tolist()] + df_final.values.tolist()

worksheet.update(values=sheet_output, range_name='A1', value_input_option='USER_ENTERED')
worksheet.freeze(rows=1, cols=1)

# Format Headers
worksheet.format('A1:BZ1', {
    "backgroundColor": {"red": 0.1, "green": 0.2, "blue": 0.4},
    "textFormat": {"foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}, "bold": True},
    "horizontalAlignment": "CENTER"
})

print(f"✅ Fundamental Data fully updated with 10 Quarters of YoY math for {len(approved_symbols)} stocks!")
