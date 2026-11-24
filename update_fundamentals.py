import os
import json
import time
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials

warnings.simplefilter(action='ignore')

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
# 2. READ MASTER UNIVERSE
# ==========================================
SINGLE_SHEET_URL = 'https://docs.google.com/spreadsheets/d/1A2fUfXGKXXQxzFnoR30cFVqtmb-28KTi4fR4N0e507g/edit?usp=sharing'
MIN_RUPEE_VOL_CR = 1.0  # Covers all stocks in Daily_Technicals & Master Scanner

spreadsheet = gc.open_by_url(SINGLE_SHEET_URL)
master_ws = spreadsheet.worksheet("Avg_Rupee_Volume_Master")
df_master = pd.DataFrame(master_ws.get_all_records())

symbols = []
for _, row in df_master.iterrows():
    sym = str(row.get('Symbol', '')).replace('.NS', '').strip().upper()
    try:
        vol_cr = float(str(row.get('Avg_Rupee_Volume_Crores', 0)).replace(',', '').strip())
    except Exception:
        vol_cr = 0.0
    if sym and vol_cr >= MIN_RUPEE_VOL_CR:
        symbols.append(sym)

symbols = sorted(list(set(symbols)))
print(f"Fetching Quarterly OPM% for {len(symbols)} liquid NSE stocks (>= {MIN_RUPEE_VOL_CR} Cr)...", flush=True)

# ==========================================
# 3. HELPER: EXTRACT OPM% FROM INCOME STATEMENT
# ==========================================
OP_INC_KEYS = ['Operating Income', 'EBIT', 'Operating Profit']
REV_KEYS = ['Total Revenue', 'Operating Revenue', 'Revenue']

def get_row_value(stmt, keys, col):
    for k in keys:
        if k in stmt.index:
            val = stmt.loc[k, col]
            if pd.notna(val):
                return float(val)
    return np.nan

def calc_opm(stmt, col):
    op_inc = get_row_value(stmt, OP_INC_KEYS, col)
    rev = get_row_value(stmt, REV_KEYS, col)
    if pd.notna(op_inc) and pd.notna(rev) and rev > 0:
        return (op_inc / rev) * 100.0
    return np.nan

# ==========================================
# 4. FETCH QUARTERLY OPM%, YoY (pp), & QoQ (pp)
# ==========================================
rows_out = []

for idx, sym in enumerate(symbols, 1):
    ticker_ns = sym + '.NS'
    q0_date_str = ""
    opm_curr = np.nan
    opm_yoy_pp = np.nan
    opm_qoq_pp = np.nan

    try:
        stmt = yf.Ticker(ticker_ns).quarterly_income_stmt
        if stmt is not None and not stmt.empty:
            # Sort quarter columns newest -> oldest
            cols = sorted(stmt.columns, reverse=True)
            q0_col = cols[0]
            q0_dt = pd.to_datetime(q0_col)
            q0_date_str = q0_dt.strftime('%Y-%m-%d')
            opm_curr = calc_opm(stmt, q0_col)

            if pd.notna(opm_curr):
                # QoQ: previous quarter (~60 to 130 days prior)
                for c in cols[1:]:
                    diff_days = (q0_dt - pd.to_datetime(c)).days
                    if 60 <= diff_days <= 130:
                        opm_q1 = calc_opm(stmt, c)
                        if pd.notna(opm_q1):
                            opm_qoq_pp = opm_curr - opm_q1
                        break

                # YoY: same quarter last year (~300 to 430 days prior)
                for c in cols[1:]:
                    diff_days = (q0_dt - pd.to_datetime(c)).days
                    if 300 <= diff_days <= 430:
                        opm_q4 = calc_opm(stmt, c)
                        if pd.notna(opm_q4):
                            opm_yoy_pp = opm_curr - opm_q4
                        break
    except Exception:
        pass

    rows_out.append({
        'Symbol': sym,
        'Latest_Qtr_Date': q0_date_str,
        'Curr Qtr OPM %': round(opm_curr, 2) if pd.notna(opm_curr) else "N/A",
        'OPM YoY Chg (pp)': round(opm_yoy_pp, 2) if pd.notna(opm_yoy_pp) else "N/A",
        'OPM QoQ Chg (pp)': round(opm_qoq_pp, 2) if pd.notna(opm_qoq_pp) else "N/A"
    })

    if idx % 100 == 0 or idx == len(symbols):
        print(f" -> Processed {idx}/{len(symbols)} stocks...", flush=True)
    time.sleep(0.15)  # Gentle pacing to avoid Yahoo rate limits

# ==========================================
# 5. WRITE TO GOOGLE SHEET
# ==========================================
df_out = pd.DataFrame(rows_out).fillna("N/A")
tab_name = "Quarterly_Fundamentals"

try:
    qf_ws = spreadsheet.worksheet(tab_name)
    qf_ws.clear()
except gspread.exceptions.WorksheetNotFound:
    qf_ws = spreadsheet.add_worksheet(title=tab_name, rows="2500", cols="10")

sheet_data = [df_out.columns.values.tolist()] + df_out.values.tolist()
qf_ws.update(values=sheet_data, range_name='A1', value_input_option='USER_ENTERED')
qf_ws.freeze(rows=1)
qf_ws.format('A1:E1', {
    'textFormat': {'bold': True, 'foregroundColor': {'red': 1, 'green': 1, 'blue': 1}},
    'backgroundColor': {'red': 0.1, 'green': 0.45, 'blue': 0.91}
})

print(f"✅ Success! Cached Quarterly OPM% for {len(df_out)} stocks in '{tab_name}'.", flush=True)
