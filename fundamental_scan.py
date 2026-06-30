import os
import json
import time
import warnings
import pandas as pd
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials

warnings.simplefilter(action='ignore')

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
# 2. READ UNIFIED MASTER SHEET
# ==========================================
SINGLE_SHEET_URL = 'https://docs.google.com/spreadsheets/d/1A2fUfXGKXXQxzFnoR30cFVqtmb-28KTi4fR4N0e507g/edit?usp=sharing'
spreadsheet = gc.open_by_url(SINGLE_SHEET_URL)
MASTER_TAB_NAME = "Avg_Rupee_Volume_Master"
TARGET_TAB_NAME = "Fundamental_GARP"

print(f"Reading existing data from '{MASTER_TAB_NAME}'...")
try:
    master_ws = spreadsheet.worksheet(MASTER_TAB_NAME)
    df_master = pd.DataFrame(master_ws.get_all_records())
except Exception as e:
    raise SystemExit(f"🛑 Error: Could not read Master tab. {e}")

# We will only query stocks with at least SOME trading volume to save hours of processing time
df_liquid = df_master[df_master['Avg_Rupee_Volume_Crores'] >= 1.0]
tickers = (df_liquid['Symbol'].str.strip() + '.NS').tolist()
industry_map = df_liquid.set_index('Symbol')['Industry Group'].to_dict()

# ==========================================
# 3. THE FINVIZ GARP FUNDAMENTAL SCAN
# ==========================================
print(f"Running Deep Fundamental Scan on {len(tickers)} liquid stocks...")
print("This will take a few minutes to respect server rate limits...")

passed_stocks = []

# Helper function to safely handle missing Yahoo Finance data
def safe_get(info_dict, key, default):
    val = info_dict.get(key)
    return val if val is not None else default

for i, ticker in enumerate(tickers):
    stock_sym = ticker.replace('.NS', '')
    
    # Progress tracker
    if i % 100 == 0 and i > 0:
        print(f" -> Processed {i}/{len(tickers)} stocks...")
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info or 'currentPrice' not in info:
            continue
            
        # --- EXTRACT METRICS ---
        # Finviz 'cap_largeover': In India, we'll enforce a min Market Cap of ~5000 Cr to represent established mid/large caps
        mcap = safe_get(info, 'marketCap', 0) 
        
        # Valuation (P/E < 40, Fwd P/E < 25, PEG < 2)
        pe = safe_get(info, 'trailingPE', 999)
        fpe = safe_get(info, 'forwardPE', 999)
        peg = safe_get(info, 'pegRatio', 999)
        
        # Health & Margins (Cur Ratio > 1, Gross > 15%, Oper > 0, Net > 0, Debt/Eq < 1)
        # Note: Yahoo stores Debt/Equity as a percentage (e.g., 45.0 means 0.45)
        cur_ratio = safe_get(info, 'currentRatio', 0)
        gross_m = safe_get(info, 'grossMargins', 0)
        oper_m = safe_get(info, 'operatingMargins', 0)
        net_m = safe_get(info, 'profitMargins', 0)
        dte = safe_get(info, 'debtToEquity', 999) 
        
        # Growth & Returns (ROE > 10%, Rev YoY > 10%, EPS YoY > 0%)
        roe = safe_get(info, 'returnOnEquity', 0)
        rev_growth = safe_get(info, 'revenueGrowth', 0)
        eps_growth = safe_get(info, 'earningsQuarterlyGrowth', 0)
        
        # Technical (Price > 200 SMA)
        price = safe_get(info, 'currentPrice', 0)
        sma200 = safe_get(info, 'twoHundredDayAverage', 999999)

        # --- APPLY FINVIZ LOGIC ---
        if mcap < 50000000000: continue           # Must be > 5,000 Cr Market Cap
        if pe > 40: continue                      # P/E under 40
        if fpe > 25: continue                     # Forward P/E under 25
        if peg > 2.0: continue                    # PEG under 2
        if cur_ratio < 1.0: continue              # Current Ratio > 1
        if gross_m < 0.15: continue               # Gross Margin > 15%
        if oper_m <= 0: continue                  # Operating Margin Positive
        if net_m <= 0: continue                   # Net Margin Positive
        if dte > 100: continue                    # LT Debt/Equity < 1.0 (Yahoo uses 100%)
        if roe < 0.10: continue                   # ROI/ROE > 10%
        if rev_growth < 0.10: continue            # Sales YoY > 10%
        if eps_growth <= 0: continue              # EPS Growth Positive
        if price <= sma200: continue              # Price above 200 SMA
        
        # If it passes all criteria, add to our list
        industry = industry_map.get(stock_sym, "Unclassified")
        mcap_cr = round(mcap / 10000000, 2)
        
        passed_stocks.append([
            stock_sym, 
            industry, 
            mcap_cr, 
            round(pe, 2) if pe != 999 else "N/A", 
            round(fpe, 2) if fpe != 999 else "N/A",
            round(peg, 2) if peg != 999 else "N/A",
            round(dte, 2) if dte != 999 else "N/A",
            f"{round(roe * 100, 2)}%",
            f"{round(rev_growth * 100, 2)}%",
            f"{round(net_m * 100, 2)}%"
        ])
        
    except Exception as e:
        continue
        
    # Small sleep to prevent Yahoo Finance from IP banning us
    time.sleep(0.3)

# ==========================================
# 4. WRITE TO GOOGLE SHEETS
# ==========================================
print(f"\nScan complete! Found {len(passed_stocks)} stocks meeting all fundamental criteria.")
print(f"Exporting to '{TARGET_TAB_NAME}' tab...")

headers = [
    "Stock Symbol", "Industry Group", "Market Cap (Cr)", "P/E", 
    "Forward P/E", "PEG Ratio", "Debt/Equity", "ROE", "Rev Growth YoY", "Net Margin"
]

try:
    target_ws = spreadsheet.worksheet(TARGET_TAB_NAME)
    target_ws.clear()
except gspread.exceptions.WorksheetNotFound:
    target_ws = spreadsheet.add_worksheet(title=TARGET_TAB_NAME, rows="500", cols="15")

if len(passed_stocks) > 0:
    # Sort alphabetically by symbol
    passed_stocks.sort(key=lambda x: x[0])
    sheet_output = [headers] + passed_stocks
else:
    sheet_output = [headers] + [["No stocks met criteria this week.", "", "", "", "", "", "", "", "", ""]]

target_ws.update(values=sheet_output, range_name='A1', value_input_option='USER_ENTERED')

# Format Headers
target_ws.format('A1:J1', {
    "backgroundColor": {"red": 0.1, "green": 0.4, "blue": 0.2},
    "horizontalAlignment": "CENTER",
    "textFormat": {"foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}, "bold": True}
})
target_ws.freeze(rows=1)

print("✅ Fundamental Watchlist successfully updated!")
