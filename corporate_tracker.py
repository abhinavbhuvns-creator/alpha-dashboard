import os
import json
import time
import requests
import urllib.parse
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta

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
TAB_NAME = "Corporate_Announcements"

# ==========================================
# 2. GET USER STOCK UNIVERSE & INDUSTRIES
# ==========================================
print("Fetching approved stock universe from Master Sheet...")
try:
    master_ws = spreadsheet.worksheet("Avg_Rupee_Volume_Master")
    master_data = master_ws.get_all_records()
    master_df = pd.DataFrame(master_data)
    
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
# 3. ADVANCED STRICT-HIERARCHY CATEGORIZATION
# ==========================================
def categorize_announcement(subject, desc):
    text = (str(subject) + " " + str(desc)).lower()
    
    # 🟢 PRIORITY 1: Legal & Court (Catches Court Orders before Business Orders)
    # Replaced dangerous short acronyms with full phrasing to prevent false positives
    legal_keywords = [
        "court", "tribunal", "litigation", "nclt", "sebi", "central bureau of investigation", 
        "enforcement directorate", "income tax", "penalty", "show cause", "default", "fraud", 
        "forensic", "summons", "writ petition", "order passed", "order dated", "hon'ble", 
        "supreme court", "high court"
    ]
    if any(kw in text for kw in legal_keywords):
        return "Legal / Red Flags"
        
    # 🟢 PRIORITY 2: Core Business Catalysts
    if any(kw in text for kw in ["financial result", "audited", "unaudited", "earnings"]):
        return "Financial Results"
        
    if any(kw in text for kw in ["order", "award", "contract", "letter of intent", "bagged", "bagging"]):
        return "Order Wins"
        
    if any(kw in text for kw in ["capex", "capacity", "expansion", "commissioning", "new plant", "new unit", "commercial production", "new venture", "sets up"]):
        return "Capacity Expansion or New Ventures"
        
    if any(kw in text for kw in ["fund raising", "preferential issue", "qip", "rights issue", "private placement"]):
        return "Fund Raising"
        
    if any(kw in text for kw in ["acquisition", "takeover", "merger", "amalgamation", "stake sale", "buyout"]):
        return "Acquisitions"

    # 🟢 PRIORITY 3: Standard Corporate Actions & Updates
    if any(kw in text for kw in ["concall", "analyst meet", "earnings call", "investor presentation", "institutional investor", "transcript", "audio recording"]):
        return "Concall / Analyst Meet"
        
    if any(kw in text for kw in ["credit rating", "crisil", "icra", "care", "fitch", "rating downgrade", "rating upgrade"]):
        return "Credit Rating"
        
    if any(kw in text for kw in ["bonus", "split", "sub-division", "allotment", "buyback"]):
        return "Corporate Action"
        
    if any(kw in text for kw in ["dividend", "record date"]):
        return "Dividend"
        
    if any(kw in text for kw in ["shareholding", "pledge", "promoter", "block deal"]):
        return "Shareholding"
        
    if any(kw in text for kw in ["appointment", "resignation", "cessation", "change in director", "change in management", "kmp"]):
        return "Appointments / Resignations"
        
    if any(kw in text for kw in ["board meeting"]):
        return "Board Meeting"
        
    if any(kw in text for kw in ["annual report", "agm", "egm", "annual general meeting"]):
        return "Annual Report"

    return "Other Important Announcements"

# ==========================================
# 4. DETERMINE FETCH RANGES & INITIALIZE SESSION
# ==========================================
try:
    worksheet = spreadsheet.worksheet(TAB_NAME)
    existing_data = worksheet.get_all_values()
    is_empty = len(existing_data) <= 1
except gspread.exceptions.WorksheetNotFound:
    worksheet = spreadsheet.add_worksheet(title=TAB_NAME, rows="5000", cols="7")
    existing_data = []
    is_empty = True

print("Connecting to NSE API...")
session = requests.Session()
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.9'
}
session.get("https://www.nseindia.com", headers=headers, timeout=10)
time.sleep(2)

raw_data = []

# ==========================================
# 5. SMART FETCHING LOGIC (DEEP BACKFILL VS DAILY)
# ==========================================
if is_empty:
    print(f"Sheet is empty! Initiating 6-month Deep Backfill for {len(approved_symbols)} stocks...")
    print("This will take 5-10 minutes. Fetching symbol by symbol to bypass NSE time limits.")
    
    for idx, sym in enumerate(approved_symbols):
        # 🟢 ANTI-BLOCK MEASURE: Refresh session every 50 stocks
        if idx > 0 and idx % 50 == 0:
            print(f" -> Rotated session at {idx} stocks to prevent IP ban...")
            session.close()
            time.sleep(5)
            session = requests.Session()
            session.get("https://www.nseindia.com", headers=headers, timeout=10)
            time.sleep(2)
            
        safe_sym = urllib.parse.quote(sym)
        api_url = f"https://www.nseindia.com/api/corporate-announcements?index=equities&symbol={safe_sym}"
        
        for attempt in range(3):
            try:
                response = session.get(api_url, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    raw_data.extend(data)
                    break
            except Exception:
                pass
            time.sleep(2)
            
        time.sleep(1) # Base delay to mimic human behavior
        
else:
    print("Sheet has data. Fetching fast daily updates for the last 3 days...")
    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=3)
    
    api_url = f"https://www.nseindia.com/api/corporate-announcements?index=equities&from_date={start_dt.strftime('%d-%m-%Y')}&to_date={end_dt.strftime('%d-%m-%Y')}"
    
    for attempt in range(3):
        try:
            response = session.get(api_url, headers=headers, timeout=15)
            if response.status_code == 200:
                raw_data.extend(response.json())
                break
        except Exception:
            pass
        time.sleep(3)

if not raw_data:
    print("No data fetched from NSE. Try again later.")
    raise SystemExit(0)

# ==========================================
# 6. FILTER & CATEGORIZE DATA
# ==========================================
print("Filtering against Master Universe and extracting detailed announcements...")
filtered_announcements = []

for row in raw_data:
    
    # Lowercase all keys to avoid NSE case-change bugs
    row_lower = {str(k).lower(): v for k, v in row.items()}
    symbol = str(row_lower.get('symbol', '')).strip().upper()
    
    if symbol in approved_symbols:
        
        # 🟢 BULLETPROOF DATE FINDER
        raw_date = str(row_lower.get('an_dt', '') or row_lower.get('broadcastdate', '') or row_lower.get('date', '')).strip()
        if raw_date.lower() in ['none', 'nan', 'null']: raw_date = ""
        clean_date = raw_date.split(' ')[0] if raw_date else ""
        
        # 🟢 BULLETPROOF PDF LINK FINDER
        file_path = str(row_lower.get('attchmntfile', '')).strip()
        if file_path.lower() in ['none', 'nan', 'null']: file_path = ""
        
        link = ""
        if file_path and len(file_path) > 3:
            if file_path.startswith('http'):
                link = file_path
            elif file_path.startswith('/'):
                link = f"https://nsearchives.nseindia.com{file_path}"
            else:
                link = f"https://nsearchives.nseindia.com/{file_path}"

        # 🟢 EXACT TEXT EXTRACTION
        subject = str(row_lower.get('desc', '')).strip()
        if subject.lower() in ['none', 'nan', 'null']: subject = ""
        
        detailed_text = str(row_lower.get('attchmnttext', '')).strip()
        if detailed_text.lower() in ['none', 'nan', 'null']: detailed_text = ""
        
        final_announcement = detailed_text if len(detailed_text) > len(subject) else subject

        category = categorize_announcement(subject, detailed_text)
        
        # Drop meaningless daily noise unless it caught a specific bucket
        if category != "Other Important Announcements" or ("update" not in detailed_text.lower() and "loss of share" not in detailed_text.lower()):
            filtered_announcements.append({
                "Date": clean_date,
                "Industry": universe_dict[symbol]['Industry'],
                "Category": category,
                "Company Name": universe_dict[symbol]['Company Name'],
                "NSE Code": symbol,
                "Announcement": final_announcement,
                "PDF Link": link
            })

if not filtered_announcements:
    print("No relevant announcements found for your stock universe in this timeframe.")
    raise SystemExit(0)

df_new = pd.DataFrame(filtered_announcements)
desired_columns = ["Date", "Industry", "Category", "Company Name", "NSE Code", "Announcement", "PDF Link"]
df_new = df_new[desired_columns]

# ==========================================
# 7. APPEND TO GOOGLE SHEETS
# ==========================================
print(f"Appending {len(df_new)} categorized announcements to Google Sheets...")

if not is_empty:
    existing_df = pd.DataFrame(existing_data[1:], columns=existing_data[0])
    # Add any missing columns to prevent misalignment
    for col in desired_columns:
        if col not in existing_df.columns:
            existing_df[col] = ""
    existing_df = existing_df[desired_columns]
    df_final = pd.concat([df_new, existing_df]).drop_duplicates(subset=['Date', 'NSE Code', 'Category', 'Announcement'])
else:
    df_final = df_new

# Keep history clean and sorted
df_final = df_final.head(5000)
df_final['Date'] = pd.to_datetime(df_final['Date'], format='mixed', dayfirst=True, errors='coerce')
df_final = df_final.sort_values(by='Date', ascending=False)
df_final['Date'] = df_final['Date'].dt.strftime('%d-%b-%Y')

sheet_output = [df_final.columns.values.tolist()] + df_final.fillna("").values.tolist()

worksheet.clear() 
worksheet.update(values=sheet_output, range_name='A1', value_input_option='USER_ENTERED')
worksheet.freeze(rows=1)
worksheet.format('A1:G1', {
    "backgroundColor": {"red": 0.1, "green": 0.2, "blue": 0.4},
    "textFormat": {"foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}, "bold": True}
})

print("✅ Corporate Announcements Database successfully updated!")
