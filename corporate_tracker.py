import os
import json
import time
import requests
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
# 2. GET USER STOCK UNIVERSE
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
# 3. ADVANCED CATEGORIZATION ENGINE
# ==========================================
def categorize_announcement(subject, desc):
    text = (str(subject) + " " + str(desc)).lower()
    categories = {
        "Order Wins": ["order", "award", "contract", "letter of intent", "loi"],
        "Capex / Expansions": ["capex", "capacity", "expansion", "commissioning", "new plant", "new unit", "commercial production"],
        "Credit Rating": ["credit rating", "crisil", "icra", "care", "fitch", "rating downgrade", "rating upgrade"],
        "Mergers & Acquisitions": ["acquisition", "takeover", "merger", "amalgamation", "stake sale", "buyout"],
        "Corporate Action": ["bonus", "split", "sub-division", "allotment", "rights issue", "buyback"],
        "Concall / Analyst Meet": ["concall", "analyst meet", "earnings call", "investor presentation", "institutional investor", "transcript", "audio recording"],
        "Financial Results": ["financial result", "audited", "unaudited", "earnings"],
        "Board Meeting": ["board meeting"],
        "Appointments / Resignations": ["appointment", "resignation", "cessation", "change in director", "change in management", "kmp"],
        "Shareholding": ["shareholding", "pledge", "promoter", "block deal"],
        "Dividend": ["dividend", "record date"],
        "Annual Report / AGM": ["annual report", "agm", "egm", "annual general meeting"],
        "Legal / Red Flags": ["default", "penalty", "show cause", "sebi", "cbi", "ed ", "income tax", "fraud", "resignation of auditor", "nclt", "bankruptcy", "forensic"]
    }
    
    for cat, keywords in categories.items():
        if any(kw in text for kw in keywords):
            return cat
    return "Other Important Announcements"

# ==========================================
# 4. DETERMINE FETCH RANGES (6-MONTH BACKFILL LOGIC)
# ==========================================
try:
    worksheet = spreadsheet.worksheet(TAB_NAME)
    existing_data = worksheet.get_all_values()
    is_empty = len(existing_data) <= 1
except gspread.exceptions.WorksheetNotFound:
    worksheet = spreadsheet.add_worksheet(title=TAB_NAME, rows="5000", cols="8")
    existing_data = []
    is_empty = True

fetch_ranges = []
if is_empty:
    print("Sheet is empty. Initiating 6-month historical backfill (bypassing NSE limits via 30-day chunks)...")
    end_dt = datetime.today()
    for _ in range(6):
        start_dt = end_dt - timedelta(days=30)
        fetch_ranges.append((start_dt.strftime("%d-%m-%Y"), end_dt.strftime("%d-%m-%Y")))
        end_dt = start_dt - timedelta(days=1)
else:
    print("Sheet has data. Fetching recent updates for the last 7 days...")
    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=7)
    fetch_ranges.append((start_dt.strftime("%d-%m-%Y"), end_dt.strftime("%d-%m-%Y")))

# ==========================================
# 5. FETCH LIVE NSE ANNOUNCEMENTS
# ==========================================
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
for start_str, end_str in fetch_ranges:
    api_url = f"https://www.nseindia.com/api/corporate-announcements?index=equities&from_date={start_str}&to_date={end_str}"
    print(f" -> Fetching timeframe: {start_str} to {end_str}")
    
    for attempt in range(3):
        try:
            response = session.get(api_url, headers=headers, timeout=15)
            if response.status_code == 200:
                chunk_data = response.json()
                raw_data.extend(chunk_data)
                print(f"    ✅ Got {len(chunk_data)} records.")
                break
            else:
                print(f"    ⚠️ Status {response.status_code}. Retrying...")
        except Exception as e:
            pass
        time.sleep(3)
    time.sleep(2) 

if not raw_data:
    print("No data fetched from NSE. Try again later.")
    raise SystemExit(0)

# ==========================================
# 6. FILTER & CATEGORIZE DATA
# ==========================================
print("Filtering against Master Universe and separating Summary column...")
df = pd.DataFrame(raw_data)
filtered_announcements = []

for _, row in df.iterrows():
    symbol = str(row.get('symbol', '')).strip().upper()
    desc = str(row.get('desc', ''))
    subject = str(row.get('sm_name', ''))
    
    # Date Fallback Logic
    raw_date = str(row.get('anndate', '') or row.get('broadCastDate', '') or row.get('date', ''))
    clean_date = raw_date.split(' ')[0] if raw_date else ""
    
    # PDF Link Formatting
    file_path = row.get('attchmntText', '') or row.get('attchmntFile', '')
    if file_path:
        link = f"https://nsearchives.nseindia.com{file_path}" if not file_path.startswith('http') else file_path
    else:
        link = ""

    if symbol in approved_symbols:
        category = categorize_announcement(subject, desc)
        
        # Drop meaningless daily noise
        if category != "Other Important Announcements" or ("update" not in desc.lower() and "loss of share" not in desc.lower()):
            filtered_announcements.append({
                "Date": clean_date,
                "Industry": universe_dict[symbol]['Industry'],
                "Category": category,
                "Company Name": universe_dict[symbol]['Company Name'],
                "NSE Code": symbol,
                "Announcement": subject,
                "Summary": desc,
                "PDF Link": link
            })

if not filtered_announcements:
    print("No relevant announcements found for your stock universe in this timeframe.")
    raise SystemExit(0)

df_new = pd.DataFrame(filtered_announcements)

# ==========================================
# 7. APPEND TO GOOGLE SHEETS
# ==========================================
print(f"Appending {len(df_new)} categorized announcements to Google Sheets...")

if not is_empty:
    existing_df = pd.DataFrame(existing_data[1:], columns=existing_data[0])
    df_final = pd.concat([df_new, existing_df]).drop_duplicates(subset=['Date', 'NSE Code', 'Category', 'Announcement'])
else:
    df_final = df_new

# Keep history clean and sorted
df_final = df_final.head(5000)
df_final['Date'] = pd.to_datetime(df_final['Date'], format='mixed', dayfirst=True, errors='coerce')
df_final = df_final.sort_values(by='Date', ascending=False)
df_final['Date'] = df_final['Date'].dt.strftime('%d-%b-%Y')

sheet_output = [df_final.columns.values.tolist()] + df_final.fillna("").values.tolist()
worksheet.update(values=sheet_output, range_name='A1', value_input_option='USER_ENTERED')
worksheet.freeze(rows=1)
worksheet.format('A1:H1', {
    "backgroundColor": {"red": 0.1, "green": 0.2, "blue": 0.4},
    "textFormat": {"foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}, "bold": True}
})

print("✅ Corporate Announcements Database successfully updated!")
