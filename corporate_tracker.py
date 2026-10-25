import os
import json
import time
import requests
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
import warnings

warnings.simplefilter(action='ignore')

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
# 3. ADVANCED COMPOUND PHRASE CATEGORIZATION
# ==========================================
def categorize_announcement(subject, desc):
    text = (str(subject) + " " + str(desc)).lower()
    
    # 🟢 NOISE TRAP: Catch NSE boilerplate before it hits other filters
    if any(kw in text for kw in ["significant movement in price", "significant increase in volume", "clarification on price movement"]):
        return "Exchange Clarification"

    # 🟢 PRIORITY 1: Legal & Court (Full phrasing to avoid false positives)
    legal_keywords = [
        "court", "tribunal", "litigation", "nclt", "central bureau of investigation", 
        "enforcement directorate", "income tax", "penalty", "show cause", "default", "fraud", 
        "forensic", "summons", "writ petition", "order passed", "orders passed", "order dated", 
        "hon'ble", "supreme court", "high court", "order in favour", "order-in-original"
    ]
    if any(kw in text for kw in legal_keywords):
        return "Legal / Red Flags"
        
    # 🟢 PRIORITY 2: Core Business Catalysts (Compound Phrases ONLY)
    if any(kw in text for kw in ["financial result", "audited", "unaudited", "earnings"]):
        return "Financial Results"
        
    order_keywords = [
        "work order", "purchase order", "bagged an order", "received an order", 
        "receipt of order", "commercial order", "epc contract", "letter of intent", 
        "letter of award", "awarded a contract", "awarded the project", "bagged", "bagging"
    ]
    if any(kw in text for kw in order_keywords):
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
        
    if "board meeting" in text:
        return "Board Meeting"
        
    if any(kw in text for kw in ["annual report", "agm", "egm", "annual general meeting"]):
        return "Annual Report"

    return "Other Important Announcements"

# ==========================================
# 4. INITIALIZE SESSION & GET EXISTING DATA
# ==========================================
try:
    worksheet = spreadsheet.worksheet(TAB_NAME)
    existing_data = worksheet.get_all_values()
except gspread.exceptions.WorksheetNotFound:
    worksheet = spreadsheet.add_worksheet(title=TAB_NAME, rows="5000", cols="7")
    existing_data = []

desired_columns = ["Date", "Industry", "Category", "Company Name", "NSE Code", "Announcement", "PDF Link"]

if len(existing_data) > 1:
    existing_df = pd.DataFrame(existing_data[1:], columns=existing_data[0])
    for col in desired_columns:
        if col not in existing_df.columns:
            existing_df[col] = ""
    existing_df = existing_df[desired_columns]
else:
    existing_df = pd.DataFrame(columns=desired_columns)

print("Connecting to NSE API...")
session = requests.Session()
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.9'
}
session.get("https://www.nseindia.com", headers=headers, timeout=10)
time.sleep(2)

# ==========================================
# 5. FETCH 15 DAYS OF DATA IN 5-DAY CHUNKS
# ==========================================
print("Fetching 15 days of market data...")
fetch_ranges = []
end_dt = datetime.today()
for _ in range(3):
    start_dt = end_dt - timedelta(days=5)
    fetch_ranges.append((start_dt.strftime("%d-%m-%Y"), end_dt.strftime("%d-%m-%Y")))
    end_dt = start_dt - timedelta(days=1)

raw_data = []
for start_str, end_str in fetch_ranges:
    api_url = f"https://www.nseindia.com/api/corporate-announcements?index=equities&from_date={start_str}&to_date={end_str}"
    print(f" -> Fetching timeframe: {start_str} to {end_str}")
    
    for attempt in range(3):
        try:
            response = session.get(api_url, headers=headers, timeout=15)
            if response.status_code == 200:
                raw_data.extend(response.json())
                break
        except Exception:
            pass
        time.sleep(2)
    time.sleep(1) 

if not raw_data:
    print("No data fetched from NSE. Try again later.")
    raise SystemExit(0)

# ==========================================
# 6. FILTER & CATEGORIZE
# ==========================================
print("Filtering against Master Universe...")
filtered_announcements = []

for row in raw_data:
    row_lower = {str(k).lower(): v for k, v in row.items()}
    symbol = str(row_lower.get('symbol', '')).strip().upper()
    
    if symbol in approved_symbols:
        raw_date = str(row_lower.get('an_dt', '') or row_lower.get('broadcastdate', '') or row_lower.get('date', '')).strip()
        if raw_date.lower() in ['none', 'nan', 'null']: raw_date = ""
        clean_date = raw_date.split(' ')[0] if raw_date else ""
        
        file_path = str(row_lower.get('attchmntfile', '')).strip()
        if file_path.lower() in ['none', 'nan', 'null']: file_path = ""
        link = ""
        if file_path and len(file_path) > 3:
            if file_path.startswith('http'): link = file_path
            elif file_path.startswith('/'): link = f"https://nsearchives.nseindia.com{file_path}"
            else: link = f"https://nsearchives.nseindia.com/{file_path}"

        subject = str(row_lower.get('desc', '')).strip()
        if subject.lower() in ['none', 'nan', 'null']: subject = ""
        detailed_text = str(row_lower.get('attchmnttext', '')).strip()
        if detailed_text.lower() in ['none', 'nan', 'null']: detailed_text = ""
        
        final_announcement = detailed_text if len(detailed_text) > len(subject) else subject

        category = categorize_announcement(subject, detailed_text)
        
        # Drop noise unless it caught a specific bucket
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

df_new = pd.DataFrame(filtered_announcements)
if df_new.empty:
    print("No relevant announcements found in the last 15 days for your universe.")
    raise SystemExit(0)
    
df_new = df_new[desired_columns]

# ==========================================
# 7. MERGE, DROP DUPLICATES, AND APPEND
# ==========================================
print("Merging with historical data and preventing duplicates...")
# Concat new data ON TOP of old data. Old data is perfectly preserved.
df_final = pd.concat([df_new, existing_df])

# Drop identical announcements so they don't print twice
df_final = df_final.drop_duplicates(subset=['Date', 'NSE Code', 'Category', 'Announcement'])

# Sort by Date
df_final['Date_Sort'] = pd.to_datetime(df_final['Date'], format='mixed', dayfirst=True, errors='coerce')
df_final = df_final.sort_values(by='Date_Sort', ascending=False).drop(columns=['Date_Sort'])
df_final['Date'] = pd.to_datetime(df_final['Date'], format='mixed', dayfirst=True, errors='coerce').dt.strftime('%d-%b-%Y')

# Limit history to 5,000 rows to keep Google Sheets fast
df_final = df_final.head(5000)

print(f"Writing {len(df_final)} total records to Google Sheets...")
sheet_output = [df_final.columns.values.tolist()] + df_final.fillna("").values.tolist()

worksheet.clear() 
worksheet.update(values=sheet_output, range_name='A1', value_input_option='USER_ENTERED')
worksheet.freeze(rows=1)
worksheet.format('A1:G1', {
    "backgroundColor": {"red": 0.1, "green": 0.2, "blue": 0.4},
    "textFormat": {"foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}, "bold": True}
})

print("✅ Corporate Announcements Database successfully updated!")
