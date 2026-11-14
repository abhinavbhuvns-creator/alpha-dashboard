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
creds_json = os.environ.get('GOOGLE_CREDENTIALS')
if not creds_json:
    raise SystemExit("Error: GOOGLE_CREDENTIALS not found in GitHub Secrets.")

creds_dict = json.loads(creds_json)
scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
gc = gspread.authorize(creds)

SHEET_URL = 'https://docs.google.com/spreadsheets/d/1A2fUfXGKXXQxzFnoR30cFVqtmb-28KTi4fR4N0e507g/edit?usp=sharing'
spreadsheet = gc.open_by_url(SHEET_URL)
TAB_NAME = "Corporate_Announcements"

# ==========================================
# 2. GET STOCK UNIVERSE (> 3 CRORE TURNOVER FILTER)
# ==========================================
try:
    master_ws = spreadsheet.worksheet("Avg_Rupee_Volume_Master")
    master_records = master_ws.get_all_records()
    universe_dict = {}

    for row in master_records:
        sym = str(row.get('Symbol', '')).replace('.NS', '').strip().upper()
        if not sym:
            continue

        # Parse Average Rupee Turnover / Volume
        turnover_val = 0.0
        for k, v in row.items():
            if any(w in k.lower() for w in ['rupee', 'turnover', 'volume', 'value']):
                try:
                    val_str = str(v).replace(',', '').replace('Cr', '').replace('₹', '').strip()
                    turnover_val = float(val_str)
                    break
                except Exception:
                    pass

        # Normalize to absolute INR if stored in Crores (e.g. 3.5 -> 35,000,000)
        if 0 < turnover_val < 100000:
            turnover_val = turnover_val * 10000000

        # Strict > 3 Crore (30,000,000 INR) Filter
        if turnover_val >= 30000000:
            universe_dict[sym] = {
                'Company Name': row.get('Company Name', sym),
                'Industry': row.get('Industry Group', '') or row.get('Industry', 'Unclassified')
            }

    print(f"Loaded {len(universe_dict)} liquid stocks (> 3 Cr turnover) from Avg_Rupee_Volume_Master.", flush=True)
except Exception as e:
    raise SystemExit(f"Error reading Master Data: {e}")

approved_symbols = set(universe_dict.keys())

# ==========================================
# 3. ADVANCED CATEGORIZATION & NOISE FILTER
# ==========================================
NON_MATERIAL_NOISE = [
    "trading window", "closure of trading window", "loss of share", "issue of duplicate",
    "duplicate share", "share certificate", "iepf", "unclaimed dividend",
    "newspaper", "regulation 47", "regulation 74(5)", "regulation 30(5)",
    "scrutinizer", "postal ballot notice", "intimation of schedule", "esop allotment",
    "significant movement in price", "significant increase in volume", "clarification on price"
]

def is_non_material_noise(subject, desc):
    text = (str(subject) + " " + str(desc)).lower()
    return any(noise in text for noise in NON_MATERIAL_NOISE)

def categorize_announcement(subject, desc):
    text = (str(subject) + " " + str(desc)).lower()

    # PRIORITY 1: Legal & Court
    legal_keywords = [
        "court", "tribunal", "litigation", "nclt", "central bureau of investigation",
        "enforcement directorate", "income tax", "penalty", "show cause", "default", "fraud",
        "forensic", "summons", "writ petition", "orders passed", "order dated",
        "hon'ble", "supreme court", "high court", "order in favour", "order-in-original"
    ]
    if any(kw in text for kw in legal_keywords):
        return "Legal / Red Flags"

    # PRIORITY 2: Core Business Catalysts
    if any(kw in text for kw in ["financial result", "audited", "unaudited", "earnings"]):
        return "Financial Results"

    if any(kw in text for kw in ["new product", "product launch", "commercial launch", "patent", "marketing authorisation", "fda approval", "usfda", "ce mark"]):
        return "New Product / PATENT"

    clean_text_for_orders = (
        text.replace("in order to", "")
        .replace("order passed", "")
        .replace("order dated", "")
        .replace("court order", "")
        .replace("order-in-original", "")
        .replace("order of sebi", "")
        .replace("order of any", "")
        .replace("order of final hearing", "")
        .replace("scheme of amalgamation", "")
    )

    order_keywords = [
        "order", "orders", "bagged", "bagging", "bags",
        "winning", "wins", "awarded", "awarding",
        "letter of award", "loi", "letter of intent",
        "letter of acceptance", "loa",
        "execution of contract", "revenue sharing contract", "rate contract",
        "turnkey contract", "epc contract", "work order", "order book",
        "secures engagement"
    ]
    if any(kw in clean_text_for_orders for kw in order_keywords):
        return "Order Wins"

    if any(kw in text for kw in ["capex", "capacity", "expansion", "commissioning", "new plant", "new unit", "commercial production", "new venture", "sets up"]):
        return "Capacity Expansion or New Ventures"

    if any(kw in text for kw in ["fund raising", "preferential issue", "qip", "rights issue", "private placement"]):
        return "Fund Raising"

    if any(kw in text for kw in ["acquisition", "takeover", "merger", "amalgamation", "stake sale", "buyout"]):
        return "Acquisitions"

    # PRIORITY 3: Standard Corporate Actions & Updates
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
# 4. INITIALIZE SESSION & READ EXISTING SHEET
# ==========================================
try:
    worksheet = spreadsheet.worksheet(TAB_NAME)
    existing_data = worksheet.get_all_values()
except gspread.exceptions.WorksheetNotFound:
    worksheet = spreadsheet.add_worksheet(title=TAB_NAME, rows="15000", cols="7")
    existing_data = []

desired_columns = ["Date", "Industry", "Category", "Company Name", "NSE Code", "Announcement", "PDF Link"]

if len(existing_data) > 1:
    existing_df = pd.DataFrame(existing_data[1:], columns=existing_data[0])
    for col in desired_columns:
        if col not in existing_df.columns:
            existing_df[col] = ""
    existing_df = existing_df[desired_columns]
    # Filter existing rows to enforce > 3 Cr universe and strip noise
    existing_df = existing_df[existing_df['NSE Code'].str.strip().str.upper().isin(approved_symbols)]
    existing_df = existing_df[~existing_df.apply(lambda r: is_non_material_noise(r['Category'], r['Announcement']), axis=1)]
else:
    existing_df = pd.DataFrame(columns=desired_columns)

# ==========================================
# 5. SMART BACKFILL FROM 12-OCT-2026 OR DAILY FETCH
# ==========================================
target_backfill_start = datetime(2026, 10, 12)
today_dt = datetime.today()

need_oct12_backfill = True
if not existing_df.empty:
    parsed_existing_dates = pd.to_datetime(existing_df['Date'], format='mixed', dayfirst=True, errors='coerce').dropna()
    if not parsed_existing_dates.empty and parsed_existing_dates.min() <= datetime(2026, 10, 15):
        need_oct12_backfill = False

fetch_Start_bound = target_backfill_start if need_oct12_backfill else (today_dt - timedelta(days=10))
print(f"Fetching NSE Announcements from {fetch_Start_bound.strftime('%d-%b-%Y')} to {today_dt.strftime('%d-%b-%Y')}...", flush=True)

fetch_ranges = []
curr_end = today_dt
while curr_end >= fetch_Start_bound:
    curr_start = max(fetch_Start_bound, curr_end - timedelta(days=4))
    fetch_ranges.append((curr_start.strftime("%d-%m-%Y"), curr_end.strftime("%d-%m-%Y")))
    curr_end = curr_start - timedelta(days=1)

session = requests.Session()
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.9'
}
session.get("https://www.nseindia.com", headers=headers, timeout=15)
time.sleep(2)

raw_data = []
for start_str, end_str in fetch_ranges:
    api_url = f"https://www.nseindia.com/api/corporate-announcements?index=equities&from_date={start_str}&to_date={end_str}"
    print(f" -> Fetching chunk: {start_str} to {end_str}", flush=True)
    for attempt in range(3):
        try:
            response = session.get(api_url, headers=headers, timeout=20)
            if response.status_code == 200:
                chunk = response.json()
                if isinstance(chunk, list):
                    raw_data.extend(chunk)
                break
        except Exception:
            pass
        time.sleep(2)
    time.sleep(1)

# ==========================================
# 6. FILTER & CATEGORIZE
# ==========================================
filtered_announcements = []

for row in raw_data:
    row_lower = {str(k).lower(): v for k, v in row.items()}
    symbol = str(row_lower.get('symbol', '')).replace('.NS', '').strip().upper()

    if symbol not in approved_symbols:
        continue

    subject = str(row_lower.get('desc', '')).strip()
    if subject.lower() in ['none', 'nan', 'null']:
        subject = ""
    detailed_text = str(row_lower.get('attchmnttext', '')).strip()
    if detailed_text.lower() in ['none', 'nan', 'null']:
        detailed_text = ""

    # Drop non-material noise (trading window closures, share certificates, newspaper ads)
    if is_non_material_noise(subject, detailed_text):
        continue

    category = categorize_announcement(subject, detailed_text)
    if category == "Other Important Announcements":
        continue

    raw_date = str(row_lower.get('an_dt', '') or row_lower.get('broadcastdate', '') or row_lower.get('date', '')).strip()
    if raw_date.lower() in ['none', 'nan', 'null']:
        raw_date = ""
    clean_date = raw_date.split(' ')[0] if raw_date else ""

    file_path = str(row_lower.get('attchmntfile', '')).strip()
    if file_path.lower() in ['none', 'nan', 'null']:
        file_path = ""
    link = ""
    if file_path and len(file_path) > 3:
        if file_path.startswith('http'):
            link = file_path
        elif file_path.startswith('/'):
            link = f"https://nsearchives.nseindia.com{file_path}"
        else:
            link = f"https://nsearchives.nseindia.com/{file_path}"

    final_announcement = detailed_text if len(detailed_text) > len(subject) else subject

    filtered_announcements.append({
        "Date": clean_date,
        "Industry": universe_dict[symbol]['Industry'],
        "Category": category,
        "Company Name": universe_dict[symbol]['Company Name'],
        "NSE Code": symbol,
        "Announcement": final_announcement,
        "PDF Link": link
    })

df_new = pd.DataFrame(filtered_announcements, columns=desired_columns) if filtered_announcements else pd.DataFrame(columns=desired_columns)

# ==========================================
# 7. MERGE, DEDUPLICATE, & 1-YEAR ROLLING RETENTION
# ==========================================
df_final = pd.concat([df_new, existing_df], ignore_index=True)

# Deduplicate by PDF Link when present, otherwise by Date + NSE Code + Category + Announcement
has_pdf = df_final['PDF Link'].astype(str).str.strip() != ""
df_with_pdf = df_final[has_pdf].drop_duplicates(subset=['NSE Code', 'PDF Link'], keep='first')
df_no_pdf = df_final[~has_pdf].drop_duplicates(subset=['Date', 'NSE Code', 'Category', 'Announcement'], keep='first')
df_final = pd.concat([df_with_pdf, df_no_pdf], ignore_index=True)

df_final['Date_Sort'] = pd.to_datetime(df_final['Date'], format='mixed', dayfirst=True, errors='coerce')

# Enforce 1-Year (365-Day) Rolling Purge
one_year_ago = datetime.today() - timedelta(days=365)
df_final = df_final[df_final['Date_Sort'].isna() | (df_final['Date_Sort'] >= one_year_ago)]

df_final = df_final.sort_values(by='Date_Sort', ascending=False)
df_final['Date'] = df_final['Date_Sort'].dt.strftime('%d-%b-%Y').fillna(df_final['Date'])
df_final = df_final.drop(columns=['Date_Sort']).head(15000)

sheet_output = [df_final.columns.values.tolist()] + df_final.fillna("").values.tolist()

worksheet.clear()
worksheet.update(values=sheet_output, range_name='A1', value_input_option='USER_ENTERED')
worksheet.freeze(rows=1)
worksheet.format('A1:G1', {
    "backgroundColor": {"red": 0.1, "green": 0.2, "blue": 0.4},
    "textFormat": {"foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}, "bold": True}
})
print(f"Saved {len(df_final)} material announcements (1-year rolling window).", flush=True)


