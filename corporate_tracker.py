import os
import json
import time
import requests
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta

print("Authenticating with Google Service Account...")
creds_dict = json.loads(os.environ.get('GOOGLE_CREDENTIALS'))
gc = gspread.authorize(Credentials.from_service_account_info(creds_dict, scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]))
spreadsheet = gc.open_by_url('https://docs.google.com/spreadsheets/d/1A2fUfXGKXXQxzFnoR30cFVqtmb-28KTi4fR4N0e507g/edit?usp=sharing')
TAB_NAME = "Corporate_Announcements"

print("Fetching approved stock universe...")
master_df = pd.DataFrame(spreadsheet.worksheet("Avg_Rupee_Volume_Master").get_all_records())
universe_dict = {str(row.get('Symbol', '')).strip().upper(): {'Company Name': row.get('Company Name', ''), 'Industry': row.get('Industry Group', 'Unclassified')} for _, row in master_df.iterrows() if str(row.get('Symbol', '')).strip()}
approved_symbols = list(universe_dict.keys())

def categorize_announcement(subject, desc):
    text = (str(subject) + " " + str(desc)).lower()
    
    # Priority 1: Legal (Strict full phrasing to avoid false positives)
    if any(kw in text for kw in ["court", "tribunal", "litigation", "nclt", "central bureau of investigation", "enforcement directorate", "income tax", "penalty", "show cause", "default", "fraud", "forensic", "summons", "writ petition", "order passed", "order dated", "hon'ble", "supreme court", "high court"]):
        return "Legal / Red Flags"
        
    # Priority 2: Catalysts
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

    # Priority 3: Routine Updates
    if any(kw in text for kw in ["concall", "analyst meet", "earnings call", "investor presentation", "transcript", "audio recording"]): return "Concall / Analyst Meet"
    if any(kw in text for kw in ["credit rating", "crisil", "icra", "care", "fitch", "rating downgrade", "rating upgrade"]): return "Credit Rating"
    if any(kw in text for kw in ["bonus", "split", "sub-division", "allotment", "buyback"]): return "Corporate Action"
    if any(kw in text for kw in ["dividend", "record date"]): return "Dividend"
    if any(kw in text for kw in ["shareholding", "pledge", "promoter", "block deal"]): return "Shareholding"
    if any(kw in text for kw in ["appointment", "resignation", "cessation", "change in director", "change in management", "kmp"]): return "Appointments / Resignations"
    if "board meeting" in text: return "Board Meeting"
    if any(kw in text for kw in ["annual report", "agm", "egm", "annual general meeting"]): return "Annual Report"

    return "Other Important Announcements"

try:
    worksheet = spreadsheet.worksheet(TAB_NAME)
    existing_data = worksheet.get_all_values()
    is_empty = len(existing_data) <= 1
except gspread.exceptions.WorksheetNotFound:
    worksheet = spreadsheet.add_worksheet(title=TAB_NAME, rows="5000", cols="7")
    existing_data = []
    is_empty = True

fetch_ranges = []
if is_empty:
    print("Initiating 6-month Market-Wide Backfill (7-day chunks)...")
    end_dt = datetime.today()
    for _ in range(26): # 26 weeks = 6 months
        start_dt = end_dt - timedelta(days=7)
        fetch_ranges.append((start_dt.strftime("%d-%m-%Y"), end_dt.strftime("%d-%m-%Y")))
        end_dt = start_dt - timedelta(days=1)
else:
    print("Fetching daily updates...")
    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=3)
    fetch_ranges.append((start_dt.strftime("%d-%m-%Y"), end_dt.strftime("%d-%m-%Y")))

session = requests.Session()
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)', 'Accept': '*/*'}
session.get("https://www.nseindia.com", headers=headers, timeout=10)
time.sleep(2)

raw_data = []
for start_str, end_str in fetch_ranges:
    print(f" -> Fetching {start_str} to {end_str}")
    for _ in range(3):
        try:
            res = session.get(f"https://www.nseindia.com/api/corporate-announcements?index=equities&from_date={start_str}&to_date={end_str}", headers=headers, timeout=15)
            if res.status_code == 200:
                raw_data.extend(res.json())
                break
        except: pass
        time.sleep(2)
    time.sleep(1)

filtered_announcements = []
for row in raw_data:
    row_lower = {str(k).lower(): v for k, v in row.items()}
    symbol = str(row_lower.get('symbol', '')).strip().upper()
    
    if symbol in approved_symbols:
        raw_date = str(row_lower.get('an_dt', '') or row_lower.get('broadcastdate', '')).strip()
        clean_date = raw_date.split(' ')[0] if raw_date.lower() not in ['none', 'nan', 'null', ''] else ""
        
        file_path = str(row_lower.get('attchmntfile', '')).strip()
        link = f"https://nsearchives.nseindia.com{file_path}" if file_path and file_path.startswith('/') else (f"https://nsearchives.nseindia.com/{file_path}" if file_path else "")
        if file_path.startswith('http'): link = file_path

        subject = str(row_lower.get('desc', '')).strip()
        detailed_text = str(row_lower.get('attchmnttext', '')).strip()
        final_text = detailed_text if len(detailed_text) > len(subject) else subject

        category = categorize_announcement(subject, detailed_text)
        if category != "Other Important Announcements" or ("update" not in detailed_text.lower() and "loss of share" not in detailed_text.lower()):
            filtered_announcements.append({"Date": clean_date, "Industry": universe_dict[symbol]['Industry'], "Category": category, "Company Name": universe_dict[symbol]['Company Name'], "NSE Code": symbol, "Announcement": final_text, "PDF Link": link})

df_new = pd.DataFrame(filtered_announcements).drop_duplicates()
cols = ["Date", "Industry", "Category", "Company Name", "NSE Code", "Announcement", "PDF Link"]
if df_new.empty: raise SystemExit("No relevant data found.")
df_new = df_new[cols]

if not is_empty:
    existing_df = pd.DataFrame(existing_data[1:], columns=existing_data[0])
    df_final = pd.concat([df_new, existing_df]).drop_duplicates(subset=['Date', 'NSE Code', 'Category', 'Announcement'])
else:
    df_final = df_new

df_final['Date'] = pd.to_datetime(df_final['Date'], format='mixed', dayfirst=True, errors='coerce')
df_final = df_final.sort_values(by='Date', ascending=False).head(5000)
df_final['Date'] = df_final['Date'].dt.strftime('%d-%b-%Y')

worksheet.clear()
worksheet.update(values=[df_final.columns.values.tolist()] + df_final.fillna("").values.tolist(), range_name='A1', value_input_option='USER_ENTERED')
worksheet.freeze(rows=1)
worksheet.format('A1:G1', {"backgroundColor": {"red": 0.1, "green": 0.2, "blue": 0.4}, "textFormat": {"foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}, "bold": True}})
print("✅ Corporate Announcements updated successfully!")
