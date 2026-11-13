import os
import json
import time
from datetime import datetime, timedelta
import requests
import pymupdf  # PyMuPDF
import gspread
from google.oauth2.service_account import Credentials

SINGLE_SHEET_URL = 'https://docs.google.com/spreadsheets/d/1A2fUfXGKXXQxzFnoR30cFVqtmb-28KTi4fR4N0e507g/edit?usp=sharing'

# Active Free-Tier Models
LITE_MODEL = "gemini-3.5-flash-lite"      # 500 RPD Free Tier (Scout + Lite Summary)
PREMIUM_MODEL = "gemini-3.8-flash"        # 20 RPD Free Tier (Full 8-Point Deep Dive)

HEADERS = [
    "Date", "Symbol", "Company", "Verdict", "Summary",
    "Growth Drivers", "Growth", "Capacity", "Order Book",
    "Margins", "Risks", "Other", "PDF Link", "Model Tier"
]

def parse_flexible_date(date_str):
    """Parses various date formats into a datetime object."""
    if not date_str:
        return None
    s = str(date_str).strip()
    if "T" in s:
        s = s.split("T")[0]
    if " " in s:
        s = s.split(" ")[0]
        
    formats = [
        "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d",
        "%d-%b-%Y", "%d %b %Y", "%b %d, %Y", "%d-%B-%Y"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None

def download_and_extract_pdf(pdf_url):
    """Downloads the transcript PDF into memory and extracts all text with a strict timeout."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(pdf_url, headers=headers, timeout=25)
    response.raise_for_status()
    
    with open("temp_transcript.pdf", "wb") as f:
        f.write(response.content)
        
    doc = pymupdf.open("temp_transcript.pdf")
    full_text = "\n".join([page.get_text() for page in doc])
    doc.close()
    if os.path.exists("temp_transcript.pdf"):
        os.remove("temp_transcript.pdf")
    return full_text

def run_scout_and_lite_summary(transcript_text, company_name, api_key):
    """
    Pass 1 (Gemini 3.5 Flash-Lite):
    Determines the strict Verdict AND writes a concise summary for Steady/Mixed/Weak calls.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{LITE_MODEL}:generateContent?key={api_key}"
    
    prompt = f"""
    You are a ruthless quantitative equity research assistant evaluating the earnings call transcript for {company_name}.
    Ignore standard corporate pleasantries. Base your evaluation strictly on forward-looking statements, guidance revisions, order book health, and margin trajectories.
    
    Classify the transcript into EXACTLY ONE of the following four verdict categories:
    - Confident: Management explicitly raises revenue/EPS guidance, reports a surging order book/backlog, notes expanding margins, announces aggressive growth CAPEX, or states demand is outstripping supply.
    - Steady: Guidance is maintained. Execution is inline with past quarters. Demand is stable.
    - Mixed: Conflicting signals (e.g., revenue growing but margins compressing; strong domestic demand but weak exports).
    - Weak: Management explicitly lowers guidance, reports shrinking margins, delays CAPEX, or notes rising inventory/canceled orders.
    
    Return strictly a JSON object matching this schema:
    {{
        "verdict": "Choose strictly one: Confident, Steady, Mixed, or Weak",
        "summary": "2-3 sentence executive summary highlighting revenue growth, EBITDA/PAT margins, and forward guidance. Bold key numbers and percentages."
    }}
    
    Transcript:
    {transcript_text[:40000]}
    """
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"}
    }
    headers = {'Content-Type': 'application/json'}
    
    response = requests.post(url, headers=headers, json=payload, timeout=35)
    if response.status_code != 200:
        raise Exception(f"Lite Model API Failed ({response.status_code}): {response.text}")
        
    result_data = response.json()
    raw_json_text = result_data['candidates'][0]['content']['parts'][0]['text']
    parsed = json.loads(raw_json_text)
    
    verdict = str(parsed.get("verdict", "Mixed")).strip().capitalize()
    if verdict not in ["Confident", "Steady", "Mixed", "Weak"]:
        verdict = "Mixed"
    summary = str(parsed.get("summary", "")).strip()
    return verdict, summary

def run_premium_deep_dive(transcript_text, company_name, verdict, api_key):
    """
    Pass 2 (Gemini 3.8 Flash):
    Executes the full 8-point institutional JSON extraction.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{PREMIUM_MODEL}:generateContent?key={api_key}"
    
    prompt = f"""
    You are an expert institutional equity research analyst. Analyze the following earnings call transcript for {company_name}.
    Extract the requested information strictly as a JSON object matching the schema below. Keep bullet points concise and bold key numbers/percentages.
    
    JSON Schema:
    {{
        "verdict": "{verdict}",
        "summary": "1-2 sentence TL;DR on revenue growth, EPS, and forward guidance",
        "growth_drivers": ["bullet 1", "bullet 2"],
        "growth": ["bullet 1", "bullet 2"],
        "capacity": ["bullet 1", "bullet 2"],
        "order_book": ["bullet 1", "bullet 2"],
        "margins": ["bullet 1", "bullet 2"],
        "risks_and_red_flags": ["bullet 1", "bullet 2"],
        "other_points": ["bullet 1", "bullet 2"]
    }}
    
    Transcript:
    {transcript_text[:150000]}
    """
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"}
    }
    headers = {'Content-Type': 'application/json'}
    
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    if response.status_code != 200:
        raise Exception(f"Premium Model API Failed ({response.status_code}): {response.text}")
        
    result_data = response.json()
    raw_json_text = result_data['candidates'][0]['content']['parts'][0]['text']
    return json.loads(raw_json_text)

def format_bullets(items):
    if not items or not isinstance(items, list):
        return ""
    cleaned = [str(i).strip() for i in items if str(i).strip()]
    return "• " + "\n• ".join(cleaned) if cleaned else ""

def build_row_from_premium(date, symbol, company, fallback_verdict, ai_data, pdf_link):
    return [
        date,
        symbol,
        company,
        ai_data.get("verdict", fallback_verdict),
        ai_data.get("summary", ""),
        format_bullets(ai_data.get("growth_drivers", [])),
        format_bullets(ai_data.get("growth", [])),
        format_bullets(ai_data.get("capacity", [])),
        format_bullets(ai_data.get("order_book", [])),
        format_bullets(ai_data.get("margins", [])),
        format_bullets(ai_data.get("risks_and_red_flags", [])),
        format_bullets(ai_data.get("other_points", [])),
        pdf_link,
        "Premium"
    ]

def build_row_from_lite(date, symbol, company, verdict, summary, pdf_link):
    return [
        date, symbol, company, verdict, summary,
        "", "", "", "", "", "", "",
        pdf_link, "Lite"
    ]

def is_quota_error(err_msg):
    msg = str(err_msg).lower()
    return "429" in msg or "quota" in msg or "resource_exhausted" in msg

def main():
    print("Authenticating with Google Service Account...", flush=True)
    creds_json = os.environ.get('GOOGLE_CREDENTIALS')
    api_key = os.environ.get("GEMINI_API_KEY", "").strip(' "\'\n\r\t')
    
    if not creds_json or not api_key:
        raise SystemExit("🛑 CRITICAL ERROR: GOOGLE_CREDENTIALS or GEMINI_API_KEY missing in GitHub Secrets.")
    
    creds_dict = json.loads(creds_json)
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    gc = gspread.authorize(creds)
    spreadsheet = gc.open_by_url(SINGLE_SHEET_URL)
    
    one_year_cutoff = datetime.now() - timedelta(days=365)

    # 1. Build Liquidity Map (> 5 Crore Average Rupee Turnover Filter)
    print("Building 5 Crore liquidity map from Avg_Rupee_Volume_Master...", flush=True)
    liquidity_map = {}
    try:
        vol_ws = spreadsheet.worksheet("Avg_Rupee_Volume_Master")
        vol_data = vol_ws.get_all_records()
        for r in vol_data:
            sym = str(r.get("Symbol", "")).replace(".NS", "").strip().upper()
            if not sym:
                continue
            val = 0.0
            for k, v in r.items():
                if any(w in k.lower() for w in ['rupee', 'turnover', 'volume', 'value']):
                    try:
                        val_str = str(v).replace(',', '').replace('Cr', '').replace('₹', '').strip()
                        val = float(val_str)
                        break
                    except:
                        pass
            # Normalize to absolute INR if stored in Crores (e.g. 5.5 -> 55,000,000)
            if 0 < val < 100000:
                val = val * 10000000
            liquidity_map[sym] = val
    except Exception as e:
        print(f"⚠️ Warning: Could not load Avg_Rupee_Volume_Master: {e}", flush=True)

    # 2. Load Concall_Analysis & Perform 1-Year Rolling Purge
    print("Checking Concall_Analysis & purging records older than 1 year...", flush=True)
    try:
        analysis_ws = spreadsheet.worksheet("Concall_Analysis")
        all_values = analysis_ws.get_all_values()
    except gspread.exceptions.WorksheetNotFound:
        analysis_ws = spreadsheet.add_worksheet(title="Concall_Analysis", rows="1000", cols="15")
        analysis_ws.update(values=[HEADERS], range_name='A1:N1')
        analysis_ws.format('A1:N1', {'textFormat': {'bold': True}, "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9}})
        all_values = [HEADERS]

    kept_rows = [HEADERS]
    purged_count = 0
    if len(all_values) > 1:
        for raw_row in all_values[1:]:
            if not any(raw_row):
                continue
            # Pad to 14 columns if needed
            row = (raw_row + [""] * 14)[:14]
            row_date = parse_flexible_date(row[0])
            
            # Purge if older than 365 days
            if row_date and row_date < one_year_cutoff:
                purged_count += 1
                continue
                
            # Backfill Model Tier for legacy rows
            if not row[13].strip():
                has_deep_dive = any(row[idx].strip() for idx in range(5, 12))
                row[13] = "Premium" if has_deep_dive else "Lite"
                
            kept_rows.append(row)

    if purged_count > 0 or (all_values and len(all_values[0]) < 14):
        print(f" -> Purged {purged_count} rows older than 1 year. Syncing sheet...", flush=True)
        analysis_ws.clear()
        analysis_ws.update(values=kept_rows, range_name=f"A1:N{len(kept_rows)}")
        analysis_ws.format('A1:N1', {'textFormat': {'bold': True}, "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9}})

    processed_pdfs = set(str(r[12]).strip() for r in kept_rows[1:] if len(r) > 12 and str(r[12]).strip())

    # 3. Scan Corporate_Announcements for New Transcripts (Within Last 1 Year & >= 5 Cr Turnover)
    print("Reading Corporate_Announcements...", flush=True)
    corp_ws = spreadsheet.worksheet("Corporate_Announcements")
    corp_data = corp_ws.get_all_records()

    premium_quota_exhausted = False
    lite_quota_exhausted = False
    new_processed_count = 0

    print("Scanning for new Earnings Call Transcripts...", flush=True)
    for row in corp_data:
        announcement = str(row.get("Announcement", "")).lower()
        pdf_link = str(row.get("PDF Link", "")).strip()
        company = str(row.get("Company Name", "")).strip()
        symbol = str(row.get("NSE Code", "")).replace(".NS", "").strip().upper()
        date_str = str(row.get("Date", "")).strip()

        is_transcript = any(kw in announcement for kw in ["transcript", "earnings call transcript", "concall transcript", "call transcript"])
        if not (is_transcript and pdf_link and pdf_link.endswith(".pdf") and pdf_link not in processed_pdfs):
            continue

        # 1-Year Date Check
        filing_date = parse_flexible_date(date_str)
        if filing_date and filing_date < one_year_cutoff:
            continue

        # 5 Crore Average Rupee Volume Filter
        turnover = liquidity_map.get(symbol, 0)
        if liquidity_map and turnover < 50000000:
            print(f" -> Skipping {symbol}: Turnover < 5 Cr", flush=True)
            processed_pdfs.add(pdf_link)
            continue

        if lite_quota_exhausted:
            print("🛑 Lite Model daily quota exhausted. Stopping new transcript scans for today.", flush=True)
            break

        print(f" -> Scouting new transcript: {company} ({symbol})...", flush=True)
        try:
            text = download_and_extract_pdf(pdf_link)
            verdict, lite_summary = run_scout_and_lite_summary(text, company, api_key)
            print(f"    [Scout: {LITE_MODEL}] Verdict: {verdict}", flush=True)
            time.sleep(2)

            if verdict == "Confident" and not premium_quota_exhausted:
                try:
                    print(f"    [Deep Dive: {PREMIUM_MODEL}] Extracting 8-point analysis for {symbol}...", flush=True)
                    ai_data = run_premium_deep_dive(text, company, verdict, api_key)
                    new_row = build_row_from_premium(date_str, symbol, company, verdict, ai_data, pdf_link)
                    analysis_ws.append_row(new_row)
                    kept_rows.append(new_row)
                    processed_pdfs.add(pdf_link)
                    new_processed_count += 1
                    print(f"    ✅ Saved PREMIUM analysis for {symbol}", flush=True)
                    time.sleep(4)
                except Exception as pe:
                    if is_quota_error(pe):
                        print(f"    ⚠️ {PREMIUM_MODEL} 20/day quota reached! Saving Lite summary for {symbol} to upgrade tomorrow.", flush=True)
                        premium_quota_exhausted = True
                        lite_row = build_row_from_lite(date_str, symbol, company, verdict, lite_summary, pdf_link)
                        analysis_ws.append_row(lite_row)
                        kept_rows.append(lite_row)
                        processed_pdfs.add(pdf_link)
                        new_processed_count += 1
                    else:
                        raise pe
            else:
                # Steady, Mixed, Weak (or Confident when Premium quota is exhausted) -> Save Lite summary immediately
                lite_row = build_row_from_lite(date_str, symbol, company, verdict, lite_summary, pdf_link)
                analysis_ws.append_row(lite_row)
                kept_rows.append(lite_row)
                processed_pdfs.add(pdf_link)
                new_processed_count += 1
                print(f"    ✅ Saved LITE summary for {symbol} ({verdict})", flush=True)

        except requests.exceptions.Timeout:
            print(f"⚠️ Timeout processing {symbol}. Skipping for now.", flush=True)
        except Exception as e:
            if is_quota_error(e):
                print(f"🛑 Lite Model quota exceeded: {e}", flush=True)
                lite_quota_exhausted = True
                break
            print(f"⚠️ Error processing {symbol}: {e}", flush=True)

    # 4. Backlog & Idle Upgrade Protocol: Upgrade 'Lite' Rows to 'Premium'
    # Strict Priority Order: Confident (Backlog) -> Steady -> Mixed -> Weak
    if not premium_quota_exhausted:
        print("\nChecking Backlog & Idle Upgrade Queue (Confident -> Steady -> Mixed -> Weak)...", flush=True)
        priority_map = {"confident": 0, "steady": 1, "mixed": 2, "weak": 3}
        upgrade_candidates = []

        for sheet_row_idx in range(1, len(kept_rows)):
            r = kept_rows[sheet_row_idx]
            tier = str(r[13]).strip().lower() if len(r) > 13 else "lite"
            if tier == "lite":
                sym = str(r[1]).replace(".NS", "").strip().upper()
                # Respect the 5 Cr liquidity filter during upgrades too
                if liquidity_map and liquidity_map.get(sym, 0) < 50000000:
                    continue
                verd = str(r[3]).strip().lower()
                prio = priority_map.get(verd, 4)
                dt = parse_flexible_date(r[0]) or datetime(2000, 1, 1)
                # Sort by priority ascending (0=Confident, 1=Steady, 2=Mixed, 3=Weak), then newest date first
                upgrade_candidates.append((prio, -dt.timestamp(), sheet_row_idx, r))

        upgrade_candidates.sort(key=lambda x: (x[0], x[1]))

        if not upgrade_candidates:
            print(" -> No Lite rows waiting for Premium upgrade. Database is 100% Premium!", flush=True)
        else:
            print(f" -> Found {len(upgrade_candidates)} Lite rows eligible for Premium upgrade.", flush=True)
            for prio, _, sheet_row_idx, r in upgrade_candidates:
                date_str, symbol, company, verdict, _, _, _, _, _, _, _, _, pdf_link, _ = r
                sheet_line_number = sheet_row_idx + 1  # 1-indexed in Google Sheets
                
                print(f" -> Upgrading [{verdict}] {company} ({symbol}) at Row {sheet_line_number} to Premium...", flush=True)
                try:
                    text = download_and_extract_pdf(pdf_link)
                    ai_data = run_premium_deep_dive(text, company, verdict, api_key)
                    upgraded_row = build_row_from_premium(date_str, symbol, company, verdict, ai_data, pdf_link)
                    
                    # Overwrite the Lite row in-place in Google Sheets
                    analysis_ws.update(values=[upgraded_row], range_name=f"A{sheet_line_number}:N{sheet_line_number}")
                    kept_rows[sheet_row_idx] = upgraded_row
                    print(f"    ✅ Upgraded {symbol} ({verdict}) from Lite to Premium!", flush=True)
                    time.sleep(4)
                except requests.exceptions.Timeout:
                    print(f"    ⚠️ Timeout downloading/upgrading {symbol}. Moving to next.", flush=True)
                except Exception as e:
                    if is_quota_error(e):
                        print(f"🛑 {PREMIUM_MODEL} daily quota reached during upgrades. Stopping cleanly for today!", flush=True)
                        break
                    print(f"    ⚠️ Failed to upgrade {symbol}: {e}", flush=True)

    print("\n🎉 Concall Parser run complete!", flush=True)

if __name__ == "__main__":
    main()
