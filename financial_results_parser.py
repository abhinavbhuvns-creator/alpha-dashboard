import os
import json
import time
from datetime import datetime, timedelta
import requests
import pymupdf  # PyMuPDF
import gspread
from google.oauth2.service_account import Credentials

SINGLE_SHEET_URL = 'https://docs.google.com/spreadsheets/d/1A2fUfXGKXXQxzFnoR30cFVqtmb-28KTi4fR4N0e507g/edit?usp=sharing'

# Model Hierarchy
SCOUT_LITE_MODELS = ["gemini-3.1-flash-lite", "gemini-2.5-flash-lite"]
PREMIUM_MODELS = ["gemini-3.6-flash", "gemini-2.5-flash"]

HEADERS = [
    "Date", "Symbol", "Company", "Industry", "Verdict", "Turnaround", "Result Type",
    "Curr Qtr", "Prev Qtr", "YoY Qtr",
    "Rev YoY %", "Rev QoQ %", "Rev Curr", "Rev Prev", "Rev YoY",
    "OP YoY %", "OP QoQ %", "OP Curr", "OP Prev", "OP YoY",
    "OPM Curr %", "OPM Prev %", "OPM YoY %",
    "PAT YoY %", "PAT QoQ %", "PAT Curr", "PAT Prev", "PAT YoY",
    "EPS YoY %", "EPS QoQ %", "EPS Curr", "EPS Prev", "EPS YoY",
    "PDF Link", "Model Tier"
]

def parse_flexible_date(date_str):
    """Parses various date formats into a datetime object."""
    if not date_str:
        return None
    s = str(date_str).strip()
    if "T" in s:
        s = s.split("T")[0]
    if " " in s and not any(m in s for m in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]):
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
    """Downloads the financial result PDF and extracts text with a strict 25s timeout."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(pdf_url, headers=headers, timeout=25)
    response.raise_for_status()
    
    temp_file = "temp_result.pdf"
    with open(temp_file, "wb") as f:
        f.write(response.content)
        
    doc = pymupdf.open(temp_file)
    full_text = "\n".join([page.get_text() for page in doc])
    doc.close()
    if os.path.exists(temp_file):
        os.remove(temp_file)
    return full_text

def is_quota_error(status_code, err_text):
    msg = str(err_text).lower()
    return status_code == 429 or "429" in msg or "quota" in msg or "resource_exhausted" in msg

def call_gemini_with_fallback(prompt, model_list, api_key, exhausted_set, timeout=50):
    """
    Calls Gemini REST API across a prioritized model list.
    Auto-retries 503 temporary spikes once and switches to the next model on 429/404.
    """
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"}
    }

    last_error = None
    for model_name in model_list:
        if model_name in exhausted_set:
            continue
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        
        for attempt in range(2):  # 1 automatic retry for 503 high-demand spikes
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            
            if response.status_code == 200:
                result_data = response.json()
                raw_json = result_data['candidates'][0]['content']['parts'][0]['text']
                return json.loads(raw_json), model_name
                
            if response.status_code == 503 and attempt == 0:
                print(f"      ⏳ 503 High Demand on {model_name}. Waiting 15s to retry...", flush=True)
                time.sleep(15)
                continue
                
            if is_quota_error(response.status_code, response.text):
                print(f"      ⚠️ Quota exhausted on {model_name}. Marking exhausted for today.", flush=True)
                exhausted_set.add(model_name)
                last_error = f"429 Quota Exceeded on {model_name}"
                break
                
            if response.status_code == 404:
                print(f"      ℹ️ Endpoint {model_name} unavailable (404). Trying fallback model...", flush=True)
                exhausted_set.add(model_name)
                last_error = f"404 Not Found on {model_name}"
                break
                
            last_error = f"API Error ({response.status_code}) on {model_name}: {response.text}"
            break

    raise Exception(last_error or "All models in chain exhausted or failed.")

def fmt_pct(val):
    """Formats a percentage value to the nearest integer with a % sign."""
    if val is None or str(val).strip() in ["", "-", "N/A", "null", "None"]:
        return ""
    s = str(val).replace("%", "").replace(",", "").replace("+", "").strip()
    try:
        num = round(float(s))
        return f"{num}%"
    except:
        return str(val).strip()

def fmt_num(val):
    """Formats absolute financial figures cleanly."""
    if val is None or str(val).strip() in ["", "-", "N/A", "null", "None"]:
        return ""
    s = str(val).replace(",", "").strip()
    try:
        f = float(s)
        if abs(f) >= 100:
            return str(round(f))
        return str(round(f, 1)).rstrip('0').rstrip('.') if '.' in str(round(f, 1)) else str(round(f, 1))
    except:
        return str(val).strip()

def run_scout_and_mixed_pass(transcript_text, company_name, api_key, exhausted_lite):
    """
    Pass 1 (Gemini 3.1 Flash-Lite):
    Classifies result into Excellent, Good, Mixed, or Weak + Turnaround flag.
    Also extracts Revenue & EPS YoY/QoQ % (nearest integer) so 'Mixed' results are immediately complete.
    """
    prompt = f"""
    You are an institutional quantitative equity analyst evaluating the quarterly financial results filing for {company_name}.
    Always prioritize CONSOLIDATED numbers over Standalone numbers (use Standalone only if Consolidated is not reported).
    Convert absolute Revenue, Operating Profit, and PAT figures into Crores (Cr) if reported in Lakhs or Millions (1 Crore = 100 Lakhs = 10 Million).
    If YoY % or QoQ % is not explicitly printed, calculate it from the absolute quarterly numbers and round to the NEAREST INTEGER.

    Classify the main verdict into EXACTLY ONE of these 4 tags:
    - "Excellent": Strong double-digit YoY revenue & PAT/EPS growth (e.g., >15-20% YoY) OR massive operating profit & margin expansion.
    - "Good": Solid positive YoY growth in revenue and profit (e.g., >8-15% YoY) with stable or improving margins.
    - "Mixed": Conflicting numbers (e.g., Revenue growing well but PAT/margins declining, or strong YoY but weak QoQ).
    - "Weak": Declining revenue and declining profits/margins, or widening losses.

    Separately evaluate the "Turnaround" add-on tag ("Yes" or "No"):
    - Set "turnaround" to "Yes" ONLY if the company swung from a Net Loss (negative PAT/EPS) in the prior YoY or QoQ period to a Net Profit (positive PAT/EPS) in the current quarter, OR staged a dramatic recovery from severe multi-quarter contraction. Otherwise set to "No".

    Return strictly a JSON object matching this schema:
    {{
        "verdict": "Excellent, Good, Mixed, or Weak",
        "turnaround": "Yes or No",
        "result_type": "Consolidated or Standalone",
        "curr_qtr": "e.g. Sep 2026",
        "prev_qtr": "e.g. Jun 2026",
        "yoy_qtr": "e.g. Sep 2025",
        "rev_yoy_pct": "nearest integer % e.g. 19%",
        "rev_qoq_pct": "nearest integer % e.g. 13%",
        "eps_yoy_pct": "nearest integer % e.g. 82%",
        "eps_qoq_pct": "nearest integer % e.g. 38%"
    }}

    Financial Filing Text:
    {transcript_text[:45000]}
    """
    data, used_model = call_gemini_with_fallback(prompt, SCOUT_LITE_MODELS, api_key, exhausted_lite, timeout=40)
    
    verdict = str(data.get("verdict", "Mixed")).strip().capitalize()
    if verdict not in ["Excellent", "Good", "Mixed", "Weak"]:
        verdict = "Mixed"
    data["verdict"] = verdict
    
    turnaround = str(data.get("turnaround", "No")).strip().capitalize()
    data["turnaround"] = "Yes" if turnaround == "Yes" else "No"
    return data, used_model

def run_premium_financial_extraction(transcript_text, company_name, verdict, turnaround, api_key, exhausted_premium):
    """
    Pass 2 (Gemini 3.6 Flash first -> Gemini 2.5 Flash second):
    Extracts the full institutional matrix (Revenue, Operating Profit, OPM %, PAT, EPS) across YoY, QoQ, and 3 quarters.
    """
    prompt = f"""
    You are an expert institutional financial analyst extracting quarterly financial results for {company_name}.
    
    CRITICAL RULES:
    1. Prioritize CONSOLIDATED financial results. Only use Standalone if Consolidated results are not in the filing.
    2. Express Revenue, Operating Profit (EBITDA / EBIT before exceptional items), and PAT (Net Profit after tax) in Indian Crores (Cr), rounded to the nearest integer or 1 decimal place (1 Crore = 100 Lakhs = 10 Million).
    3. Express EPS in Rupees (₹) up to 1 decimal place.
    4. If YoY %, QoQ %, or OPM % (Operating Profit / Revenue * 100) are not explicitly stated, calculate them mathematically from the quarterly numbers and round ALL percentages to the NEAREST INTEGER (e.g., "19%", "-59%", "331%").

    Return strictly a JSON object matching this exact schema:
    {{
        "result_type": "Consolidated or Standalone",
        "curr_qtr": "Short label for latest quarter, e.g. Sep 2026",
        "prev_qtr": "Short label for preceding quarter, e.g. Jun 2026",
        "yoy_qtr": "Short label for year-ago quarter, e.g. Sep 2025",
        "rev_yoy_pct": "Nearest integer %, e.g. 19%",
        "rev_qoq_pct": "Nearest integer %, e.g. 13%",
        "rev_curr": "Number in Cr, e.g. 886",
        "rev_prev": "Number in Cr, e.g. 786",
        "rev_yoy": "Number in Cr, e.g. 744",
        "op_yoy_pct": "Nearest integer %, e.g. 23%",
        "op_qoq_pct": "Nearest integer %, e.g. 14%",
        "op_curr": "Number in Cr, e.g. 134",
        "op_prev": "Number in Cr, e.g. 118",
        "op_yoy": "Number in Cr, e.g. 109",
        "opm_curr_pct": "Nearest integer %, e.g. 15%",
        "opm_prev_pct": "Nearest integer %, e.g. 15%",
        "opm_yoy_pct": "Nearest integer %, e.g. 15%",
        "pat_yoy_pct": "Nearest integer %, e.g. 82%",
        "pat_qoq_pct": "Nearest integer %, e.g. 38%",
        "pat_curr": "Number in Cr, e.g. 164",
        "pat_prev": "Number in Cr, e.g. 119",
        "pat_yoy": "Number in Cr, e.g. 90",
        "eps_yoy_pct": "Nearest integer %, e.g. 82%",
        "eps_qoq_pct": "Nearest integer %, e.g. 38%",
        "eps_curr": "EPS in Rs, e.g. 13.8",
        "eps_prev": "EPS in Rs, e.g. 10.0",
        "eps_yoy": "EPS in Rs, e.g. 7.6"
    }}

    Financial Filing Text:
    {transcript_text[:120000]}
    """
    return call_gemini_with_fallback(prompt, PREMIUM_MODELS, api_key, exhausted_premium, timeout=60)

def build_full_row(date_str, symbol, company, industry, verdict, turnaround, data, pdf_link, tier):
    return [
        date_str,
        symbol,
        company,
        industry,
        verdict,
        turnaround,
        data.get("result_type", "Consolidated"),
        data.get("curr_qtr", ""),
        data.get("prev_qtr", ""),
        data.get("yoy_qtr", ""),
        fmt_pct(data.get("rev_yoy_pct", "")),
        fmt_pct(data.get("rev_qoq_pct", "")),
        fmt_num(data.get("rev_curr", "")),
        fmt_num(data.get("rev_prev", "")),
        fmt_num(data.get("rev_yoy", "")),
        fmt_pct(data.get("op_yoy_pct", "")),
        fmt_pct(data.get("op_qoq_pct", "")),
        fmt_num(data.get("op_curr", "")),
        fmt_num(data.get("op_prev", "")),
        fmt_num(data.get("op_yoy", "")),
        fmt_pct(data.get("opm_curr_pct", "")),
        fmt_pct(data.get("opm_prev_pct", "")),
        fmt_pct(data.get("opm_yoy_pct", "")),
        fmt_pct(data.get("pat_yoy_pct", "")),
        fmt_pct(data.get("pat_qoq_pct", "")),
        fmt_num(data.get("pat_curr", "")),
        fmt_num(data.get("pat_prev", "")),
        fmt_num(data.get("pat_yoy", "")),
        fmt_pct(data.get("eps_yoy_pct", "")),
        fmt_pct(data.get("eps_qoq_pct", "")),
        fmt_num(data.get("eps_curr", "")),
        fmt_num(data.get("eps_prev", "")),
        fmt_num(data.get("eps_yoy", "")),
        pdf_link,
        tier
    ]

def main():
    print("Authenticating with Google Service Account for Financial Results Parser...", flush=True)
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

    # 1. Build 5 Crore Liquidity Map
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
            if 0 < val < 100000:
                val = val * 10000000
            liquidity_map[sym] = val
    except Exception as e:
        print(f"⚠️ Warning: Could not load Avg_Rupee_Volume_Master: {e}", flush=True)

    # 2. Load Financial_Results & Perform 1-Year Rolling Purge
    print("Checking Financial_Results sheet & purging records older than 1 year...", flush=True)
    num_cols = len(HEADERS)
    try:
        results_ws = spreadsheet.worksheet("Financial_Results")
        all_values = results_ws.get_all_values()
    except gspread.exceptions.WorksheetNotFound:
        results_ws = spreadsheet.add_worksheet(title="Financial_Results", rows="1000", cols=str(num_cols))
        results_ws.update(values=[HEADERS], range_name='A1:AI1')
        results_ws.format('A1:AI1', {'textFormat': {'bold': True}, "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9}})
        all_values = [HEADERS]

    kept_rows = [HEADERS]
    purged_count = 0
    if len(all_values) > 1:
        for raw_row in all_values[1:]:
            if not any(raw_row):
                continue
            row = (raw_row + [""] * num_cols)[:num_cols]
            row_date = parse_flexible_date(row[0])
            if row_date and row_date < one_year_cutoff:
                purged_count += 1
                continue
            kept_rows.append(row)

    if purged_count > 0 or (all_values and len(all_values[0]) < num_cols):
        print(f" -> Purged {purged_count} financial result rows older than 1 year. Syncing sheet...", flush=True)
        results_ws.clear()
        results_ws.update(values=kept_rows, range_name=f"A1:AI{len(kept_rows)}")
        results_ws.format('A1:AI1', {'textFormat': {'bold': True}, "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9}})

    processed_pdfs = set(str(r[33]).strip() for r in kept_rows[1:] if len(r) > 33 and str(r[33]).strip())

    # 3. Scan Corporate_Announcements for New Financial Results
    print("Reading Corporate_Announcements for Financial Results...", flush=True)
    corp_ws = spreadsheet.worksheet("Corporate_Announcements")
    corp_data = corp_ws.get_all_records()

    exhausted_lite = set()
    exhausted_premium = set()

    noise_phrases = [
        "audio recording", "video recording", "link of audio", "newspaper",
        "intimation of board meeting", "prior intimation", "trading window",
        "transcript", "investor presentation", "press release"
    ]

    for row in corp_data:
        category = str(row.get("Category", "")).strip().lower()
        announcement = str(row.get("Announcement", "")).strip().lower()
        pdf_link = str(row.get("PDF Link", "")).strip()
        company = str(row.get("Company Name", "")).strip()
        symbol = str(row.get("NSE Code", "")).replace(".NS", "").strip().upper()
        industry = str(row.get("Industry", "")).strip()
        date_str = str(row.get("Date", "")).strip()

        is_fin_result = (category == "financial results") or ("financial results" in announcement and "submitted to the exchange" in announcement)
        is_noise = any(np in announcement for np in noise_phrases)

        if not (is_fin_result and not is_noise and pdf_link and pdf_link.endswith(".pdf") and pdf_link not in processed_pdfs):
            continue

        # 1-Year Date Check
        filing_date = parse_flexible_date(date_str)
        if filing_date and filing_date < one_year_cutoff:
            continue

        # > 5 Crore Average Rupee Turnover Filter
        turnover = liquidity_map.get(symbol, 0)
        if liquidity_map and turnover < 50000000:
            print(f" -> Skipping {symbol}: Turnover < 5 Cr", flush=True)
            processed_pdfs.add(pdf_link)
            continue

        if len(exhausted_lite) >= len(SCOUT_LITE_MODELS):
            print("🛑 Scout Lite quota exhausted for today. Stopping new scans.", flush=True)
            break

        print(f" -> Scouting Financial Result: {company} ({symbol})...", flush=True)
        try:
            text = download_and_extract_pdf(pdf_link)
            if len(text.strip()) < 100:
                print(f"    ⚠️ PDF for {symbol} appears to be a scanned image without selectable text. Skipping.", flush=True)
                processed_pdfs.add(pdf_link)
                continue

            scout_data, lite_model_used = run_scout_and_mixed_pass(text, company, api_key, exhausted_lite)
            verdict = scout_data["verdict"]
            turnaround = scout_data["turnaround"]
            t_badge = " + TURNAROUND" if turnaround == "Yes" else ""
            print(f"    [Scout: {lite_model_used}] Verdict: {verdict}{t_badge}", flush=True)
            time.sleep(2)

            # Case 1: WEAK -> Stamp Weak and skip extraction
            if verdict == "Weak":
                weak_row = build_full_row(date_str, symbol, company, industry, "Weak", turnaround, {"result_type": scout_data.get("result_type", "Consolidated")}, pdf_link, "Skip")
                results_ws.append_row(weak_row)
                kept_rows.append(weak_row)
                processed_pdfs.add(pdf_link)
                print(f"    ✅ Stamped WEAK for {symbol} (Skipped deep extraction).", flush=True)
                continue

            # Case 2: MIXED -> Save Sales & EPS YoY/QoQ % written by Lite model
            if verdict == "Mixed":
                mixed_row = build_full_row(date_str, symbol, company, industry, "Mixed", turnaround, scout_data, pdf_link, "Lite")
                results_ws.append_row(mixed_row)
                kept_rows.append(mixed_row)
                processed_pdfs.add(pdf_link)
                print(f"    ✅ Saved MIXED summary (Sales & EPS %) for {symbol}.", flush=True)
                continue

            # Case 3: EXCELLENT or GOOD -> Extract Full Matrix with 3.6-Flash (or 2.5-Flash fallback)
            if len(exhausted_premium) < len(PREMIUM_MODELS):
                try:
                    prem_data, prem_model_used = run_premium_financial_extraction(text, company, verdict, turnaround, api_key, exhausted_premium)
                    prem_row = build_full_row(date_str, symbol, company, industry, verdict, turnaround, prem_data, pdf_link, "Premium")
                    results_ws.append_row(prem_row)
                    kept_rows.append(prem_row)
                    processed_pdfs.add(pdf_link)
                    print(f"    ✅ Saved PREMIUM ({prem_model_used}) Financial Matrix for {symbol} ({verdict}{t_badge})", flush=True)
                    time.sleep(4)
                except Exception as pe:
                    print(f"    ⚠️ Premium extraction paused ({pe}). Saving {symbol} to Backlog for tomorrow.", flush=True)
                    backlog_row = build_full_row(date_str, symbol, company, industry, verdict, turnaround, scout_data, pdf_link, "Backlog")
                    results_ws.append_row(backlog_row)
                    kept_rows.append(backlog_row)
                    processed_pdfs.add(pdf_link)
            else:
                # Both 3.6-Flash and 2.5-Flash exhausted -> Save as Backlog so it upgrades tomorrow
                backlog_row = build_full_row(date_str, symbol, company, industry, verdict, turnaround, scout_data, pdf_link, "Backlog")
                results_ws.append_row(backlog_row)
                kept_rows.append(backlog_row)
                processed_pdfs.add(pdf_link)
                print(f"    📌 Queued {symbol} ({verdict}{t_badge}) in Backlog for tomorrow's Premium quota.", flush=True)

        except requests.exceptions.Timeout:
            print(f"⚠️ Timeout processing {symbol}. Will retry next run.", flush=True)
        except Exception as e:
            print(f"⚠️ Error processing {symbol}: {e}", flush=True)

    # 4. Self-Healing Backlog Queue: Upgrade 'Backlog' rows (Excellent -> Good) when Premium quota is available
    if len(exhausted_premium) < len(PREMIUM_MODELS):
        print("\nChecking Financial Results Backlog Queue (Excellent -> Good)...", flush=True)
        prio_map = {"excellent": 0, "good": 1}
        backlog_candidates = []

        for idx in range(1, len(kept_rows)):
            r = kept_rows[idx]
            tier = str(r[34]).strip().lower() if len(r) > 34 else ""
            verd = str(r[4]).strip().lower() if len(r) > 4 else ""
            if tier == "backlog" and verd in prio_map:
                dt = parse_flexible_date(r[0]) or datetime(2000, 1, 1)
                backlog_candidates.append((prio_map[verd], -dt.timestamp(), idx, r))

        backlog_candidates.sort(key=lambda x: (x[0], x[1]))

        if not backlog_candidates:
            print(" -> No pending Backlog rows. All Excellent & Good results are fully extracted!", flush=True)
        else:
            print(f" -> Found {len(backlog_candidates)} Backlog rows to upgrade to Premium.", flush=True)
            for _, _, sheet_row_idx, r in backlog_candidates:
                if len(exhausted_premium) >= len(PREMIUM_MODELS):
                    print("🛑 All Premium models exhausted during backlog upgrade. Will resume tomorrow.", flush=True)
                    break
                date_str, symbol, company, industry, verdict, turnaround = r[0], r[1], r[2], r[3], r[4], r[5]
                pdf_link = r[33]
                sheet_line = sheet_row_idx + 1
                print(f" -> Upgrading Backlog [{verdict}] {company} ({symbol}) at Row {sheet_line}...", flush=True)
                try:
                    text = download_and_extract_pdf(pdf_link)
                    prem_data, prem_model_used = run_premium_financial_extraction(text, company, verdict, turnaround, api_key, exhausted_premium)
                    upgraded_row = build_full_row(date_str, symbol, company, industry, verdict, turnaround, prem_data, pdf_link, "Premium")
                    results_ws.update(values=[upgraded_row], range_name=f"A{sheet_line}:AI{sheet_line}")
                    kept_rows[sheet_row_idx] = upgraded_row
                    print(f"    ✅ Upgraded {symbol} to Premium via {prem_model_used}!", flush=True)
                    time.sleep(4)
                except Exception as e:
                    print(f"    ⚠️ Could not upgrade {symbol}: {e}", flush=True)

    print("\n🎉 Financial Results Parser run complete!", flush=True)

if __name__ == "__main__":
    main()
