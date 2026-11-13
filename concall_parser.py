import os
import json
import time
import requests
import pymupdf  # PyMuPDF
import gspread
from google.oauth2.service_account import Credentials

SINGLE_SHEET_URL = 'https://docs.google.com/spreadsheets/d/1A2fUfXGKXXQxzFnoR30cFVqtmb-28KTi4fR4N0e507g/edit?usp=sharing'

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
    os.remove("temp_transcript.pdf")
    return full_text

def run_scout_pass(transcript_text, company_name, api_key):
    """Pass 1: Uses the cheap/fast 2.5-flash model to determine the verdict strictly based on guidance."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    prompt = f"""
    You are a ruthless quantitative equity research assistant. Your sole job is to evaluate the management's tone and forward guidance in the following earnings call for {company_name}. 
    
    Ignore standard corporate pleasantries. Base your evaluation strictly on forward-looking statements, guidance revisions, order book health, and margin trajectories.
    
    Classify the transcript into EXACTLY ONE of the following four categories:
    1. Confident: Management explicitly raises revenue/EPS guidance, reports a surging order book/backlog, notes expanding margins, announces aggressive growth CAPEX, or explicitly states demand is outstripping supply.
    2. Steady: Guidance is maintained. Execution is inline with past quarters. Demand is stable.
    3. Mixed: Conflicting signals (e.g., Revenue growing but margins compressing). 
    4. Weak: Management explicitly lowers guidance, reports shrinking margins, delays CAPEX, or notes rising inventory/canceled orders.
    
    Transcript:
    {transcript_text[:25000]}
    
    Respond with EXACTLY ONE WORD from the list above [Confident, Steady, Mixed, Weak]. Do not output any punctuation or markdown.
    """
    
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {'Content-Type': 'application/json'}
    
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    if response.status_code != 200:
        return "Mixed" # Fallback on error
        
    result_data = response.json()
    verdict = result_data['candidates'][0]['content']['parts'][0]['text'].strip().replace("*", "")
    
    valid_verdicts = ["Confident", "Steady", "Mixed", "Weak"]
    return verdict if verdict in valid_verdicts else "Mixed"


def run_deep_dive(transcript_text, company_name, verdict, api_key):
    """Pass 2: Uses Gemini 4 Argon for 'Confident' calls, and 2.5-Flash for the rest."""
    
    # 🟢 ROUTING ENGINE: Send high-value setups to the frontier Argon model
    target_model = "gemini-4-argon" if verdict == "Confident" else "gemini-2.5-flash"
    print(f"      [AI Router] Verdict is {verdict}. Routing to {target_model}...", flush=True)
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"
    
    prompt = f"""
    You are an expert equity research analyst. Analyze the following earnings call transcript for {company_name}.
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
        raise Exception(f"Gemini Deep Dive Failed: {response.text}")
        
    result_data = response.json()
    raw_json_text = result_data['candidates'][0]['content']['parts'][0]['text']
    return json.loads(raw_json_text)


def main():
    print("Authenticating with Google Service Account...", flush=True)
    creds_json = os.environ.get('GOOGLE_CREDENTIALS')
    api_key = os.environ.get("GEMINI_API_KEY", "").strip(' "\'\n\r\t')
    
    if not creds_json or not api_key: 
        raise SystemExit("🛑 CRITICAL ERROR: Credentials or API Key missing in GitHub Secrets.")
    
    creds_dict = json.loads(creds_json)
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    gc = gspread.authorize(creds)
    spreadsheet = gc.open_by_url(SINGLE_SHEET_URL)
    
    # 1. Map Average Rupee Volume (The 5 Crore Filter)
    print("Building liquidity map...", flush=True)
    try:
        vol_ws = spreadsheet.worksheet("Avg_Rupee_Volume_Master")
        vol_data = vol_ws.get_all_records()
        liquidity_map = {}
        
        for r in vol_data:
            sym = str(r.get("Symbol", "")).strip().upper()
            if not sym: continue
            
            # Hunts for the column containing turnover/volume
            val = 0
            for k, v in r.items():
                if 'rupee' in k.lower() or 'turnover' in k.lower() or 'volume' in k.lower():
                    try:
                        val_str = str(v).replace(',', '').replace('Cr', '').replace('₹', '').strip()
                        val = float(val_str)
                        break
                    except: pass
            
            # Normalize to absolute rupees if the sheet uses Crores (e.g., 5.5 = 55,000,000)
            if 0 < val < 10000: val = val * 10000000
            liquidity_map[sym] = val
            
    except Exception as e:
        print(f"⚠️ Warning: Could not map liquidity. Filter will be bypassed. Error: {e}", flush=True)
        liquidity_map = {}

    # 2. Check existing analyses to prevent duplicates
    try:
        analysis_ws = spreadsheet.worksheet("Concall_Analysis")
        existing_data = analysis_ws.get_all_records()
        processed_pdfs = [str(row.get("PDF Link", "")).strip() for row in existing_data]
    except gspread.exceptions.WorksheetNotFound:
        analysis_ws = spreadsheet.add_worksheet(title="Concall_Analysis", rows="500", cols="15")
        headers = ["Date", "Symbol", "Company", "Verdict", "Summary", "Growth Drivers", "Growth", "Capacity", "Order Book", "Margins", "Risks", "Other", "PDF Link"]
        analysis_ws.update(values=[headers], range_name='A1')
        analysis_ws.format('A1:M1', {'textFormat': {'bold': True}, "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9}})
        processed_pdfs = []

    print("Reading Corporate Announcements...", flush=True)
    corp_ws = spreadsheet.worksheet("Corporate_Announcements")
    corp_data = corp_ws.get_all_records()

    print("Scanning for new Earnings Call Transcripts...", flush=True)
    found_new = False
    
    for row in corp_data:
        announcement = str(row.get("Announcement", "")).lower()
        pdf_link = str(row.get("PDF Link", "")).strip()
        company = str(row.get("Company Name", ""))
        symbol = str(row.get("NSE Code", "")).strip().upper()
        date = str(row.get("Date", ""))
        
        is_transcript = any(kw in announcement for kw in ["transcript", "earnings call transcript", "concall transcript", "call transcript"])
        
        if is_transcript and pdf_link and pdf_link.endswith(".pdf") and pdf_link not in processed_pdfs:
            
            # 🟢 LIQUIDITY FILTER (Strict 5 Crore cutoff)
            turnover = liquidity_map.get(symbol, 0)
            if turnover > 0 and turnover < 50000000:
                print(f" -> Skipping {symbol}: Illiquid (Turnover < 5Cr)", flush=True)
                # Mark it processed so we don't check it again tomorrow
                processed_pdfs.append(pdf_link)
                continue
                
            found_new = True
            print(f" -> Processing {company} ({symbol})...", flush=True)
            
            try:
                # 1. Download
                text = download_and_extract_pdf(pdf_link)
                
                # 2. Scout Pass (Flash)
                verdict = run_scout_pass(text, company, api_key)
                
                # 3. Deep Dive Pass (Argon for Confident, Flash for others)
                ai_data = run_deep_dive(text, company, verdict, api_key)
                
                def format_bullets(key):
                    items = ai_data.get(key, [])
                    if not items: return ""
                    return "• " + "\n• ".join([str(i) for i in items])

                new_row = [
                    date, symbol, company, 
                    ai_data.get("verdict", verdict), 
                    ai_data.get("summary", ""),
                    format_bullets("growth_drivers"),
                    format_bullets("growth"),
                    format_bullets("capacity"),
                    format_bullets("order_book"),
                    format_bullets("margins"),
                    format_bullets("risks_and_red_flags"),
                    format_bullets("other_points"),
                    pdf_link
                ]
                
                analysis_ws.append_row(new_row)
                processed_pdfs.append(pdf_link)
                print(f"✅ Saved deep analysis for {symbol}", flush=True)
                time.sleep(3) 
                
            except requests.exceptions.Timeout:
                print(f"⚠️ Timeout Error: Download or Gemini API took too long for {symbol}. Skipping.", flush=True)
            except Exception as e:
                print(f"⚠️ Error processing {symbol}: {e}", flush=True)
                # If we hit a 429 quota error, stop the entire script cleanly for today
                if "429" in str(e) or "quota" in str(e).lower():
                    print("🛑 Quota exceeded. Shutting down gracefully for today. Will resume tomorrow.", flush=True)
                    break

    if not found_new:
        print("No new transcripts found today. All caught up!", flush=True)

if __name__ == "__main__":
    main()
