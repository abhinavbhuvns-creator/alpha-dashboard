import os
import json
import time
import requests
import pymupdf  # PyMuPDF
import gspread
from google.oauth2.service_account import Credentials

SINGLE_SHEET_URL = 'https://docs.google.com/spreadsheets/d/1A2fUfXGKXXQxzFnoR30cFVqtmb-28KTi4fR4N0e507g/edit?usp=sharing'

def download_and_extract_pdf(pdf_url):
    """Downloads the transcript PDF into memory and extracts all text."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(pdf_url, headers=headers)
    
    with open("temp_transcript.pdf", "wb") as f:
        f.write(response.content)
        
    doc = pymupdf.open("temp_transcript.pdf")
    full_text = "\n".join([page.get_text() for page in doc])
    doc.close()
    os.remove("temp_transcript.pdf")
    return full_text

def analyze_with_gemini(transcript_text, company_name):
    """Bypasses the buggy Google Python library and uses a direct REST API call."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip(' "\'\n\r\t')
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing or empty.")
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    prompt = f"""
    You are an expert equity research analyst. Analyze the following earnings call transcript for {company_name}.
    Extract the requested information strictly as a JSON object matching the schema below. Keep bullet points concise and bold key numbers/percentages.
    
    JSON Schema:
    {{
        "verdict": "Choose strictly one: Confident, Steady, Mixed, or Weak",
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
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise Exception(f"Gemini API Failed: {response.text}")
        
    result_data = response.json()
    raw_json_text = result_data['candidates'][0]['content']['parts'][0]['text']
    return json.loads(raw_json_text)

def main():
    print("Authenticating with Google Service Account...")
    creds_json = os.environ.get('GOOGLE_CREDENTIALS')
    if not creds_json: raise SystemExit("🛑 Error: GOOGLE_CREDENTIALS not found in GitHub Secrets.")
    
    creds_dict = json.loads(creds_json)
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    gc = gspread.authorize(creds)
    
    spreadsheet = gc.open_by_url(SINGLE_SHEET_URL)
    
    # Check existing analyses to prevent duplicates
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

    print("Reading Corporate Announcements...")
    try:
        corp_ws = spreadsheet.worksheet("Corporate_Announcements")
        corp_data = corp_ws.get_all_records()
    except Exception as e:
        raise SystemExit(f"🛑 Error: Could not read Corporate Announcements. {e}")

    print("Scanning for new Earnings Call Transcripts...")
    found_new = False
    
    for row in corp_data:
        announcement = str(row.get("Announcement", "")).lower()
        pdf_link = str(row.get("PDF Link", "")).strip()
        company = str(row.get("Company Name", ""))
        symbol = str(row.get("NSE Code", ""))
        date = str(row.get("Date", ""))
        
        # Trigger words that identify a transcript filing
        is_transcript = any(kw in announcement for kw in ["transcript", "earnings call transcript", "concall transcript", "call transcript"])
        
        if is_transcript and pdf_link and pdf_link.endswith(".pdf") and pdf_link not in processed_pdfs:
            found_new = True
            print(f" -> Processing new transcript: {company} ({symbol})")
            try:
                text = download_and_extract_pdf(pdf_link)
                ai_data = analyze_with_gemini(text, company)
                
                # Format bullet points safely
                def format_bullets(key):
                    items = ai_data.get(key, [])
                    if not items: return ""
                    return "• " + "\n• ".join([str(i) for i in items])

                new_row = [
                    date, symbol, company, 
                    ai_data.get("verdict", ""), 
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
                print(f"✅ Saved analysis for {symbol}")
                time.sleep(3)  # Respect API rate limits
                
            except Exception as e:
                print(f"⚠️ Error processing {symbol}: {e}")

    if not found_new:
        print("No new transcripts found today. All caught up!")

if __name__ == "__main__":
    main()
