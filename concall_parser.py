import os
import json
import requests
import fitz  # PyMuPDF
import google.generativeai as genai
import gspread
from google.oauth2.service_account import Credentials

# 1. Initialize Gemini API (Uses the secret you saved in GitHub)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

def download_and_extract_pdf(pdf_url):
    """Downloads the transcript PDF into memory and extracts all text."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(pdf_url, headers=headers)
    
    # Write to a temporary file
    with open("temp_transcript.pdf", "wb") as f:
        f.write(response.content)
        
    # Extract Text via PyMuPDF
    doc = fitz.open("temp_transcript.pdf")
    full_text = "\n".join([page.get_text() for page in doc])
    doc.close()
    os.remove("temp_transcript.pdf")
    
    return full_text

def analyze_with_gemini(transcript_text, company_name):
    """Forces Gemini 1.5 Flash to return pure JSON mapping to the dashboard matrix."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    
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
    
    # 🟢 JSON ENFORCEMENT: Guarantees the AI doesn't output conversational text
    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"}
    )
    
    return json.loads(response.text)

def write_to_google_sheets(company_name, symbol, date, pdf_url, ai_data):
    """Pushes the structured JSON into your Concall Google Sheet."""
    creds_dict = json.loads(os.environ.get('GOOGLE_CREDENTIALS'))
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    gc = gspread.authorize(creds)
    
    spreadsheet = gc.open_by_url('YOUR_SHEET_URL_HERE')
    
    try:
        ws = spreadsheet.worksheet("Concall_Analysis")
    except:
        ws = spreadsheet.add_worksheet(title="Concall_Analysis", rows="500", cols="15")
        headers = ["Date", "Symbol", "Company", "Verdict", "Summary", "Growth Drivers", "Growth", "Capacity", "Order Book", "Margins", "Risks", "Other", "PDF Link"]
        ws.update(values=[headers], range_name='A1')
    
    # Flatten lists into clean strings for the cell
    row_data = [
        date, symbol, company_name, 
        ai_data.get("verdict", ""), 
        ai_data.get("summary", ""),
        "\n• ".join(ai_data.get("growth_drivers", [])),
        "\n• ".join(ai_data.get("growth", [])),
        "\n• ".join(ai_data.get("capacity", [])),
        "\n• ".join(ai_data.get("order_book", [])),
        "\n• ".join(ai_data.get("margins", [])),
        "\n• ".join(ai_data.get("risks_and_red_flags", [])),
        "\n• ".join(ai_data.get("other_points", [])),
        pdf_url
    ]
    
    ws.append_row(row_data)
    print(f"✅ {company_name} transcript analyzed and saved.")
