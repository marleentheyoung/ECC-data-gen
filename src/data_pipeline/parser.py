import os
import json
from PyPDF2 import PdfReader
import re
import pandas as pd
from tqdm import tqdm
from dateutil import parser

def parse_pdf_first_page(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            page = reader.pages[0]
            text = page.extract_text()
            return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def extract_company_ticker_date_US(file, text):
    filename_date_pattern = r'(\d{2}-\d{2}-\d{4})'
    match = re.search(filename_date_pattern, file)
    if match:
        filename_date_str = match.group(1)
        filename_date = pd.to_datetime(filename_date_str, format='%d-%m-%Y')
        is_before_2011 = filename_date < pd.Timestamp('2011-06-01')
    else:
        filename_date = None
        is_before_2011 = None

    year = filename_date.year

    if not is_before_2011:
        target = ''.join(text[7:])
        matches = re.findall(r'\((.*?)\)', target)

        if len(matches) >= 2:
            ticker = matches[1]
            ticker = re.sub(r'\((.*?)\)', '', ticker, count=1).strip()
            cleaned_name = target.split("\n")[7]
        elif len(matches) == 1:
            ticker = matches[0]
            ticker = re.sub(r'\((.*?)\)', '', ticker).strip()
            cleaned_name = target.split("\n")[7]
        else:
            ticker = None
            cleaned_name = target.split("\n")[7]
        
        lines_2 = target.split()
        quarter = None
        company_name = cleaned_name  # fallback

        for idx, item in enumerate(lines_2):
            quarter_match = re.search(r'Q[1-4]', item)
            if quarter_match:
                quarter = lines_2[idx]
                break
    else:
        lines = text.splitlines()
        line = lines[1] if len(lines) > 1 else lines[0]
        items = line.split()
        
        if len(items) == 0:
            line = lines[-3] + lines[-2] + lines[-1]
            items = line.split()
        elif len(items) < 7 and len(lines) > 2:
            line = lines[1] + lines[2]
            items = line.split()

        for idx, item in enumerate(items):
            quarter_match = re.search(r'Q[1-6]', item)
            if quarter_match:
                company_name = ' '.join(items[:idx-1])
                quarter = items[idx]
                ticker = items[idx-1]
                ticker = ticker.replace('(', '').replace(')', '').strip()
                break   
        if not quarter_match:
            quarter, company_name, ticker = None, None, None

    return filename_date.strftime('%Y-%m-%d') if filename_date else None, quarter, year, ticker, company_name

def extract_company_ticker_date_EU(file, text):
    import re
    import pandas as pd
    from dateutil import parser
    
    # Initialize variables
    quarter_match = None
    quarter = None
    company_name = None
    ticker = None
    year = None
    
    filename_date_pattern = r'(\d{2}-\d{2}-\d{4})'
    match = re.search(filename_date_pattern, file)

    # Define regex patterns for various date formats
    date_patterns = [
        r"\b\d{1,2}[A-Za-z]{3,9}\d{2,4}\b",         # 29July2009 or 5September2010
        r"\b\d{1,2}-[A-Za-z]{3}-\d{2,4}\b",         # 12-Mar-10 or 01-Jan-2022
        r"\b\d{4}-\d{2}-\d{2}\b",                   # ISO: 2024-06-01
    ]

    # Combine into one pattern
    combined_pattern = "|".join(date_patterns)

    # Find all matching date strings
    matches = re.findall(combined_pattern, file)

    if matches:
        parsed_dates = []
        for match_item in matches:
            try:
                parsed_dates.append(parser.parse(match_item, dayfirst=True))
            except Exception as e:
                print(f"Could not parse {match_item}: {e}")

        if parsed_dates:
            # Convert to yyyy-mm-dd string format
            formatted_dates = [d.strftime("%Y-%m-%d") for d in parsed_dates]

            # Find the earliest date
            filename_date = pd.to_datetime(min(formatted_dates))
            is_before_2011 = filename_date < pd.Timestamp('2011-06-01')
            year = filename_date.year
        else:
            filename_date = None
            is_before_2011 = None
    else:
        filename_date = None
        is_before_2011 = None
    
    # Default to post-2011 if can't determine
    if is_before_2011 is None:
        is_before_2011 = False
    
    # PRE-2011 LOGIC
    if is_before_2011 and len(text) > 300:
        lines = text.splitlines()
        line = lines[1] if len(lines) > 1 else lines[0]
        items = line.split()
        
        if len(items) == 0:
            line = lines[-3] + lines[-2] + lines[-1]
            items = line.split()
        elif len(items) < 7 and len(lines) > 2:
            line = lines[1] + lines[2]
            items = line.split()

        quarter_match = None
        for idx, item in enumerate(items):
            quarter_match = re.search(r'Q[1-6]', item)
            if quarter_match:
                company_name = ' '.join(items[:idx-1]) if idx > 1 else None
                quarter = items[idx]
                ticker = items[idx-1] if idx > 0 else None
                if ticker:
                    ticker = ticker.replace('(', '').replace(')', '').strip()
                break   
        
        if not quarter_match:
            quarter, company_name, ticker = None, None, None
    
    # POST-2011 LOGIC
    else:
        target = ''.join(text[7:]) if len(text) > 7 else text
        
        # Check if filename starts with "CORRECTED TRANSCRIPT"
        filename_starts_with_corrected = file.upper().startswith("CORRECTED TRANSCRIPT")
        
        if filename_starts_with_corrected:
            # CORRECTED TRANSCRIPT: Use old logic (find ticker in PDF brackets, replace . with -)
            matches = re.findall(r'\((.*?)\)', target)

            if len(matches) >= 2:
                ticker = matches[1]
                ticker = re.sub(r'\((.*?)\)', '', ticker, count=1).strip()
            elif len(matches) == 1:
                ticker = matches[0]
                ticker = re.sub(r'\((.*?)\)', '', ticker).strip()
            else:
                ticker = None
            
            # Get company name
            try:
                cleaned_name = target.split("\n")[7] if len(target.split("\n")) > 7 else None
            except (IndexError, AttributeError):
                cleaned_name = None
                
        else:
            # REGULAR TRANSCRIPT: Hybrid approach (filename + PDF country code)
            
            # Get first part from filename
            filename_first_word = file.split()[0] if ' ' in file else file.split('.')[0]
            
            # Remove leading number if present
            if filename_first_word and filename_first_word[0].isdigit():
                filename_ticker_part = filename_first_word[1:]
            else:
                filename_ticker_part = filename_first_word
            
            # Get country code from PDF content (within brackets, split by '.')
            matches = re.findall(r'\((.*?)\)', target)
            pdf_country_part = None
            
            if matches:
                # Look for content with dots and extract the part after the dot
                for match_item in matches:
                    if '.' in match_item:
                        parts = match_item.split('.')
                        if len(parts) >= 2:
                            pdf_country_part = parts[-1]  # Take the last part after splitting by '.'
                            break
            
            # Combine the parts
            if filename_ticker_part and pdf_country_part:
                ticker = f"{filename_ticker_part}.{pdf_country_part}"
            elif filename_ticker_part:
                # No country code found (no dot in brackets), just use filename part
                ticker = filename_ticker_part
            else:
                # Fallback: try to get full ticker from parentheses
                if matches:
                    ticker = matches[0] if matches else None
                else:
                    ticker = None
            
            # Get company name
            try:
                cleaned_name = target.split("\n")[7] if len(target.split("\n")) > 7 else None
            except (IndexError, AttributeError):
                cleaned_name = None
        
        # Find quarter (common for both post-2011 strategies)
        lines_2 = target.split()
        quarter = None
        company_name = cleaned_name  # fallback

        for idx, item in enumerate(lines_2):
            quarter_match = re.search(r'Q[1-4]', item)
            if quarter_match:
                quarter = lines_2[idx]
                break        

    # Clean and standardize ticker (replace . with -)
    if ticker:
        ticker = ticker.strip().replace('.', '-')

    final_date = filename_date.strftime('%Y-%m-%d') if filename_date else None
    return final_date, quarter, year, ticker, company_name


def process_all_pdfs_in_directory(folder_path, pdf_folder, index):
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

    for json_file_name in tqdm(json_files, desc="📦 Processing JSON files", unit="file"):
        json_path = os.path.join(folder_path, json_file_name)
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Check if JSON is a list of entries
        if not isinstance(data, list):
            tqdm.write(f"⚠️ Skipping {json_file_name}: JSON is not a list")
            continue

        updated = False

        for entry in tqdm(data, desc=f"📝 {json_file_name}", leave=False):
            filename = entry.get('file')
            if not filename:
                continue

            pdf_path = os.path.join(pdf_folder, filename)
            text = parse_pdf_first_page(pdf_path)

            if not text or "Error reading PDF" in text:
                tqdm.write(f"❌ Failed to extract text from: {filename}")
                continue

            if index == 'SP500':
                date, quarter, year, ticker, company_name = extract_company_ticker_date_US(filename, text)
            elif index == 'STOXX600':
                date, quarter, year, ticker, company_name = extract_company_ticker_date_EU(filename, text)

            # Update the entry with new metadata
            entry['date'] = str(date) if date else None
            entry['quarter'] = quarter
            entry['year'] = year
            entry['ticker'] = ticker
            entry['company_name'] = company_name
            updated = True

        if updated:
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)