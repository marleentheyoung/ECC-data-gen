import os
import json
from tqdm import tqdm
import sys
import re
import pandas as pd
import spacy

from math import ceil

# Add the parent directory to sys.path so Python can find the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import PDFreader

import os
from tqdm import tqdm

# Load spaCy's English model for Named Entity Recognition (NER)
print("🔹 Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")
print("✅ spaCy model loaded successfully!")

def remove_factset_metadata(text):
    """
    Removes the recurring FactSet metadata block by deleting 'FactSet CallStreet, LLC' 
    and the 10 lines before it.
    """
    lines = text.split("\n")  # Split text into lines

    indices_to_remove = [i for i, line in enumerate(lines) if "FactSet CallStreet, LLC" in line]
    
    for index in reversed(indices_to_remove):  # Reverse to avoid shifting indices
        start_index = max(0, index - 10)  # Ensure we don't go below 0
        del lines[start_index:index + 2]  # Delete the block

    return "\n".join(lines)  # Reconstruct text

def split_and_extract_speakers(text, is_qna_section=False):
    """Splits text into speaker segments, extracts speaker names + professions, and optionally adds Q/A type.
    Each segment will contain a list of paragraphs instead of a single text block.
    """
    if pd.isna(text) or text.strip() == "":
        return []

    # Split on FactSet-style dotted separator
    segments = re.split(r'\.{10,}', text)

    extracted_segments = []

    for segment in segments:

        segment = segment.strip()
        if not segment:
            continue

        # Special handling for the Operator case
        if segment.startswith("Operator:"):
            speaker = "Operator"
            profession = "Operator"

            # Everything after "Operator:" is the text
            text_after_colon = segment.partition(":")[2].strip()
            
            paragraphs = [p.replace('\n', ' ').strip() for p in text_after_colon.split("\n \n") if p.strip()]
        elif segment.startswith("[Abrupt Start]"):
            continue
        else:
            # Normal Case: Speaker on first line, profession on second line
            lines = segment.split("\n")

            speaker = lines[0]

            try:
                profession = lines[1]
            except:
                continue

            # Everything after "Operator:" is the text
            text_after_colon = '\n'.join(lines[2:])

            paragraphs = [p.replace('\n', ' ').strip() for p in text_after_colon.split("\n \n") if p.strip()]

        # Extract Q/A type in Q&A Section (first paragraph starts with Q or A)
        qa_type = None
        if is_qna_section and paragraphs:
            first_para = paragraphs[0].strip()
            if first_para.startswith('Q '):
                qa_type = 'Q'
                paragraphs[0] = first_para[2:].strip()
            elif first_para.startswith('A '):
                qa_type = 'A'
                paragraphs[0] = first_para[2:].strip()

        # Store result
        if paragraphs:
            segment_data = {
                'speaker': speaker,
                'profession': profession,
                'paragraphs': paragraphs
            }
            if is_qna_section:
                segment_data['qa_type'] = qa_type

            extracted_segments.append(segment_data)

    return extracted_segments

def parse_filename(file):
    # Remove the prefix like 'CORRECTED TRANSCRIPT' and the suffix like '.pdf'
    base = file.replace('CORRECTED TRANSCRIPT', '').replace('.pdf', '').strip()

    # Regex to match company name, ticker, quarter, year, date
    match = re.search(r'(.+?)([A-Z]{2,}[A-Z\d-]*)( Q[1-4]) (\d{4}) Earnings Call (\d{1,2}[A-Za-z]+?\d{4})', base)

    if match:
        company_name = match.group(1).strip()
        ticker = match.group(2).strip()
        quarter = match.group(3).strip()
        year = match.group(4).strip()
        date = match.group(5).strip()
    else:
        # Fallback in case the pattern fails
        company_name = None
        ticker = None
        quarter = None
        year = None
        date = None

    return {
        'filename': file,
        'company_name': company_name,
        'ticker': ticker,
        'quarter': quarter,
        'year': year,
        'date': date
    }

def remove_person_names(text, keep='[REDACTED]'):
    """
    Uses spaCy's Named Entity Recognition (NER) to remove person names from the transcript.
    """
    doc = nlp(text)  # Process text with spaCy
    cleaned_text = []
    name_count = 0  # Counter for number of names removed

    for token in doc:
        if token.ent_type_ == "PERSON":  
            cleaned_text.append(keep)  # Replace names with a placeholder
            name_count += 1
        else:
            cleaned_text.append(token.text)

    if name_count > 0:
        print(f"🔹 Replaced {name_count} person names with '{keep}'.")
    else:
        print("✅ No person names detected in this transcript.")

    return " ".join(cleaned_text)  # Reconstruct cleaned text

def extract_transcripts(pdf_root_folder, output_basename, num_parts):
    # Get all PDF files in the input folder
    all_files = sorted([f for f in os.listdir(pdf_root_folder) if f.lower().endswith('.pdf')])
    total_files = len(all_files)

    if total_files == 0:
        print("❌ No PDF files found in the input folder.")
        return

    print(f"📄 Found {total_files} PDF transcripts. Splitting into {num_parts} parts.")

    # Calculate batch size
    chunk_size = ceil(total_files / num_parts)

    for i in range(num_parts):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_files)
        batch_files = all_files[start:end]
        all_transcripts = {}

        print(f"\n🧩 Processing part {i + 1} | Files {start + 1} to {end}")

        for filename in tqdm(batch_files, desc=f"Part {i + 1}", unit="file"):
            pdf_path = os.path.join(pdf_root_folder, filename)
            try:
                text = PDFreader.extract_text_from_pdf(pdf_path)
                sections = PDFreader.split_text_sections(filename, text)
                if sections:
                    all_transcripts[filename] = sections
            except Exception as e:
                print(f"⚠️ Error processing {filename}: {e}")

        output_path = f"{output_basename}_part{i + 1}.json"
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(all_transcripts, json_file, indent=4, ensure_ascii=False)

        print(f"✅ Saved part {i + 1} with {len(all_transcripts)} transcripts to {output_path}")

def find_and_delete_duplicate_filenames(pdf_root_folder, delete=False):
    seen_filenames = {}
    deleted_files = []

    folders = sorted([f for f in os.listdir(pdf_root_folder) if os.path.isdir(os.path.join(pdf_root_folder, f))])

    for folder_name in tqdm(folders, desc="Checking & deleting duplicates", unit="folder"):
        folder_path = os.path.join(pdf_root_folder, folder_name)

        for file in os.listdir(folder_path):
            if not file.lower().endswith(".pdf"):
                continue

            full_path = os.path.join(folder_path, file)

            if file in seen_filenames:
                # Duplicate found → delete it
                if delete:
                    try:
                        os.remove(full_path)
                        deleted_files.append(full_path)
                    except Exception as e:
                        print(f"⚠️ Could not delete {full_path}: {e}")
            else:
                seen_filenames[file] = full_path

    print(f"\n🗑️ Deleted {len(deleted_files)} duplicate files (keeping one copy of each).")
    return deleted_files
