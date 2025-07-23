import sys
import os
import json
from tqdm import tqdm  # For progress bar
import re
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter
from glob import glob

import load_transcripts as pp

# Add the parent directory to sys.path so Python can find the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def structure_all_transcripts_from_parts(input_folder, output_folder):
    """
    Loads and structures each raw transcript part file individually.

    Parameters:
        input_folder (str): Folder containing transcripts_data_part*.json
        output_folder (str): Folder to save structured files (e.g., structured_calls_1.json)
    """
    os.makedirs(output_folder, exist_ok=True)
    input_files = sorted(glob(os.path.join(input_folder, "transcripts_data_part*.json")))

    if not input_files:
        print("❌ No part files found.")
        return

    print(f"🔍 Found {len(input_files)} raw JSON chunks. Structuring now...")

    for i, input_path in enumerate(input_files, start=1):
        with open(input_path, "r", encoding="utf-8") as f:
            transcripts = json.load(f)

        data = []

        for file, sections in tqdm(transcripts.items(), desc=f"Structuring part {i}", unit="file"):
            file_info = pp.parse_filename(file)

            management_text = pp.remove_factset_metadata(sections.get('Management Discussion', ''))
            qna_text = pp.remove_factset_metadata(sections.get('Q&A Section', ''))

            management_segments = pp.split_and_extract_speakers(management_text)
            qna_segments = pp.split_and_extract_speakers(qna_text, is_qna_section=True)

            management_paragraphs = management_text.split("\n\n")
            qna_paragraphs = qna_text.split("\n\n")

            call_data = {
                'file': file_info['filename'],
                'filename': file_info['filename'],
                'company_name': file_info['company_name'],
                'ticker': file_info['ticker'],
                'quarter': file_info['quarter'],
                'year': file_info['year'],
                'date': file_info['date'],
                'management_discussion_full': management_text,
                'qa_section_full': qna_text,
                'speaker_segments_management': management_segments,
                'speaker_segments_qa': qna_segments,
                'management_paragraphs': management_paragraphs,
                'qa_paragraphs': qna_paragraphs
            }

            data.append(call_data)

        output_path = os.path.join(output_folder, f"structured_calls_{i}.json")
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(data, out_f, indent=4, ensure_ascii=False)

        print(f"✅ Saved structured file to {output_path}")

    print("\n🎉 All parts structured and saved!")

def extract_dates_from_filenames(pdf_root_folder):
    date_pattern = re.compile(r"(\d{2}-\d{2}-\d{4})")

    date_list = []

    folders = sorted([f for f in os.listdir(pdf_root_folder) if os.path.isdir(os.path.join(pdf_root_folder, f))])

    for folder_name in tqdm(folders, desc="Scanning folders"):
        folder_path = os.path.join(pdf_root_folder, folder_name)

        for file in os.listdir(folder_path):
            if not file.lower().endswith(".pdf"):
                continue

            match = date_pattern.search(file)
            if match:
                try:
                    date = datetime.strptime(match.group(1), "%d-%m-%Y")
                    date_list.append(date)
                except ValueError:
                    continue  # Skip files with invalid date format

    return date_list

def plot_transcripts_over_time(date_list, freq="M"):
    # Round dates to month or year
    if freq == "M":
        rounded_dates = [datetime(d.year, d.month, 1) for d in date_list]
    elif freq == "Y":
        rounded_dates = [datetime(d.year, 1, 1) for d in date_list]
    else:
        raise ValueError("freq must be 'M' for monthly or 'Y' for yearly")

    # Count transcripts per period
    counts = Counter(rounded_dates)
    sorted_dates = sorted(counts)
    values = [counts[dt] for dt in sorted_dates]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_dates, values, marker="o")
    plt.title(f"Transcripts Over Time ({'Monthly' if freq == 'M' else 'Yearly'})")
    plt.xlabel("Date")
    plt.ylabel("Number of Transcripts")
    plt.grid(True)
    plt.tight_layout()
    plt.show()