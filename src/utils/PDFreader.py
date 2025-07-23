import fitz  # PyMuPDF
import pandas as pd
import os

def extract_text_from_pdf(pdf_path):
    """Extract text from a given PDF file."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])  # Extract text from all pages
    return text

def extract_text_from_folder(pdf_folder, verbose=True):
    """Extract text from all PDFs in a folder."""
    all_texts = {}

    files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    n = len(files)

    for i, file in enumerate(files):        
        pdf_path = os.path.join(pdf_folder, file)
        all_texts[file] = extract_text_from_pdf(pdf_path)

    return all_texts

def split_text_sections(file_name, text, verbose=False):
    """Splits transcript text into 'Management Discussion Section' and 'Q&A Section'."""
    
    # Define section headers
    management_header = "MANAGEMENT DISCUSSION SECTION"
    qa_header = "QUESTION AND ANSWER SECTION"
    
    # Check if both sections exist
    if management_header not in text or qa_header not in text:
        if verbose:
            print(f"❌ Error in {file_name}: Could not find both 'Management Discussion Section' and 'Q&A Section'.")
            print("🔍 Make sure the transcript follows the expected format.")
        return None  # Return None if sections are missing

    # Split at management discussion section
    parts = text.split(management_header, 1)
    if len(parts) < 2:
        if verbose:
            print(f"❌ Error in {file_name}: Could not correctly split at 'MANAGEMENT DISCUSSION SECTION'.")
        return None
    
    pre_mgmt_text, post_mgmt_text = parts  # Everything before is irrelevant

    # Split at Q&A section
    parts = post_mgmt_text.split(qa_header, 1)
    if len(parts) < 2:
        if verbose:
            print(f"❌ Error in {file_name}: Could not correctly split at 'QUESTION AND ANSWER SECTION'.")
        return None
    
    management_section, qa_section = parts

    if verbose:
        print(f"✅ Successfully split {file_name} into 'Management Discussion Section' and 'Q&A Section'.")
        print(f"📄 Management Section: {len(management_section.split())} words")
        print(f"📄 Q&A Section: {len(qa_section.split())} words")

    return {"File": file_name.strip("CORRECTED TRANSCRIPT "), "Management Discussion": management_section.strip(), "Q&A Section": qa_section.strip()}

def save_texts_to_csv(text_dict, output_csv, verbose=True):
    """
    Save extracted management and Q&A sections to a CSV file.
    
    :param text_dict: List of dictionaries with 'File', 'Management Discussion', and 'Q&A Section'.
    :param output_csv: Path to save the CSV file.
    :param verbose: Whether to print status messages.
    """

    # Convert list of dictionaries into DataFrame
    df = pd.DataFrame(text_dict)

    # Save as CSV
    df.to_csv(output_csv, index=False, encoding="utf-8")

    if verbose:
        print(f"✅ Successfully saved {len(df)} transcripts to {output_csv}")