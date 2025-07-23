# ECC Data Generation Pipeline

A professional pipeline for processing raw earnings call transcript PDFs into structured JSON files for analysis.

## 📁 Project Structure

```
ECC-data-generation/
├── data/
│   ├── raw/                    # Raw PDF transcripts
│   └── processed/              # Intermediate processing files
├── outputs/                    # Final structured JSONs
├── src/                        # Source code modules
├── notebooks/                  # Jupyter notebooks for exploration
├── main.py                     # Main pipeline script
├── load_transcripts.py         # PDF extraction functions
├── parse_data.py              # Data structuring functions
├── parser.py                  # Metadata extraction functions
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites

Make sure you have the required Python packages installed:

```bash
pip install PyPDF2 pandas tqdm spacy matplotlib python-dateutil
python -m spacy download en_core_web_sm
```

### Setup

1. **Place your PDF files** in the correct directory structure:
   ```
   data/raw/FactSet_Transcripts/STOXX600/
   data/raw/FactSet_Transcripts/SP500/
   ```

2. **Navigate to the project directory**:
   ```bash
   cd /Users/marleendejonge/Desktop/ECC-data-generation
   ```

### Running the Pipeline

**Basic usage:**
```bash
python main.py STOXX600
```

**For different stock indices:**
```bash
python main.py SP500
python main.py STOXX600
```

## 📊 What the Pipeline Does

The pipeline processes your PDF transcripts through 5 steps:

1. **🔍 Duplicate Detection**: Finds and removes duplicate PDF files
2. **📄 PDF Extraction**: Extracts text from PDFs and splits into parts
3. **🏗️ Structure Creation**: Converts raw text into structured JSON format
4. **🏷️ Metadata Extraction**: Adds company names, tickers, dates, and quarters
5. **📁 Output Organization**: Saves final results to the `outputs/` folder

## 📂 Output Files

After running, you'll find your structured transcripts in:
```
outputs/processed_transcripts/STOXX600/
├── structured_calls_1.json    # First batch of structured transcripts
├── structured_calls_2.json    # Second batch
└── ...
```

Each JSON file contains an array of transcript objects with:
- **Metadata**: company name, ticker, date, quarter, year
- **Full text**: complete management discussion and Q&A sections
- **Speaker segments**: parsed by individual speakers
- **Paragraphs**: text broken into manageable chunks

## 📋 Logging

The pipeline creates detailed logs in:
- **Console output**: Real-time progress and status updates
- **Log file**: `transcript_processing.log` (detailed debugging info)

## 🛠️ Troubleshooting

**Common issues:**

1. **"No PDF files found"**
   - Check that your PDFs are in `data/raw/FactSet_Transcripts/[INDEX]/`
   - Ensure PDF files have `.pdf` extension

2. **"Module not found"**
   - Make sure all Python files are in the same directory as `main.py`
   - Install required packages: `pip install -r requirements.txt`

3. **"Permission denied"**
   - Check file permissions on the data directory
   - Ensure you have write access to create output folders

## 🔧 Advanced Usage

**Custom processing:**
- Edit the `num_parts` parameter in `main.py` to change batch sizes
- Modify logging levels in the `setup_logging()` function
- Adjust file paths if your directory structure differs

**Individual functions:**
You can also import and use individual functions:
```python
from load_transcripts import extract_transcripts
from parse_data import structure_all_transcripts_from_parts
from parser import process_all_pdfs_in_directory
```

## 📄 Requirements

- Python 3.7+
- PyPDF2
- pandas  
- tqdm
- spacy (with en_core_web_sm model)
- matplotlib
- python-dateutil

## 📞 Support

For issues or questions, check the log files first. The pipeline provides detailed error messages to help diagnose problems.