# GRAIL Data Automation

This project provides a comprehensive automated pipeline for collecting, standardizing, and analyzing public comments on federal notices. It is specifically tuned to track regulations and public discourse surrounding Artificial Intelligence (AI) and Machine Learning (ML).

## Methodology

The data pipeline consists of four distinct stages, leveraging both traditional web scraping techniques and modern Large Language Models (LLMs) for semantic understanding and data extraction.

### 1. Data Collection

**Federal Notices (`noticeCollection.py`)**
- **Source:** Queries the **Federal Register API** to identify relevant notices (RFIs, RFCs, NPRMs).
- **Filtering:** Uses a curated list of search terms related to AI/ML (e.g., "generative pre-trained transformer", "algorithmic accountability") to filter for relevant dockets.
- **Summarization:** integrating with `Ollama` (using models like `Llama 3.2`), it generates concise summaries of the notices.
- **Storage:** Metadata (Title, Agency, Dates, Abstract) is logged directly to a **Google Sheet**.

**Public Comments (`commentCollection.py`)**
- **Source:** Scrapes *Regulations.gov* for a specific Document ID using **Selenium**.
- **Relevance Check:** Before downloading, it analyzes the comment abstract using an LLM to determine if it is a substantive response or merely a description.
- **Acquisition:** Downloads all associated PDF attachments and extracts metadata (Submitter Name, Date, ID).

### 2. Data Processing & Extraction

- **PDF Parsing:** Utilizes `PyMuPDF` (fitz) to extract text from PDF attachments.
- **OCR Fallback:** For scanned or image-based PDFs, the system automatically employs `pytesseract` to perform Optical Character Recognition (OCR), ensuring no data is lost.
- **Organization:** Files are stored in a structured local directory format: `./comments/{DocumentID}/{CommentID}.pdf`.

### 3. Standardization (`commentStandardization.py`)

- **Normalization:** Raw extracted text is processed by an LLM to convert it into a clean, human-readable **Markdown** format.
- **Metadata Embedding:** Essential metadata is baked into the file structure.
- **Vector Embeddings:** The system generates vector embeddings for the standardized text using the `mxbai-embed-large` model. This allows for downstream semantic search and clustering analysis.

### 4. Analysis (`commentAnalysis.py`)

- **Structured Extraction:** The system uses `QueryFile` modules to query the document content.
- **Entity Classification:** It automatically classifies submitters by:
    - Organization Name & Type
    - 501(c) Status (for non-profits)
    - Role (e.g., "Political Advocacy", "Service Provision")
- **Insight Generation:**
    - **Summarization:** Generates brief 1-2 sentence summaries of the core arguments.
    - **Key Issues:** Extracts specific keywords and relevant issues addressed in the comment.
- **Reporting:** All analysis results are pushed to a **Google Sheet** for review.

---

## Setup & Usage

### Prerequisites
1. **Google Cloud Service Account:** You need a `service_account.json` file for Google Sheets API access.
2. **Ollama:** An Ollama instance (local or cloud) must be running for LLM inference.
3. **Python Environment:** Install dependencies (see requirements).
4. **Tesseract:** Install Tesseract OCR on your system for PDF processing.

### Directory Structure
Ensure the following folders exist (or let the scripts create them):
- `./logs/`
- `./comments/`
- `./standardized_comments/`
- `./embeddings/`

### Running the Scripts

1. **Collect Notices**
   ```bash
   python noticeCollection.py
   ```
   *Scrapes recent notices and updates the tracker spreadsheet.*

2. **Collect Comments**
   ```bash
   python commentCollection.py
   ```
   *Prompts for a Document ID and downloads all relevant comments and PDFs.*

3. **Standardize Comments**
   ```bash
   python commentStandardization.py
   ```
   *Converts PDFs to Markdown and generates embeddings.*

4. **Analyze Comments**
   ```bash
   python commentAnalysis.py
   ```
   *Extracts insights and updates the analysis spreadsheet.*

### Key Files
- `noticeCollection.py`: Scraper for Federal Register notices.
- `commentCollection.py`: Selenium scraper for Regulations.gov.
- `commentStandardization.py`: PDF-to-Markdown + Embedding pipeline.
- `commentAnalysis.py`: LLM-based analysis and spreadsheet reporter.
- `NoticeAnalyzer.py` & `QueryFile.py`: Helper classes for specific LLM tasks.
