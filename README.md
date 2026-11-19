# ⚡ SPARK. Smart Platform for AI Recognition and Knowledge
SPARK is a honest AI detector that looks at sentence rhythm, how varied the sentences are and how words change. SPARK uses this information to tell real human writing apart from large AI models such as ChatGPT, Claude and Gemini.
SPARK analyzes content from:
1. **Direct Text Input** 
2.  **Document Files** 
3. **URLs** 
SPARK gives detailed percentages showing how much AI and how much human writing are present. SPARK also provides a verdict, a confidence rating, sentence‑by‑sentence colour maps and six style checks: sentence variety, use of contractions first‑person voice, punctuation variety, word variety and common AI patterns.

## Key Features
- **Percentage Breakdown**: Exact calculation of chance for example `85.2% AI‑Generated 14.8% Human‑Authored`.
- **Multi‑Document Upload**: Quick parser for PDF files using `pypdf` for Word files using `python‑docx` with an XML fallback and for text files.
- **Article & Blog URL Scraper**: Pulls clean article text. Removes ads, menus, headers and footers.
- **Sentence‑Level AI Heatmap**: It colours each sentence according to how likely it's to be AI:
**Red**: Likely AI (over 65%)
**Yellow**: Uncertain (between 38% and 65%)
**Green**: Very likely human (under 38%)
- **Stylometric & Linguistic Diagnostics (6 Core Signals)**:
1. **Burstiness (CV)**: Looks at how sentence lengths vary showing rhythm.
2. **Contractions**: Counts how often contractions appear per 100 words, a sign of conversation.
3. **1St‑Person Voice**: Looks for perspective compared to an impersonal AI tone.
4. **Punctuation Diversity**: Measures variety of dashes, semicolons and parentheses.
5. **Lexical Diversity (TTR)**: Calculates the ratio of words to total words.
6. **LLM Clichés**: Lists than 120 common AI transition patterns.
- **Configurable Sensitivity**: SPARK offers Balanced, Strict AI. Human‑safe preset settings.
- **Exportable Reports**: SPARK lets you download analysis reports in JSON with one click.

## Project Structure
SPARK works the same whether files are stored in the `src/` folder or in the root of the GitHub repository:

```

project SPARK/

│

├── data/

│   ├── dataset.csv                  # Labeled training dataset (Human vs. AI)

│   └── generate_dataset.py          # Dataset generator and synthesizer

│

├── models/

│   └── ai_detector.joblib           # Trained model artifact (TF-IDF + Calibrated Classifier)

│

├── src/

│   ├── __init__.py

│   ├── extractors.py                # Text extraction from PDF, DOCX, TXT and web links

│   ├── features.py                  # Burstiness, contractions, pronouns, lexical diversity

│   └── model.py                     # Detector classifier, calibration, sentence-level analyzer

│

├── templates/

│   └── index.html                   # Modern glassmorphism HTML dashboard for Flask

│

├── static/

│   ├── css/style.css                # Dark tech Hostinger console CSS

│   └── js/app.js                    # Dynamic frontend application controller

│

├── app.py                           # Streamlit Web Application (SPARK)

├── flask_app.py                     # Flask Web Application & REST API

├── requirements.txt                 # Python dependencies

├── Dockerfile                       # Container definition, for Cloud Run / Render / AWS

└── README.md

```

## Running Locally

### Option A: Streamlit Web App (Recommended)

```bash

streamlit run app.py

```

### Option B: Flask Web App & REST API

```bash

python flask_app.py

```

Open the address http://localhost:5000 in your web browser.

---
## REST API Endpoints (Flask)

### 1. Analyze Plain Text
### 2.. Analyze Document (PDF/Word/TXT)
### 3. Extract and Analyze Article URL

## Running Tests & Retraining

### Run Test Suite

```bash

python -m unittest test_detector.py

```

```bash

python -m unittest test_flask.py

```

### Retrain Model Pipeline

```bash

python data/generate_dataset.py

```

```bash

python train.py

```

---

## Deployment Guide

### Deploying with Docker

Build and start the Docker container:

```bash

# Build Docker image

docker build -t ai-content-detector.

# Run container (Streamlit, on port 8501)

docker run -p 8501:8501 ai-content-detector

```

### Deploying to Render / Cloud

1. Connect your repository to Render.

2. Create a **Web Service**.

3. Set the following options:

**Environment**: `Python`

**Build Command**: `pip install -r requirements.txt && python train.py`

**Start Command**: `streamlit run app.py --server.address=0.0.0.0` (or `python flask_app.py`)
