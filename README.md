# ✈️ Aviation Engine Maintenance RAG

A Retrieval-Augmented Generation (RAG) system for **aircraft engine maintenance manuals**.  
Built on FAA Aviation Maintenance Technician Handbooks, it enables accurate, context-grounded question answering and evaluation.

---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suniltyagi/aviation-engine-maintenance-rag/blob/main/notebooks/demo_rag.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)

---


## 🔄 RAG Pipeline Flow

```mermaid
flowchart LR
    A[FAA PDFs in data/pdfs/] --> B[Chunking<br>build_corpus.py]
    B --> C[Embeddings + FAISS Index<br>embed_index.py]
    D[User Query (--q)] --> E[Retrieval of Relevant Chunks]
    C --> E
    E --> F[Answer Generation<br>rag_pipeline.py]
```


## 🚀 Demo 

You can try the pipeline in one click:

- **[Run on Google Colab](https://colab.research.google.com/github/suniltyagi/aviation-engine-maintenance-rag/blob/main/notebooks/demo_rag.ipynb)** (recommended)  
- Or open the notebook locally: `notebooks/demo_rag.ipynb`

*A lightweight Gradio/Streamlit app version is planned — see placeholder badge above if you want to host on Hugging Face Spaces later.*

---

## ✨ Features
- Ingests FAA handbooks in PDF form and chunks text  
- Embeds with `sentence-transformers/all-MiniLM-L6-v2`  
- Vector retrieval with FAISS  
- Context-grounded answering via LLMs (`Gemma-2`, `LLaMA-3.1`, `Flan-T5`)  
- Evaluation with EM, ROUGE, and RAGAS metrics  
- Ready-to-run Colab notebook and Windows batch scripts 
 
⚠️ Scope: Powerplant Only
This project currently includes only the FAA Aviation Maintenance Technician Handbook – Powerplant (FAA-H-8083-32B) as the knowledge base.
Non-Powerplant handbooks (General, Airframe) are not included to keep repository size manageable.
Queries outside the Powerplant domain may not be adequately answered or may return generic/uninformed responses.
You can refresh or update Powerplant content using scripts/fetch_pdfs.py or setup_and_fetch.bat — but those commands only affect Powerplant materials.

---

## 💡 Example

**Question:** What does the accessory gearbox drive?  
**Retrieved Context:** “The accessory gearbox provides drive for the starter, fuel pump, oil pump, hydraulic pump, and generators.”  
**Answer (RAG):** `starter, fuel pump, oil pump, hydraulic pump, generator`

---

## 📊 Results

We evaluated the pipeline on a gold Q&A set (`data/qna_eval.jsonl`) using **RAGAS**, **Exact Match (EM)**, and **ROUGE** metrics.  

| Metric       | Score | Notes |
|--------------|-------|-------|
| **Exact Match (EM)** | 1.00 | All 8 predictions exactly matched references |
| **ROUGE-L**          | 1.00 | Perfect overlap with reference answers |
| **Faithfulness (RAGAS)** | – | Not computed yet |
| **Relevance (RAGAS)**    | – | Not computed yet |

📌 *Based on 8 Q&A pairs. RAGAS scores can be added once evaluated with `src/eval_ragas.py`.*

---

## 📚 Source Material

This project uses the official **FAA Aviation Maintenance Technician Handbooks – Aircraft**:  
- [FAA Handbooks & Manuals – Aircraft](https://www.faa.gov/regulations_policies/handbooks_manuals/aircraft)

---

## 📖 Overview
- Ingest PDF manuals/logs → text chunks  
- Embed chunks with `sentence-transformers`  
- Build FAISS vector index  
- Retrieve + answer via LLM  
- Evaluate with RAGAS, EM, ROUGE  

---

## 🛠️ Tech Stack
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`  
- **Index:** FAISS  
- **LLMs:** `google/gemma-2-2b-it`, `meta-llama/Llama-3.1-8B-Instruct`, `google/flan-t5-base`  
- **Evaluation:** RAGAS, EM, ROUGE  
- **Utilities:** PyPDF, Pandas, NumPy  

---

## 📂 Structure
```text
aviation-engine-maintenance-rag/
├── data/
│   ├── pdfs/                # FAA handbook PDFs
│   └── qna_eval.jsonl       # gold Q&A set
├── notebooks/
│   └── demo_rag.ipynb       # Jupyter demo
├── scripts/
│   └── fetch_pdfs.py        # downloads FAA PDFs
├── src/
│   ├── build_corpus.py      # PDF → text chunks
│   ├── embed_index.py       # embeddings + FAISS index
│   ├── rag_pipeline.py      # retrieval + generation
│   ├── eval_ragas.py        # evaluation metrics
│   └── utils.py             # helper functions
├── setup_and_fetch.bat      # one-click setup (Windows)
├── run_end_to_end.bat       # one-click demo run
├── requirements.txt
├── README.md
└── LICENSE
```


---

## 📂 Knowledge Base: FAA PDFs

The PDFs placed in `data/pdfs/` are the **heart of this project** – they form the **source knowledge base**.  
All retrieval and answering is grounded in these FAA Aviation Maintenance Technician Handbooks.

When you run the pipeline, it processes the PDFs in three steps:

1. **Chunking** – `src/build_corpus.py` splits the PDFs into manageable text blocks.  
2. **Embedding & Indexing** – `src/embed_index.py` converts chunks into dense embeddings and builds a FAISS index.  
3. **Retrieval + Answer Generation** – `src/rag_pipeline.py` retrieves the most relevant chunks for a query and generates an answer.

This ensures that every answer is derived from the FAA handbooks rather than model hallucination.  
The FAA PDFs are included in this repo under `data/pdfs/` so the demo works out of the box.  
(You can also refresh them anytime using `scripts/fetch_pdfs.py` or `setup_and_fetch.bat`.)

---
## ⚡ Quickstart 

### 🪟 Quickstart Windows

```powershell
# Clone repo
git clone https://github.com/suniltyagi/aviation-engine-maintenance-rag.git
cd aviation-engine-maintenance-rag

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Refresh FAA PDFs
setup_and_fetch.bat

# Run the full pipeline (index → RAG → evaluation)
run_end_to_end.bat

```

### 🐧 Linux / macOS

```bash
# Clone repo
git clone https://github.com/suniltyagi/aviation-engine-maintenance-rag.git
cd aviation-engine-maintenance-rag

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Refresh FAA PDFs
python scripts/fetch_pdfs.py

# Run pipeline manually (index → RAG → evaluation)
python src/build_corpus.py
python src/embed_index.py
python src/rag_pipeline.py
python src/eval_ragas.py
```
