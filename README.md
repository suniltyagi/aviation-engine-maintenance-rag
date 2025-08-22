# aviation-engine-maintenance-rag

Retrieval-Augmented Generation (RAG) pipeline for the *Aviation Maintenance Technician Handbook – Engine* volume, with evaluation.

## Overview
This project demonstrates how to:
- Ingest PDF manuals/logs and chunk text.
- Embed chunks with `sentence-transformers/all-MiniLM-L6-v2`.
- Build a FAISS vector index for fast retrieval.
- Use an LLM to generate answers from retrieved context.
- Evaluate results with RAGAS (faithfulness, relevance) and EM/ROUGE.

## Tech Stack
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2  
- **Index:** FAISS  
- **LLM:** google/gemma-2-2b-it or meta-llama/Llama-3.1-8B-Instruct  
- **Eval:** ragas, EM, ROUGE  
- **Utils:** langchain, pypdf, pandas, numpy  

## Structure
```
aviation-engine-maintenance-rag/
├── data/
│   ├── pdfs/                # downloaded FAA handbook PDFs
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
