@echo off
setlocal enabledelayedexpansion
REM === Run the full RAG demo pipeline (Windows) ===
REM Repo root expected: aviation-engine-maintenance-rag
cd /d "%~dp0"

REM ---- 0) Check venv ----
echo [info] Checking Python venv...
if not exist ".venv\Scripts\python.exe" (
  echo [error] .venv not found. Please run setup_and_fetch.bat first.
  pause & exit /b 1
)

echo [info] Activating venv...
call ".venv\Scripts\activate" || (echo [error] Could not activate venv & pause & exit /b 1)

REM Make output dirs
if not exist "data\pdfs" mkdir "data\pdfs"
if not exist "runs" mkdir "runs"

REM Useful env for cleaner logs
set PYTHONUTF8=1
set TRANSFORMERS_VERBOSITY=error

REM Optional: let user override defaults via env before running the .bat
if "%QUESTION%"=="" set "QUESTION=The turboprop engine is a gas turbine engine that turns a propeller through a speed reduction gear box. Approximately how much percent of the energy developed by the gas turbine engine is used to drive the propeller? And what happens to the rest of the available energy?"
if "%K%"=="" set "K=3"
if "%GEN_MODEL%"=="" set "GEN_MODEL=Qwen/Qwen2.5-1.5B-Instruct"

echo.
echo [step 1/5] Fetch FAA PDFs (idempotent)
python "scripts\fetch_pdfs.py" --out "data\pdfs"
if errorlevel 1 (echo [error] fetch_pdfs failed & pause & exit /b 1)

echo.
echo [step 2/5] Build corpus (PDF -> text chunks)
python "src\build_corpus.py" --pdf_dir "data\pdfs" --out "data\corpus.jsonl"
if errorlevel 1 (echo [error] build_corpus failed & pause & exit /b 1)

echo.
echo [step 3/5] Build embeddings + FAISS index
python "src\embed_index.py" --corpus "data\corpus.jsonl" --index "data\faiss.index"
if errorlevel 1 (echo [error] embed_index failed & pause & exit /b 1)

echo.
echo [step 4/5] Retrieval + Generation (demo query)
echo        Q: %QUESTION%
python "src\rag_pipeline.py" --q "%QUESTION%" --k %K% --out "runs\preds.jsonl" --gen_model "%GEN_MODEL%"
if errorlevel 1 (echo [error] rag_pipeline failed & pause & exit /b 1)

echo.
echo [step 5/5] Evaluation (EM/ROUGE and RAGAS if available)
if exist "data\qna_eval.jsonl" (
  python "src\eval_ragas.py" --pred "runs\preds.jsonl" --gold "data\qna_eval.jsonl"
  if errorlevel 1 (echo [error] eval_ragas failed & pause & exit /b 1)
) else (
  echo [warn] data\qna_eval.jsonl not found. Skipping evaluation step.
)

echo.
echo [ok] End-to-end run finished.
echo     - Corpus: data\corpus.jsonl
echo     - Index : data\faiss.index
echo     - Preds : runs\preds.jsonl
pause
