@echo off
setlocal
echo [info] Creating virtual environment...
if not exist .venv\Scripts\python.exe (
  py -m venv .venv || python -m venv .venv
)

echo [info] Activating venv...
call .venv\Scripts\activate || (echo [error] could not activate venv & pause & exit /b 1)

echo [info] Upgrading pip...
python -m pip install --upgrade pip

echo [info] Installing requirements...
if not exist requirements.txt (
  echo [error] requirements.txt not found in %cd%
  pause
  exit /b 1
)
python -m pip install -r requirements.txt || (echo [error] pip failed & pause & exit /b 1)

echo [info] Fetching FAA PDFs...
if not exist scripts\fetch_pdfs.py (
  echo [error] scripts\fetch_pdfs.py not found
  pause
  exit /b 1
)
python scripts\fetch_pdfs.py --out data\pdfs || (echo [error] fetch failed & pause & exit /b 1)

echo [ok] Setup complete.
pause