import argparse
from pathlib import Path
from pypdf import PdfReader
from utils import write_jsonl, simple_word_chunks

def extract_pdf_text(pdf_path: Path) -> list[tuple[int,str]]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, p in enumerate(reader.pages):
        try:
            txt = p.extract_text() or ""
        except Exception:
            txt = ""
        pages.append((i+1, txt))
    return pages

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True, help="folder with PDFs")
    ap.add_argument("--out", dest="out_file", required=True, help="output corpus.jsonl")
    ap.add_argument("--max_words", type=int, default=600)
    ap.add_argument("--overlap", type=int, default=120)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    rows = []
    sid = 0
    for pdf in sorted(in_dir.glob("*.pdf")):
        pages = extract_pdf_text(pdf)
        for page_no, txt in pages:
            for chunk in simple_word_chunks(txt, max_words=args.max_words, overlap=args.overlap):
                sid += 1
                rows.append({
                    "id": f"{pdf.stem}_p{page_no}_c{sid}",
                    "text": chunk,
                    "source": pdf.name,
                    "page": page_no
                })
    write_jsonl(args.out_file, rows)
    print(f"[ok] wrote {len(rows)} chunks to {args.out_file}")

if __name__ == "__main__":
    main()
