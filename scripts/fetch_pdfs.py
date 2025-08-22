#!/usr/bin/env python3
"""
Fetch all chapter PDFs for FAA-H-8083-32B (Aviation Maintenance Technician
Handbook – Powerplant) directly from the FAA website.

Usage:
  python scripts/fetch_pdfs.py [--out data/pdfs]
"""
import os
import sys
import time
import argparse
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

INDEX_URL = "https://www.faa.gov/regulationspolicies/handbooksmanuals/aviation/faa-h-8083-32b-aviation-maintenance-technician"
TIMEOUT = 60
CHUNK = 1024 * 1024  # 1 MB
UA = {"User-Agent": "mech-maint-log-rag/1.0 (education use)"}

def find_pdf_links(index_url: str) -> list[str]:
    resp = requests.get(index_url, headers=UA, timeout=TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    # collect index links
    seeds = set()
    for a in soup.find_all("a", href=True):
        url = urljoin(index_url, a["href"].strip())
        if url.lower().endswith(".pdf") or ("faa-h-8083-32b" in url and "faa.gov" in urlparse(url).netloc):
            seeds.add(url)

    # follow non-PDF seeds to find .pdf
    pdfs = set()
    for link in sorted(seeds):
        if link.lower().endswith(".pdf"):
            pdfs.add(link)
            continue
        try:
            r = requests.get(link, headers=UA, timeout=TIMEOUT)
            r.raise_for_status()
            s2 = BeautifulSoup(r.text, "lxml")
            for a2 in s2.find_all("a", href=True):
                u2 = urljoin(link, a2["href"].strip())
                if u2.lower().endswith(".pdf") and "faa.gov" in urlparse(u2).netloc:
                    pdfs.add(u2)
        except Exception as e:
            print(f"[warn] couldn't inspect {link}: {e}", file=sys.stderr)

    return sorted(pdfs)

def download(url: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    fname = url.split("/")[-1]
    out_path = os.path.join(out_dir, fname)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"[skip] {fname}")
        return
    print(f"[get ] {fname}")
    with requests.get(url, headers=UA, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        got = 0
        t0 = time.time()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(CHUNK):
                if not chunk:
                    continue
                f.write(chunk)
                got += len(chunk)
                if total:
                    pct = got * 100 // total
                    sys.stdout.write(f"\r      {pct:3d}% ({got/1e6:.1f}MB/{total/1e6:.1f}MB)")
                    sys.stdout.flush()
        sys.stdout.write(f"\r      100% done in {time.time()-t0:.1f}s\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pdfs", help="output folder for PDFs")
    args = ap.parse_args()
    print("[info] scanning FAA index…")
    urls = find_pdf_links(INDEX_URL)
    if not urls:
        print("[error] no PDFs found; page layout may have changed.", file=sys.stderr)
        sys.exit(2)
    print(f"[info] found {len(urls)} PDF(s)")
    for u in urls:
        download(u, args.out)
    print(f"[ok] PDFs saved under {args.out}")

if __name__ == "__main__":
    main()
