import fitz
from pathlib import Path

PDF_PATH = Path("backend/data/33.pdf")
OUTPUT_PATH = Path("backend/data/33_raw.txt")

if not PDF_PATH.exists():
    raise FileNotFoundError(f"PDF introuvable : {PDF_PATH}")

document = fitz.open(PDF_PATH)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text")
        f.write(f"\n\n===== PAGE {page_number} =====\n\n")
        f.write(text)

document.close()

print("Extraction terminée.")
print(f"Pages extraites : {page_number}")
print(f"Fichier créé : {OUTPUT_PATH}")
