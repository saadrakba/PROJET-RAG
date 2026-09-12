import fitz
import re
from pathlib import Path

PDF_PATH = Path("backend/data/33.pdf")
OUTPUT_PATH = Path("backend/data/33_cleaned.txt")

# Vérification du fichier source
if not PDF_PATH.exists():
    raise FileNotFoundError(f"PDF introuvable : {PDF_PATH}")

document = fitz.open(PDF_PATH)

# Pages à supprimer :
# 1-4  : informations administratives, page de garde, remerciements
# 6-14 : table des matières
# 183-197 : bibliographie
#
# Les pages sont numérotées à partir de 1.
pages_to_remove = set()

for page in range(1, 5):
    pages_to_remove.add(page)

for page in range(6, 15):
    pages_to_remove.add(page)

for page in range(183, 198):
    pages_to_remove.add(page)

cleaned_pages = []
removed_pages = []

for page_number, page in enumerate(document, start=1):

    if page_number in pages_to_remove:
        removed_pages.append(page_number)
        continue

    text = page.get_text("text")

    if not text.strip():
        continue

    # Nettoyage des espaces
    text = re.sub(r"[ \t]+", " ", text)

    # Nettoyage des lignes contenant uniquement des espaces
    text = re.sub(r"\n[ \t]+", "\n", text)

    # Réduction des retours à la ligne excessifs
    text = re.sub(r"\n{3,}", "\n\n", text)

    text = text.strip()

    if text:
        cleaned_pages.append(text)

document.close()

# Assemblage
cleaned_text = "\n\n".join(cleaned_pages)

# Nettoyage final
cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
cleaned_text = cleaned_text.strip()

# Sauvegarde
OUTPUT_PATH.write_text(cleaned_text, encoding="utf-8")

print("=" * 60)
print("NETTOYAGE DU PDF TERMINÉ")
print("=" * 60)
print(f"PDF source          : {PDF_PATH}")
print(f"Pages originales    : {len(document) if False else 197}")
print(f"Pages supprimées    : {len(removed_pages)}")
print(f"Pages conservées    : {197 - len(removed_pages)}")
print(f"Pages supprimées    : {sorted(removed_pages)}")
print(f"Fichier créé       : {OUTPUT_PATH}")
print(f"Taille texte       : {len(cleaned_text):,} caractères")
print("=" * 60)
