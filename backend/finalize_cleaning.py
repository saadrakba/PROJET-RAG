from pathlib import Path
import re

INPUT_PATH = Path("backend/data/33_cleaned.txt")
OUTPUT_PATH = Path("backend/data/33_cleaned_final.txt")

text = INPUT_PATH.read_text(encoding="utf-8")

# ---------------------------------------------------------
# 1. Supprimer les marqueurs de pages générés lors de
#    l'extraction précédente s'ils existent encore.
# ---------------------------------------------------------

text = re.sub(
    r"===== PAGE \d+ =====",
    "",
    text
)

# ---------------------------------------------------------
# 2. Supprimer les numéros de pages isolés.
#    On ne supprime que les lignes constituées uniquement
#    d'un nombre, afin de ne pas toucher aux nombres
#    scientifiques présents dans le texte.
# ---------------------------------------------------------

text = re.sub(
    r"(?m)^\s*\d{1,3}\s*$",
    "",
    text
)

# ---------------------------------------------------------
# 3. Nettoyer les espaces inutiles
# ---------------------------------------------------------

text = re.sub(r"[ \t]+", " ", text)

# ---------------------------------------------------------
# 4. Nettoyer les lignes vides multiples
# ---------------------------------------------------------

text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text)

# ---------------------------------------------------------
# 5. Nettoyage des espaces autour des retours à la ligne
# ---------------------------------------------------------

text = re.sub(r"[ \t]+\n", "\n", text)
text = re.sub(r"\n[ \t]+", "\n", text)

text = text.strip()

OUTPUT_PATH.write_text(text, encoding="utf-8")

print("=" * 65)
print("NETTOYAGE FINAL TERMINÉ")
print("=" * 65)
print(f"Fichier source : {INPUT_PATH}")
print(f"Fichier final  : {OUTPUT_PATH}")
print(f"Caractères     : {len(text):,}")
print(f"Lignes         : {len(text.splitlines()):,}")
print("=" * 65)
