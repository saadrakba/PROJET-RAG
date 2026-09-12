from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CHUNKS_FILE = DATA_DIR / "chunking_results.json"
OUTPUT_FILE = DATA_DIR / "vector_index.npz"

print("=" * 70)
print("CONSTRUCTION DE L'INDEX VECTORIEL")
print("=" * 70)

print()
print("Chargement des chunks...")

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

chunks = data["Fixed Token"]["chunks"]

print(f"Chunks chargés : {len(chunks)}")
print("Méthode : Fixed Token")

print()
print("Chargement de Sentence-BERT...")

model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

print("Modèle chargé.")

print()
print("Encodage des chunks...")

texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.asarray(
    embeddings,
    dtype=np.float32
)

print()
print(f"Shape des embeddings : {embeddings.shape}")

np.savez_compressed(
    OUTPUT_FILE,
    embeddings=embeddings
)

print()
print("=" * 70)
print("INDEX VECTORIEL CRÉÉ")
print("=" * 70)
print(f"Fichier : {OUTPUT_FILE}")
print(f"Chunks : {len(chunks)}")
print(f"Dimensions : {embeddings.shape[1]}")
print("=" * 70)
