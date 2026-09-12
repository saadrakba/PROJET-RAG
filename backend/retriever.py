from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CHUNKS_FILE = DATA_DIR / "chunking_results.json"
INDEX_FILE = DATA_DIR / "vector_index.npz"


MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


print("Chargement des chunks...")

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

chunks = data["Fixed Token"]["chunks"]

print(f"{len(chunks)} chunks chargés.")


print("Chargement de l'index vectoriel...")

index = np.load(INDEX_FILE)

embeddings = index["embeddings"]

print(f"Index chargé : {embeddings.shape}")


print("Chargement de Sentence-BERT...")

model = SentenceTransformer(MODEL_NAME)

print("Sentence-BERT chargé.")


def search(query, top_k=5):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    scores = np.dot(
        embeddings,
        query_embedding
    )

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for rank, idx in enumerate(top_indices, start=1):

        chunk = chunks[int(idx)]

        results.append({
            "rank": rank,
            "score": float(scores[idx]),
            "text": chunk["text"],
            "start": chunk.get("start"),
            "end": chunk.get("end")
        })

    return results


if __name__ == "__main__":

    print()
    print("=" * 70)
    print("TEST DU RETRIEVER")
    print("=" * 70)

    query = "Quel est le rôle des ondes planétaires dans les réchauffements stratosphériques ?"

    print()
    print(f"Question : {query}")
    print()

    results = search(query, top_k=5)

    for result in results:

        print("-" * 70)
        print(
            f"#{result['rank']} | "
            f"Score : {result['score']:.4f}"
        )
        print()
        print(result["text"][:700])
        print()

    print("=" * 70)
    print("RETRIEVER TEST TERMINE")
    print("=" * 70)
