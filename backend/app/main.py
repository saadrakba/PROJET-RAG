from pathlib import Path
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi import Query
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = BASE_DIR / "backend" / "data"
EVAL_DIR = BASE_DIR / "backend" / "evaluation"

app = FastAPI(
    title="RAG Lab",
    description="Plateforme d'évaluation Chunking, Embedding et RAG",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_json(filename):
    path = EVAL_DIR / filename

    if not path.exists():
        return {"error": f"Fichier introuvable : {filename}"}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "project": "RAG Lab"
    }


@app.get("/api/dashboard")
def dashboard():
    chunking = load_json("retrieval_summary.json")
    embedding = load_json("embedding_summary.json")

    return {
        "document": {
            "name": "33.pdf",
            "cleaned_file": "33_cleaned_final.txt",
            "pages": 197,
            "chunks": 693
        },
        "chunking": chunking,
        "embedding": embedding
    }


@app.get("/api/chunking")
def chunking():
    return load_json("retrieval_summary.json")


@app.get("/api/embedding")
def embedding():
    return load_json("embedding_summary.json")


@app.get("/api/chunking-details")
def chunking_details():
    return load_json("retrieval_evaluation.json")


@app.get("/api/embedding-details")
def embedding_details():
    return load_json("embedding_evaluation.json")


@app.get("/api/questions")
def questions():
    return load_json("questions.json")


@app.get("/")
def frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/{path:path}")
def frontend_files(path: str):
    file_path = FRONTEND_DIR / path

    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)

    return FileResponse(FRONTEND_DIR / "index.html")


# -------------------------------------------------------------------
# RECHERCHE SEMANTIQUE
# -------------------------------------------------------------------

_sbert_model = None
_chunk_data = None


def get_sbert():

    global _sbert_model

    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer

        print("Chargement du mod?le Sentence-BERT...")

        _sbert_model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )

    return _sbert_model


def get_chunks():

    global _chunk_data

    if _chunk_data is None:

        path = DATA_DIR / "chunking_results.json"

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        _chunk_data = data["Fixed Token"]["chunks"]

    return _chunk_data


@app.get("/api/search")
def search(q: str = Query(..., min_length=1)):

    import numpy as np

    model = get_sbert()
    chunks = get_chunks()

    texts = [chunk["text"] for chunk in chunks]

    query_embedding = model.encode(
        [q],
        normalize_embeddings=True
    )[0]

    chunk_embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    scores = np.dot(
        chunk_embeddings,
        query_embedding
    )

    top_indices = np.argsort(scores)[::-1][:5]

    results = []

    for idx in top_indices:

        results.append({
            "rank": len(results) + 1,
            "score": float(scores[idx]),
            "text": texts[idx],
            "start": chunks[idx].get("start"),
            "end": chunks[idx].get("end")
        })

    return {
        "query": q,
        "method": "Sentence-BERT",
        "chunking": "Fixed Token",
        "top_k": 5,
        "results": results
    }


# -------------------------------------------------------------------
# RAG COMPLET : RETRIEVAL + OLLAMA + QWEN
# -------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/api/query")
def query_rag(request: QueryRequest):

    from backend.retriever import search
    import ollama

    query = request.query.strip()

    if not query:
        return {
            "error": "La question ne peut pas ?tre vide."
        }

    top_k = max(1, min(request.top_k, 10))

    # ---------------------------------------------------------------
    # 1. RETRIEVAL
    # ---------------------------------------------------------------

    results = search(
        query,
        top_k=top_k
    )

    # ---------------------------------------------------------------
    # 2. CONSTRUCTION DU CONTEXTE
    # ---------------------------------------------------------------

    context_parts = []

    for result in results:

        context_parts.append(
            f"""
--- SOURCE {result['rank']} ---
Score de pertinence : {result['score']:.4f}

{result['text']}
"""
        )

    context = "\n".join(context_parts)

    # ---------------------------------------------------------------
    # 3. PROMPT RAG
    # ---------------------------------------------------------------

    system_prompt = """
Tu es un assistant sp?cialis? dans l'analyse d'un document
scientifique.

Tu dois r?pondre UNIQUEMENT ? partir du CONTEXTE fourni.

R?GLES STRICTES :

1. Utilise uniquement les informations pr?sentes dans le contexte.
2. N'utilise pas tes connaissances g?n?rales pour compl?ter la r?ponse.
3. N'invente aucune information.
4. Ne fais aucune d?duction qui n'est pas soutenue par le contexte.
5. Si le contexte ne permet pas de r?pondre correctement,
   indique clairement que l'information n'est pas suffisamment
   pr?sente dans les passages r?cup?r?s.
6. R?ponds exclusivement en fran?ais.
7. Donne une r?ponse claire, pr?cise et structur?e.
8. Pour les affirmations importantes, indique les sources sous
   la forme [Source 1], [Source 2], etc.
9. Une source doit ?tre cit?e uniquement si son contenu soutient
   r?ellement l'affirmation.
"""

    user_prompt = f"""
QUESTION :
{query}

CONTEXTE DU DOCUMENT :
{context}

R?ponds ? la question en utilisant uniquement le contexte.
"""

    # ---------------------------------------------------------------
    # 4. OLLAMA / QWEN
    # ---------------------------------------------------------------

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    answer = response["message"]["content"]

    # ---------------------------------------------------------------
    # 5. REPONSE API
    # ---------------------------------------------------------------

    return {
        "query": query,
        "answer": answer,
        "model": "qwen2.5:3b",
        "embedding": "Sentence-BERT",
        "chunking": "Fixed Token",
        "top_k": top_k,
        "sources": results
    }
