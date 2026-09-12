from pathlib import Path
import sys

import ollama

from retriever import search


MODEL_NAME = "qwen2.5:3b"

# Score minimal considéré comme suffisamment pertinent
OUT_OF_SCOPE_THRESHOLD = 0.55


def build_context(results):

    context_parts = []

    for result in results:

        context_parts.append(
            f"""
--- SOURCE {result['rank']} ---
Score de pertinence : {result['score']:.4f}

{result['text']}
"""
        )

    return "\n".join(context_parts)


def generate_answer(query, top_k=5):

    print("Recherche des passages pertinents...")

    results = search(
        query,
        top_k=top_k
    )

    # ---------------------------------------------------------------
    # DÉTECTION DES QUESTIONS HORS SUJET
    # ---------------------------------------------------------------

    if not results:

        return {
            "query": query,
            "answer": (
                "Cette question est hors du sujet du document. "
                "Je peux uniquement répondre aux questions liées "
                "au contenu scientifique de ce document."
            ),
            "model": MODEL_NAME,
            "chunking": "Fixed Token",
            "embedding": "Sentence-BERT",
            "top_k": top_k,
            "sources": []
        }

    best_score = results[0]["score"]

    print(f"Meilleur score de pertinence : {best_score:.4f}")

    if best_score < OUT_OF_SCOPE_THRESHOLD:

        print("Question détectée comme hors sujet.")

        return {
            "query": query,
            "answer": (
                "Cette question semble être hors du sujet du document. "
                "Je peux uniquement répondre aux questions liées "
                "au contenu scientifique de ce document."
            ),
            "model": MODEL_NAME,
            "chunking": "Fixed Token",
            "embedding": "Sentence-BERT",
            "top_k": top_k,
            "sources": results
        }

    # ---------------------------------------------------------------
    # CONSTRUCTION DU CONTEXTE
    # ---------------------------------------------------------------

    context = build_context(results)

    system_prompt = """
Tu es un assistant spécialisé dans l'analyse du document scientifique fourni.

RÈGLES IMPORTANTES :

1. Réponds uniquement à partir du contexte fourni.
2. N'invente aucune information.
3. Si la question est hors du sujet du document,
   indique clairement qu'elle est hors sujet.
4. Si le contexte ne permet pas de répondre à la question,
   indique clairement que l'information n'est pas suffisamment
   présente dans les passages récupérés.
5. Réponds exclusivement en français.
6. Donne une réponse claire, concise et structurée.
7. Ne transforme pas une hypothèse ou une information incertaine
   en fait établi.
8. Lorsque c'est pertinent, indique les sources utilisées
   sous la forme [Source 1], [Source 2], etc.
"""

    user_prompt = f"""
QUESTION :
{query}

CONTEXTE DU DOCUMENT :
{context}

Réponds à la question en utilisant uniquement le contexte ci-dessus.
Si le contexte ne permet pas de répondre correctement, indique-le clairement.
"""

    print("Envoi du contexte à Ollama...")
    print(f"Modèle : {MODEL_NAME}")

    response = ollama.chat(
        model=MODEL_NAME,
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

    return {
        "query": query,
        "answer": answer,
        "model": MODEL_NAME,
        "chunking": "Fixed Token",
        "embedding": "Sentence-BERT",
        "top_k": top_k,
        "sources": results
    }


if __name__ == "__main__":

    print()
    print("=" * 70)
    print("TEST DU PIPELINE RAG")
    print("=" * 70)

    query = (
        "Quel est le rôle des ondes planétaires "
        "dans les réchauffements stratosphériques ?"
    )

    print()
    print("QUESTION :")
    print(query)

    print()
    print("-" * 70)

    result = generate_answer(
        query,
        top_k=5
    )

    print()
    print("=" * 70)
    print("RÉPONSE DE QWEN")
    print("=" * 70)

    print()
    print(result["answer"])

    print()
    print("=" * 70)
    print("SOURCES RÉCUPÉRÉES")
    print("=" * 70)

    for source in result["sources"]:

        print(
            f"\nSource {source['rank']} "
            f"| Score : {source['score']:.4f}"
        )

        print(
            source["text"][:300]
            .replace("\n", " ")
            + "..."
        )

    print()
    print("=" * 70)
    print("PIPELINE RAG TERMINE")
    print("=" * 70)
