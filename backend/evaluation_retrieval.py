import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path("backend")

CHUNKS_FILE = BASE_DIR / "data" / "chunking_results.json"
QUESTIONS_FILE = BASE_DIR / "evaluation" / "questions.json"
GROUND_TRUTH_FILE = BASE_DIR / "evaluation" / "ground_truth.json"

OUTPUT_FILE = BASE_DIR / "evaluation" / "retrieval_evaluation.json"
SUMMARY_FILE = BASE_DIR / "evaluation" / "retrieval_summary.json"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cosine_scores(query_embedding, embeddings):
    query_embedding = np.asarray(query_embedding)
    embeddings = np.asarray(embeddings)

    query_norm = np.linalg.norm(query_embedding)
    embedding_norms = np.linalg.norm(embeddings, axis=1)

    denominator = embedding_norms * query_norm

    scores = np.zeros(len(embeddings))

    valid = denominator > 0

    scores[valid] = (
        embeddings[valid] @ query_embedding
    ) / denominator[valid]

    return scores


def overlap(start1, end1, start2, end2):
    return max(
        0,
        min(end1, end2) - max(start1, start2)
    )


def is_relevant(chunk, reference):
    chunk_start = chunk.get("start")
    chunk_end = chunk.get("end")

    ref_start = reference.get("start")
    ref_end = reference.get("end")

    if None in (
        chunk_start,
        chunk_end,
        ref_start,
        ref_end
    ):
        return False

    ov = overlap(
        chunk_start,
        chunk_end,
        ref_start,
        ref_end
    )

    if ov <= 0:
        return False

    chunk_length = max(
        1,
        chunk_end - chunk_start
    )

    reference_length = max(
        1,
        ref_end - ref_start
    )

    chunk_ratio = ov / chunk_length

    return (
        chunk_ratio >= 0.30
        or ov >= min(
            200,
            0.10 * reference_length
        )
    )


def get_question_list(data):
    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        if "questions" in data:
            return data["questions"]

        result = []

        for key, value in data.items():

            if isinstance(value, dict):

                item = dict(value)

                if "id" not in item:
                    item["id"] = key

                result.append(item)

        return result

    return []


def get_references(ground_truth, question_id):

    data = ground_truth.get(
        question_id,
        {}
    )

    if isinstance(data, dict):
        return data.get(
            "references",
            []
        )

    if isinstance(data, list):
        return data

    return []


def evaluate_question(
    retrieved_chunks,
    references
):

    relevant_count = 0
    matched_references = set()

    details = []

    for rank, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        matched = []

        for ref_index, reference in enumerate(
            references
        ):

            if is_relevant(
                chunk,
                reference
            ):

                matched.append(ref_index)
                matched_references.add(
                    ref_index
                )

        if matched:
            relevant_count += 1

        details.append({
            "rank": rank,
            "chunk_id": chunk.get("chunk_id"),
            "score": float(
                chunk.get("score", 0)
            ),
            "start": chunk.get("start"),
            "end": chunk.get("end"),
            "text": chunk.get("text", ""),
            "relevant": bool(matched),
            "matched_reference_indices": matched
        })

    precision = (
        relevant_count / TOP_K
    )

    recall = (
        len(matched_references)
        / len(references)
        if references
        else 0
    )

    if precision + recall > 0:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )
    else:
        f1 = 0

    return {
        "precision_at_5": precision,
        "recall_at_5": recall,
        "f1_at_5": f1,
        "relevant_retrieved": relevant_count,
        "matched_references": len(
            matched_references
        ),
        "total_references": len(
            references
        ),
        "retrieved_chunks": details
    }


def main():

    print("=" * 80)
    print("EVALUATION RETRIEVAL - VERSION UNIFIEE")
    print("=" * 80)

    print("\nChargement des fichiers...")

    chunking_data = load_json(
        CHUNKS_FILE
    )

    questions_data = load_json(
        QUESTIONS_FILE
    )

    ground_truth = load_json(
        GROUND_TRUTH_FILE
    )

    questions = get_question_list(
        questions_data
    )

    print(
        f"Questions : {len(questions)}"
    )

    print(
        f"Méthodes : {len(chunking_data)}"
    )

    print("\nNombre de chunks utilisés :")

    for method_name, method_data in chunking_data.items():

        print(
            f"  {method_name:<22} : "
            f"{len(method_data['chunks']):>5}"
        )

    print("\nChargement du modèle...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    results = {}

    for method_name, method_data in chunking_data.items():

        chunks = method_data["chunks"]

        print("\n" + "=" * 80)
        print(
            f"METHODE : {method_name}"
        )
        print(
            f"Chunks utilisés : {len(chunks)}"
        )
        print("=" * 80)

        texts = [
            chunk.get("text", "")
            for chunk in chunks
        ]

        print(
            "Calcul des embeddings..."
        )

        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=True
        )

        question_results = []

        for index, question_data in enumerate(
            questions,
            start=1
        ):

            if isinstance(
                question_data,
                dict
            ):

                raw_id = question_data.get("id", index)

                if isinstance(raw_id, str) and raw_id.lower().startswith("q"):
                    question_id = raw_id.lower()
                else:
                    question_id = f"q{raw_id}"

                question = question_data.get(
                    "question",
                    question_data.get(
                        "text",
                        ""
                    )
                )

            else:

                question_id = f"q{index}"
                question = str(
                    question_data
                )

            references = get_references(
                ground_truth,
                question_id
            )

            query_embedding = model.encode(
                question,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            scores = cosine_scores(
                query_embedding,
                embeddings
            )

            top_indices = np.argsort(
                scores
            )[::-1][:TOP_K]

            retrieved = []

            for chunk_index in top_indices:

                chunk = dict(
                    chunks[chunk_index]
                )

                chunk["score"] = float(
                    scores[chunk_index]
                )

                retrieved.append(
                    chunk
                )

            evaluation = evaluate_question(
                retrieved,
                references
            )

            question_result = {
                "question_id": question_id,
                "question": question,
                "precision_at_5": evaluation[
                    "precision_at_5"
                ],
                "recall_at_5": evaluation[
                    "recall_at_5"
                ],
                "f1_at_5": evaluation[
                    "f1_at_5"
                ],
                "relevant_retrieved": evaluation[
                    "relevant_retrieved"
                ],
                "matched_references": evaluation[
                    "matched_references"
                ],
                "total_references": evaluation[
                    "total_references"
                ],
                "retrieved_chunks": evaluation[
                    "retrieved_chunks"
                ]
            }

            question_results.append(
                question_result
            )

            print(
                f"\n{question_id} | "
                f"P@5={evaluation['precision_at_5']:.3f} | "
                f"R@5={evaluation['recall_at_5']:.3f} | "
                f"F1@5={evaluation['f1_at_5']:.3f}"
            )

        mean_precision = float(
            np.mean([
                q["precision_at_5"]
                for q in question_results
            ])
        )

        mean_recall = float(
            np.mean([
                q["recall_at_5"]
                for q in question_results
            ])
        )

        mean_f1 = float(
            np.mean([
                q["f1_at_5"]
                for q in question_results
            ])
        )

        results[method_name] = {
            "chunk_count": len(chunks),
            "precision_at_5": mean_precision,
            "recall_at_5": mean_recall,
            "f1_at_5": mean_f1,
            "questions": question_results
        }

    ranking = sorted(
        results.items(),
        key=lambda item:
            item[1]["f1_at_5"],
        reverse=True
    )

    summary = []

    for rank, (method, data) in enumerate(
        ranking,
        start=1
    ):

        summary.append({
            "rank": rank,
            "method": method,
            "chunk_count": data[
                "chunk_count"
            ],
            "precision_at_5": data[
                "precision_at_5"
            ],
            "recall_at_5": data[
                "recall_at_5"
            ],
            "f1_at_5": data[
                "f1_at_5"
            ]
        })

    final_data = {
        "configuration": {
            "model": MODEL_NAME,
            "top_k": TOP_K,
            "source": str(
                CHUNKS_FILE
            ),
            "relevance_rule": {
                "chunk_overlap_ratio": 0.30,
                "minimum_overlap": (
                    "min(200, 10% of reference length)"
                )
            }
        },
        "methods": results,
        "ranking": summary
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_data,
            f,
            ensure_ascii=False,
            indent=2
        )

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n")
    print("=" * 80)
    print("RESULTATS FINAUX")
    print("=" * 80)

    print(
        f"{'Méthode':<25}"
        f"{'Chunks':>8}"
        f"{'P@5':>10}"
        f"{'R@5':>10}"
        f"{'F1@5':>10}"
    )

    print("-" * 80)

    for item in summary:

        print(
            f"{item['method']:<25}"
            f"{item['chunk_count']:>8}"
            f"{item['precision_at_5']:>10.3f}"
            f"{item['recall_at_5']:>10.3f}"
            f"{item['f1_at_5']:>10.3f}"
        )

    print("-" * 80)

    best = summary[0]

    print(
        f"\nMeilleure méthode selon F1@5 : "
        f"{best['method']}"
    )

    print(
        f"F1@5 = {best['f1_at_5']:.3f}"
    )

    print(
        f"\nFichier détaillé : {OUTPUT_FILE}"
    )

    print(
        f"Résumé : {SUMMARY_FILE}"
    )

    print("\nEVALUATION TERMINEE")


if __name__ == "__main__":
    main()

