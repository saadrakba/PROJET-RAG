import json
import re
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from gensim.models import Word2Vec, FastText
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch


BASE_DIR = Path("backend")

TEXT_FILE = BASE_DIR / "data" / "33_cleaned_final.txt"
CHUNKS_FILE = BASE_DIR / "data" / "chunking_results.json"
QUESTIONS_FILE = BASE_DIR / "evaluation" / "questions.json"
GROUND_TRUTH_FILE = BASE_DIR / "evaluation" / "ground_truth.json"

OUTPUT_FILE = BASE_DIR / "evaluation" / "embedding_evaluation.json"
SUMMARY_FILE = BASE_DIR / "evaluation" / "embedding_summary.json"

TOP_K = 5

BERT_MODEL = "bert-base-multilingual-cased"
SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 1


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def tokenize(text):

    return re.findall(
        r"\b\w+(?:[-’']\w+)*\b",
        text.lower(),
        flags=re.UNICODE
    )


def cosine_similarity_matrix(
    query_vector,
    vectors
):

    query_vector = np.asarray(
        query_vector,
        dtype=np.float32
    )

    vectors = np.asarray(
        vectors,
        dtype=np.float32
    )

    q_norm = np.linalg.norm(
        query_vector
    )

    v_norms = np.linalg.norm(
        vectors,
        axis=1
    )

    denominator = v_norms * q_norm

    scores = np.zeros(
        len(vectors),
        dtype=np.float32
    )

    valid = denominator > 0

    scores[valid] = (
        vectors[valid] @ query_vector
    ) / denominator[valid]

    return scores


def overlap(
    start1,
    end1,
    start2,
    end2
):

    return max(
        0,
        min(end1, end2)
        - max(start1, start2)
    )


def is_relevant(
    chunk,
    reference
):

    start = chunk.get("start")
    end = chunk.get("end")

    ref_start = reference.get("start")
    ref_end = reference.get("end")

    if None in (
        start,
        end,
        ref_start,
        ref_end
    ):
        return False

    ov = overlap(
        start,
        end,
        ref_start,
        ref_end
    )

    if ov <= 0:
        return False

    chunk_length = max(
        1,
        end - start
    )

    reference_length = max(
        1,
        ref_end - ref_start
    )

    return (
        ov / chunk_length >= 0.30
        or ov >= min(
            200,
            0.10 * reference_length
        )
    )


def question_list(data):

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


def get_references(
    ground_truth,
    question_id
):

    data = ground_truth.get(
        question_id,
        {}
    )

    if isinstance(data, dict):

        return data.get(
            "references",
            []
        )

    return []


def evaluate_ranking(
    ranking,
    references
):

    relevant = []

    matched_references = set()

    for rank, chunk in enumerate(
        ranking,
        start=1
    ):

        matched = False

        for ref_index, reference in enumerate(
            references
        ):

            if is_relevant(
                chunk,
                reference
            ):

                matched = True
                matched_references.add(
                    ref_index
                )

        relevant.append(
            1 if matched else 0
        )

    relevant_count = sum(
        relevant
    )

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

    # MAP@5
    average_precision = 0.0
    hits = 0

    for i, rel in enumerate(
        relevant,
        start=1
    ):

        if rel:

            hits += 1

            average_precision += (
                hits / i
            )

    if hits > 0:

        average_precision /= min(
            len(references),
            TOP_K
        )

    # MRR@5
    reciprocal_rank = 0.0

    for i, rel in enumerate(
        relevant,
        start=1
    ):

        if rel:

            reciprocal_rank = 1 / i
            break

    # NDCG@5
    dcg = 0.0

    for i, rel in enumerate(
        relevant,
        start=1
    ):

        if rel:

            dcg += (
                1
                / np.log2(i + 1)
            )

    ideal_relevant = min(
        len(references),
        TOP_K
    )

    idcg = 0.0

    for i in range(
        1,
        ideal_relevant + 1
    ):

        idcg += (
            1
            / np.log2(i + 1)
        )

    ndcg = (
        dcg / idcg
        if idcg > 0
        else 0
    )

    return {
        "precision_at_5": precision,
        "recall_at_5": recall,
        "f1_at_5": f1,
        "map_at_5": average_precision,
        "mrr_at_5": reciprocal_rank,
        "ndcg_at_5": ndcg,
        "relevant_retrieved": relevant_count,
        "matched_references": len(
            matched_references
        ),
        "total_references": len(
            references
        )
    }


def mean_word_vectors(
    texts,
    model
):

    vectors = []

    dimension = model.vector_size

    for text in texts:

        words = tokenize(text)

        word_vectors = []

        for word in words:

            if word in model.wv:

                word_vectors.append(
                    model.wv[word]
                )

        if word_vectors:

            vector = np.mean(
                word_vectors,
                axis=0
            )

        else:

            vector = np.zeros(
                dimension,
                dtype=np.float32
            )

        vectors.append(
            vector
        )

    return np.asarray(
        vectors,
        dtype=np.float32
    )


def train_word2vec(
    sentences
):

    print(
        "\nEntraînement Word2Vec..."
    )

    model = Word2Vec(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=4,
        sg=1,
        epochs=20,
        seed=42
    )

    return model


def train_fasttext(
    sentences
):

    print(
        "\nEntraînement FastText..."
    )

    model = FastText(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=4,
        sg=1,
        epochs=20,
        seed=42
    )

    return model


def build_glove(
    sentences,
    vector_size=100,
    window=5,
    epochs=20
):

    print(
        "\nEntraînement GloVe..."
    )

    vocabulary = Counter()

    for sentence in sentences:

        vocabulary.update(
            sentence
        )

    words = [
        word
        for word, count
        in vocabulary.items()
        if count >= MIN_COUNT
    ]

    word_to_index = {
        word: i
        for i, word
        in enumerate(words)
    }

    vocab_size = len(words)

    print(
        f"Vocabulaire GloVe : {vocab_size}"
    )

    cooccurrence = defaultdict(float)

    for sentence in sentences:

        for i, word in enumerate(
            sentence
        ):

            if word not in word_to_index:
                continue

            start = max(
                0,
                i - window
            )

            end = min(
                len(sentence),
                i + window + 1
            )

            for j in range(
                start,
                end
            ):

                if i == j:
                    continue

                context = sentence[j]

                if context not in word_to_index:
                    continue

                distance = abs(
                    i - j
                )

                cooccurrence[
                    (
                        word_to_index[word],
                        word_to_index[context]
                    )
                ] += 1.0 / distance

    print(
        f"Paires de cooccurrence : "
        f"{len(cooccurrence)}"
    )

    rng = np.random.default_rng(
        42
    )

    W = (
        rng.normal(
            0,
            0.1,
            (
                vocab_size,
                vector_size
            )
        )
    )

    C = (
        rng.normal(
            0,
            0.1,
            (
                vocab_size,
                vector_size
            )
        )
    )

    bW = np.zeros(
        vocab_size
    )

    bC = np.zeros(
        vocab_size
    )

    learning_rate = 0.05

    entries = list(
        cooccurrence.items()
    )

    for epoch in range(
        epochs
    ):

        rng.shuffle(
            entries
        )

        total_loss = 0.0

        for (
            (i, j),
            x
        ) in entries:

            log_x = np.log(
                x
            )

            weight = min(
                1.0,
                (x / 100.0) ** 0.75
            )

            diff = (
                np.dot(
                    W[i],
                    C[j]
                )
                + bW[i]
                + bC[j]
                - log_x
            )

            grad = (
                weight * diff
            )

            w_i = W[i].copy()
            c_j = C[j].copy()

            W[i] -= (
                learning_rate
                * grad
                * c_j
            )

            C[j] -= (
                learning_rate
                * grad
                * w_i
            )

            bW[i] -= (
                learning_rate
                * grad
            )

            bC[j] -= (
                learning_rate
                * grad
            )

            total_loss += (
                weight
                * diff
                * diff
            )

        print(
            f"  Epoch {epoch + 1:02d}/{epochs}"
            f" - loss={total_loss:.2f}"
        )

    vectors = (
        W + C
    )

    return {
        "vectors": vectors,
        "word_to_index": word_to_index,
        "dimension": vector_size
    }


def glove_vectors(
    texts,
    glove
):

    vectors = []

    dimension = glove[
        "dimension"
    ]

    vectors_matrix = glove[
        "vectors"
    ]

    vocabulary = glove[
        "word_to_index"
    ]

    for text in texts:

        words = tokenize(
            text
        )

        selected = []

        for word in words:

            if word in vocabulary:

                selected.append(
                    vectors_matrix[
                        vocabulary[word]
                    ]
                )

        if selected:

            vector = np.mean(
                selected,
                axis=0
            )

        else:

            vector = np.zeros(
                dimension,
                dtype=np.float32
            )

        vectors.append(
            vector
        )

    return np.asarray(
        vectors,
        dtype=np.float32
    )


def bert_embeddings(
    texts
):

    print(
        "\nChargement BERT..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BERT_MODEL
    )

    model = AutoModel.from_pretrained(
        BERT_MODEL
    )

    model.eval()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)

    vectors = []

    batch_size = 8

    for start in range(
        0,
        len(texts),
        batch_size
    ):

        batch = texts[
            start:start + batch_size
        ]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        encoded = {
            key: value.to(device)
            for key, value
            in encoded.items()
        }

        with torch.no_grad():

            output = model(
                **encoded
            )

        mask = (
            encoded[
                "attention_mask"
            ]
            .unsqueeze(-1)
            .float()
        )

        pooled = (
            output.last_hidden_state
            * mask
        ).sum(
            dim=1
        ) / mask.sum(
            dim=1
        ).clamp(
            min=1e-9
        )

        vectors.append(
            pooled.cpu().numpy()
        )

    return np.vstack(
        vectors
    )


def sbert_embeddings(
    texts
):

    print(
        "\nChargement Sentence-BERT..."
    )

    model = SentenceTransformer(
        SBERT_MODEL
    )

    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True
    )


def calculate_intra(
    vectors,
    chunks
):

    values = []

    for chunk, vector in zip(
        chunks,
        vectors
    ):

        if np.linalg.norm(
            vector
        ) == 0:
            continue

        text = chunk.get(
            "text",
            ""
        )

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text
        )

        if len(sentences) < 2:
            continue

        # Pour les modèles mot,
        # on ne ré-encode pas ici.
        # L'intra est donc calculée
        # sur les chunks voisins.
        idx = chunks.index(
            chunk
        )

        if idx + 1 >= len(
            vectors
        ):
            continue

        other = vectors[
            idx + 1
        ]

        n1 = np.linalg.norm(
            vector
        )

        n2 = np.linalg.norm(
            other
        )

        if n1 > 0 and n2 > 0:

            values.append(
                np.dot(
                    vector,
                    other
                ) / (n1 * n2)
            )

    return float(
        np.mean(values)
        if values
        else 0
    )


def calculate_inter(
    vectors
):

    if len(vectors) < 2:
        return 0.0

    sample = vectors[
        :min(300, len(vectors))
    ]

    norms = np.linalg.norm(
        sample,
        axis=1
    )

    valid = norms > 0

    sample = sample[
        valid
    ]

    if len(sample) < 2:
        return 0.0

    sample = sample / np.linalg.norm(
        sample,
        axis=1,
        keepdims=True
    )

    similarity = (
        sample @ sample.T
    )

    n = len(
        similarity
    )

    upper = similarity[
        np.triu_indices(
            n,
            k=1
        )
    ]

    return float(
        np.mean(upper)
        if len(upper)
        else 0
    )


def main():

    print("=" * 80)
    print("EMBEDDING - COMPARAISON DES 5 METHODES")
    print("=" * 80)

    text = TEXT_FILE.read_text(
        encoding="utf-8"
    )

    chunking = load_json(
        CHUNKS_FILE
    )

    if "Fixed Token" not in chunking:

        raise RuntimeError(
            "Les chunks Fixed Token sont introuvables."
        )

    chunks = chunking[
        "Fixed Token"
    ][
        "chunks"
    ]

    questions_data = load_json(
        QUESTIONS_FILE
    )

    ground_truth = load_json(
        GROUND_TRUTH_FILE
    )

    questions = question_list(
        questions_data
    )

    texts = [
        chunk.get(
            "text",
            ""
        )
        for chunk in chunks
    ]

    print(
        f"\nDocument : {TEXT_FILE}"
    )

    print(
        f"Chunks utilisés : {len(chunks)}"
    )

    print(
        "Méthode de chunking : Fixed Token"
    )

    sentences = [
        tokenize(text)
        for text in text.splitlines()
        if tokenize(text)
    ]

    results = {}

    # --------------------------------------------------
    # WORD2VEC
    # --------------------------------------------------

    w2v = train_word2vec(
        sentences
    )

    vectors = mean_word_vectors(
        texts,
        w2v
    )

    results["Word2Vec"] = {
        "dimension": VECTOR_SIZE,
        "vectors": vectors
    }

    # --------------------------------------------------
    # GLOVE
    # --------------------------------------------------

    glove = build_glove(
        sentences
    )

    vectors = glove_vectors(
        texts,
        glove
    )

    results["GloVe"] = {
        "dimension": glove[
            "dimension"
        ],
        "vectors": vectors
    }

    # --------------------------------------------------
    # FASTTEXT
    # --------------------------------------------------

    fasttext = train_fasttext(
        sentences
    )

    vectors = mean_word_vectors(
        texts,
        fasttext
    )

    results["FastText"] = {
        "dimension": VECTOR_SIZE,
        "vectors": vectors
    }

    # --------------------------------------------------
    # BERT
    # --------------------------------------------------

    vectors = bert_embeddings(
        texts
    )

    results["BERT"] = {
        "dimension": vectors.shape[1],
        "vectors": vectors
    }

    # --------------------------------------------------
    # SENTENCE-BERT
    # --------------------------------------------------

    vectors = sbert_embeddings(
        texts
    )

    results["Sentence-BERT"] = {
        "dimension": vectors.shape[1],
        "vectors": vectors
    }

    # --------------------------------------------------
    # EVALUATION
    # --------------------------------------------------

    final_results = {}

    for method, data in results.items():

        vectors = data[
            "vectors"
        ]

        dimension = data[
            "dimension"
        ]

        print(
            "\n" + "=" * 80
        )

        print(
            f"EVALUATION : {method}"
        )

        print(
            "=" * 80
        )

        intra = calculate_intra(
            vectors,
            chunks
        )

        inter = calculate_inter(
            vectors
        )

        question_results = []

        for index, question_data in enumerate(
            questions,
            start=1
        ):

            raw_id = question_data.get(
                "id",
                index
            )

            if (
                isinstance(
                    raw_id,
                    str
                )
                and raw_id.lower().startswith(
                    "q"
                )
            ):

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

            references = get_references(
                ground_truth,
                question_id
            )

            if method == "BERT":

                query_vector = bert_embeddings(
                    [question]
                )[0]

            elif method == "Sentence-BERT":

                query_vector = sbert_embeddings(
                    [question]
                )[0]

            elif method == "Word2Vec":

                query_vector = mean_word_vectors(
                    [question],
                    w2v
                )[0]

            elif method == "FastText":

                query_vector = mean_word_vectors(
                    [question],
                    fasttext
                )[0]

            else:

                query_vector = glove_vectors(
                    [question],
                    glove
                )[0]

            scores = cosine_similarity_matrix(
                query_vector,
                vectors
            )

            top_indices = np.argsort(
                scores
            )[::-1][:TOP_K]

            ranking = []

            for idx in top_indices:

                chunk = dict(
                    chunks[idx]
                )

                chunk["score"] = float(
                    scores[idx]
                )

                ranking.append(
                    chunk
                )

            evaluation = evaluate_ranking(
                ranking,
                references
            )

            question_results.append({
                "question_id": question_id,
                "question": question,
                **evaluation
            })

            print(
                f"{question_id} | "
                f"P@5={evaluation['precision_at_5']:.3f} | "
                f"R@5={evaluation['recall_at_5']:.3f} | "
                f"F1@5={evaluation['f1_at_5']:.3f} | "
                f"MAP@5={evaluation['map_at_5']:.3f} | "
                f"MRR@5={evaluation['mrr_at_5']:.3f} | "
                f"NDCG@5={evaluation['ndcg_at_5']:.3f}"
            )

        final_results[method] = {
            "dimension": dimension,
            "intra_similarity": intra,
            "inter_similarity": inter,
            "questions": question_results,
            "precision_at_5": float(
                np.mean([
                    q["precision_at_5"]
                    for q in question_results
                ])
            ),
            "recall_at_5": float(
                np.mean([
                    q["recall_at_5"]
                    for q in question_results
                ])
            ),
            "f1_at_5": float(
                np.mean([
                    q["f1_at_5"]
                    for q in question_results
                ])
            ),
            "map_at_5": float(
                np.mean([
                    q["map_at_5"]
                    for q in question_results
                ])
            ),
            "mrr_at_5": float(
                np.mean([
                    q["mrr_at_5"]
                    for q in question_results
                ])
            ),
            "ndcg_at_5": float(
                np.mean([
                    q["ndcg_at_5"]
                    for q in question_results
                ])
            )
        }

    ranking = sorted(
        final_results.items(),
        key=lambda x: x[1]["f1_at_5"],
        reverse=True
    )

    summary = []

    for rank, (
        method,
        data
    ) in enumerate(
        ranking,
        start=1
    ):

        summary.append({
            "rank": rank,
            "method": method,
            "dimension": data[
                "dimension"
            ],
            "intra_similarity": data[
                "intra_similarity"
            ],
            "inter_similarity": data[
                "inter_similarity"
            ],
            "precision_at_5": data[
                "precision_at_5"
            ],
            "recall_at_5": data[
                "recall_at_5"
            ],
            "f1_at_5": data[
                "f1_at_5"
            ],
            "map_at_5": data[
                "map_at_5"
            ],
            "mrr_at_5": data[
                "mrr_at_5"
            ],
            "ndcg_at_5": data[
                "ndcg_at_5"
            ]
        })

    output = {
        "configuration": {
            "chunking_method": "Fixed Token",
            "chunk_count": len(chunks),
            "top_k": TOP_K
        },
        "methods": final_results,
        "ranking": summary
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
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
    print("=" * 100)
    print("RESULTATS FINAUX - EMBEDDING")
    print("=" * 100)

    print(
        f"{'Méthode':<18}"
        f"{'Dim':>6}"
        f"{'Intra':>10}"
        f"{'Inter':>10}"
        f"{'P@5':>9}"
        f"{'R@5':>9}"
        f"{'F1@5':>9}"
        f"{'MAP@5':>9}"
        f"{'MRR@5':>9}"
        f"{'NDCG@5':>9}"
    )

    print("-" * 100)

    for item in summary:

        print(
            f"{item['method']:<18}"
            f"{item['dimension']:>6}"
            f"{item['intra_similarity']:>10.3f}"
            f"{item['inter_similarity']:>10.3f}"
            f"{item['precision_at_5']:>9.3f}"
            f"{item['recall_at_5']:>9.3f}"
            f"{item['f1_at_5']:>9.3f}"
            f"{item['map_at_5']:>9.3f}"
            f"{item['mrr_at_5']:>9.3f}"
            f"{item['ndcg_at_5']:>9.3f}"
        )

    print("-" * 100)

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

    print(
        "\nEVALUATION EMBEDDING TERMINEE"
    )


if __name__ == "__main__":
    main()
