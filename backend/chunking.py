from pathlib import Path
import json
import re
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = Path("backend/data/33_cleaned_final.txt")
OUTPUT_PATH = Path("backend/data/chunking_results.json")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOKENIZER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

SEMANTIC_THRESHOLD = 0.65


# ============================================================
# UTILITAIRES
# ============================================================

def normalize_spaces(text):
    return re.sub(r"\s+", " ", text).strip()


def locate_chunk(source, chunk_text, search_from=0):

    pos = source.find(chunk_text, search_from)

    if pos != -1:
        return pos, pos + len(chunk_text)

    target = normalize_spaces(chunk_text)

    if not target:
        return None

    tail = source[search_from:]

    normalized = []
    positions = []

    previous_space = False

    for i, char in enumerate(tail):

        if char.isspace():

            if not previous_space:
                normalized.append(" ")
                positions.append(search_from + i)

            previous_space = True

        else:

            normalized.append(char)
            positions.append(search_from + i)
            previous_space = False

    normalized_source = "".join(normalized).strip()

    pos = normalized_source.find(target)

    if pos == -1:
        return None

    if pos >= len(positions):
        return None

    start = positions[pos]

    end_index = pos + len(target) - 1

    if end_index >= len(positions):
        end = len(source)
    else:
        end = positions[end_index] + 1

    return start, end


# ============================================================
# 1. RECURSIVE CHARACTER
# ============================================================

def recursive_character(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    raw_chunks = splitter.split_text(text)

    chunks = []
    search_from = 0

    for chunk_text in raw_chunks:

        location = locate_chunk(
            text,
            chunk_text,
            search_from
        )

        if location is None:
            location = locate_chunk(
                text,
                chunk_text,
                0
            )

        if location is None:
            continue

        start, end = location

        chunks.append({
            "text": chunk_text,
            "start": start,
            "end": end
        })

        search_from = max(
            start + 1,
            end - CHUNK_OVERLAP
        )

    return chunks


# ============================================================
# 2. CHARACTER
# ============================================================

def character(text):

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + CHUNK_SIZE,
            len(text)
        )

        chunk_text = text[start:end]

        if chunk_text.strip():

            chunks.append({
                "text": chunk_text,
                "start": start,
                "end": end
            })

        if end >= len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


# ============================================================
# 3. WORD
# ============================================================

def word(text):

    matches = list(
        re.finditer(r"\S+", text)
    )

    chunks = []

    words_per_chunk = 100
    overlap_words = 10

    start_word = 0

    while start_word < len(matches):

        end_word = min(
            start_word + words_per_chunk,
            len(matches)
        )

        start = matches[start_word].start()
        end = matches[end_word - 1].end()

        chunks.append({
            "text": text[start:end],
            "start": start,
            "end": end
        })

        if end_word >= len(matches):
            break

        start_word = end_word - overlap_words

    return chunks


# ============================================================
# 4. SENTENCE
# ============================================================

def sentence(text):

    pattern = r"(?<=[.!?])\s+"

    matches = list(
        re.finditer(pattern, text)
    )

    boundaries = []

    previous = 0

    for match in matches:

        boundaries.append(
            (previous, match.start())
        )

        previous = match.end()

    boundaries.append(
        (previous, len(text))
    )

    chunks = []

    current_start = None
    current_end = None

    for start, end in boundaries:

        if end <= start:
            continue

        sentence_text = text[start:end].strip()

        if not sentence_text:
            continue

        if current_start is None:

            current_start = start
            current_end = end

        elif (
            current_end - current_start
            + (end - start)
            <= CHUNK_SIZE
        ):

            current_end = end

        else:

            chunks.append({
                "text": text[current_start:current_end],
                "start": current_start,
                "end": current_end
            })

            current_start = start
            current_end = end

    if current_start is not None:

        chunks.append({
            "text": text[current_start:current_end],
            "start": current_start,
            "end": current_end
        })

    return chunks


# ============================================================
# 5. PARAGRAPH
# ============================================================

def paragraph(text):

    paragraphs = list(
        re.finditer(
            r"\S(?:.*?\S)(?=\n\s*\n|\Z)",
            text,
            flags=re.S
        )
    )

    chunks = []

    current_start = None
    current_end = None

    for match in paragraphs:

        start = match.start()
        end = match.end()

        if current_start is None:

            current_start = start
            current_end = end

        elif end - current_start <= CHUNK_SIZE:

            current_end = end

        else:

            chunks.append({
                "text": text[current_start:current_end],
                "start": current_start,
                "end": current_end
            })

            current_start = start
            current_end = end

    if current_start is not None:

        chunks.append({
            "text": text[current_start:current_end],
            "start": current_start,
            "end": current_end
        })

    return chunks


# ============================================================
# 6. FIXED TOKEN
# ============================================================

def fixed_token(text, tokenizer):

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True
    )

    offsets = encoded["offset_mapping"]

    tokens_per_chunk = 125
    overlap_tokens = 15

    chunks = []

    start_token = 0

    while start_token < len(offsets):

        end_token = min(
            start_token + tokens_per_chunk,
            len(offsets)
        )

        start = offsets[start_token][0]
        end = offsets[end_token - 1][1]

        if end > start:

            chunks.append({
                "text": text[start:end],
                "start": start,
                "end": end
            })

        if end_token >= len(offsets):
            break

        start_token = end_token - overlap_tokens

    return chunks


# ============================================================
# 7. SEMANTIC
# ============================================================

def semantic(text, model):

    sentence_matches = list(
        re.finditer(
            r"[^.!?]+[.!?]+",
            text,
            flags=re.S
        )
    )

    sentences = []

    for match in sentence_matches:

        sentence_text = match.group().strip()

        if sentence_text:

            sentences.append({
                "text": sentence_text,
                "start": match.start(),
                "end": match.end()
            })

    if not sentences:
        return []

    sentence_embeddings = model.encode(
        [s["text"] for s in sentences],
        normalize_embeddings=True,
        show_progress_bar=True
    )

    chunks = []

    current_start = sentences[0]["start"]
    current_end = sentences[0]["end"]

    for i in range(1, len(sentences)):

        similarity = float(
            np.dot(
                sentence_embeddings[i - 1],
                sentence_embeddings[i]
            )
        )

        proposed_end = sentences[i]["end"]

        if (
            similarity >= SEMANTIC_THRESHOLD
            and
            proposed_end - current_start
            <= CHUNK_SIZE * 1.5
        ):

            current_end = proposed_end

        else:

            chunks.append({
                "text": text[current_start:current_end],
                "start": current_start,
                "end": current_end
            })

            current_start = sentences[i]["start"]
            current_end = sentences[i]["end"]

    chunks.append({
        "text": text[current_start:current_end],
        "start": current_start,
        "end": current_end
    })

    return chunks


# ============================================================
# STATISTIQUES
# ============================================================

def calculate_statistics(chunks, model):

    lengths = [
        len(c["text"])
        for c in chunks
    ]

    if not lengths:

        return {
            "nombre_chunks": 0,
            "longueur_moyenne": 0,
            "ecart_type": 0,
            "similarite_intra": 0,
            "similarite_inter": 0
        }

    sample = chunks[:300]

    embeddings = model.encode(
        [c["text"] for c in sample],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    # Similarité intra :
    # comparaison entre la première et la dernière phrase
    intra_values = []

    for chunk in sample:

        sentences = re.split(
            r"(?<=[.!?])\s+",
            chunk["text"].strip()
        )

        sentences = [
            s for s in sentences
            if s.strip()
        ]

        if len(sentences) >= 2:

            emb = model.encode(
                sentences[:2] + sentences[-1:],
                normalize_embeddings=True,
                show_progress_bar=False
            )

            intra_values.append(
                float(
                    np.dot(
                        emb[0],
                        emb[-1]
                    )
                )
            )

    intra = (
        float(np.mean(intra_values))
        if intra_values
        else 0
    )

    # Similarité inter
    if len(embeddings) >= 2:

        matrix = cosine_similarity(embeddings)

        upper = matrix[
            np.triu_indices(
                len(matrix),
                k=1
            )
        ]

        inter = float(
            np.mean(upper)
        )

    else:

        inter = 0

    return {
        "nombre_chunks": len(chunks),
        "longueur_moyenne": float(
            np.mean(lengths)
        ),
        "ecart_type": float(
            np.std(lengths)
        ),
        "similarite_intra": intra,
        "similarite_inter": inter
    }


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("CHUNKING - VERSION UNIFIEE")
print("=" * 80)

text = INPUT_PATH.read_text(
    encoding="utf-8"
)

print(
    f"\nDocument : {INPUT_PATH}"
)

print(
    f"Longueur : {len(text):,} caractères"
)

print("\nChargement du modèle d'embedding...")

model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("\nChargement du tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_MODEL
)

methods = {
    "Recursive Character": recursive_character(text),
    "Character": character(text),
    "Word": word(text),
    "Sentence": sentence(text),
    "Paragraph": paragraph(text),
    "Fixed Token": fixed_token(text, tokenizer),
    "Semantic": semantic(text, model)
}

results = {}

for name, chunks in methods.items():

    print(
        f"\n{name:<22} : "
        f"{len(chunks):>5} chunks"
    )

    stats = calculate_statistics(
        chunks,
        model
    )

    results[name] = {
        "statistics": stats,
        "chunks": chunks
    }

    print(
        f"  Moyenne : "
        f"{stats['longueur_moyenne']:.2f}"
    )

    print(
        f"  Écart-type : "
        f"{stats['ecart_type']:.2f}"
    )

    print(
        f"  Intra : "
        f"{stats['similarite_intra']:.4f}"
    )

    print(
        f"  Inter : "
        f"{stats['similarite_inter']:.4f}"
    )


OUTPUT_PATH.write_text(
    json.dumps(
        results,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)

print("\n" + "=" * 80)
print("RESULTATS")
print("=" * 80)

print(
    f"{'Méthode':<22}"
    f"{'Chunks':>8}"
    f"{'Moyenne':>12}"
    f"{'Std':>10}"
    f"{'Intra':>10}"
    f"{'Inter':>10}"
)

print("-" * 80)

for name, data in results.items():

    s = data["statistics"]

    print(
        f"{name:<22}"
        f"{s['nombre_chunks']:>8}"
        f"{s['longueur_moyenne']:>12.2f}"
        f"{s['ecart_type']:>10.2f}"
        f"{s['similarite_intra']:>10.4f}"
        f"{s['similarite_inter']:>10.4f}"
    )

print("-" * 80)

print(
    f"\nFichier créé : {OUTPUT_PATH}"
)

print("\nCHUNKING TERMINE")

