from pathlib import Path
import re
import json

INPUT_PATH = Path("backend/data/33_cleaned_final.txt")
OUTPUT_PATH = Path("backend/evaluation/ground_truth_candidates.json")

QUESTIONS = [
    {
        "id": "q1",
        "question": "Qu'est-ce qu'un réchauffement stratosphérique majeur ?",
        "keywords": [
            "réchauffement stratosphérique majeur",
            "échauffement stratosphérique majeur",
            "température",
            "vortex polaire",
            "gradient"
        ]
    },
    {
        "id": "q2",
        "question": "Quels sont les différents types de réchauffement stratosphérique ?",
        "keywords": [
            "types",
            "réchauffement stratosphérique",
            "échauffement stratosphérique",
            "soudain",
            "majeur",
            "mineur"
        ]
    },
    {
        "id": "q3",
        "question": "Comment évolue la température dans la stratosphère lors d'un réchauffement stratosphérique ?",
        "keywords": [
            "température",
            "augmente",
            "augmentation",
            "réchauffement",
            "stratosphère",
            "anomalie"
        ]
    },
    {
        "id": "q4",
        "question": "Quels phénomènes physiques influencent la température de la stratosphère ?",
        "keywords": [
            "température",
            "stratosphère",
            "rayonnement",
            "radiatif",
            "ondes planétaires",
            "activité dynamique",
            "chauffage"
        ]
    },
    {
        "id": "q5",
        "question": "Quel est le rôle des ondes planétaires dans les réchauffements stratosphériques ?",
        "keywords": [
            "ondes planétaires",
            "ondes",
            "réchauffement stratosphérique",
            "vortex polaire",
            "propagation",
            "activité ondulatoire",
            "déferlement"
        ]
    }
]

text = INPUT_PATH.read_text(encoding="utf-8")

# Découpage en paragraphes en conservant les positions
paragraphs = []
for match in re.finditer(r"\S(?:.*?\S)(?=\n\s*\n|\Z)", text, flags=re.S):
    content = match.group().strip()
    if len(content) >= 150:
        paragraphs.append({
            "text": content,
            "start": match.start(),
            "end": match.end()
        })

def normalize(s):
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s

results = {}

for q in QUESTIONS:
    scored = []

    for i, p in enumerate(paragraphs):
        p_norm = normalize(p["text"])

        score = 0
        found = []

        for keyword in q["keywords"]:
            kw = normalize(keyword)
            if kw in p_norm:
                score += 1
                found.append(keyword)

        # Bonus pour plusieurs occurrences du concept principal
        if "réchauffement stratosphérique" in p_norm:
            score += 2

        if score > 0:
            scored.append({
                "paragraph_id": i,
                "score": score,
                "keywords_found": found,
                "start": p["start"],
                "end": p["end"],
                "text": p["text"]
            })

    scored.sort(key=lambda x: (-x["score"], x["paragraph_id"]))

    # Garder les 10 meilleurs candidats
    results[q["id"]] = {
        "question": q["question"],
        "candidates": scored[:10]
    }

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(
    json.dumps(results, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("=" * 80)
print("CANDIDATS DE VERITE TERRAIN")
print("=" * 80)

for qid, data in results.items():
    print(f"\n{qid.upper()} - {data['question']}")
    print("-" * 80)

    for rank, candidate in enumerate(data["candidates"][:5], 1):
        preview = re.sub(r"\s+", " ", candidate["text"])
        if len(preview) > 500:
            preview = preview[:500] + "..."

        print(f"\n[{rank}] Score={candidate['score']} | "
              f"paragraph_id={candidate['paragraph_id']} | "
              f"positions={candidate['start']}:{candidate['end']}")
        print(f"Mots trouvés : {', '.join(candidate['keywords_found'])}")
        print(preview)

print("\n" + "=" * 80)
print(f"Fichier créé : {OUTPUT_PATH}")
print("=" * 80)
