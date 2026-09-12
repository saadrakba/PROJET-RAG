from pathlib import Path
import json

INPUT_PATH = Path("backend/data/33_cleaned_final.txt")
OUTPUT_PATH = Path("backend/evaluation/ground_truth.json")

text = INPUT_PATH.read_text(encoding="utf-8")

GROUND_TRUTH = {
    "q1": {
        "question": "Qu'est-ce qu'un réchauffement stratosphérique majeur ?",
        "references": [
            {
                "paragraph_id": 15,
                "start": 17778,
                "end": 19885
            },
            {
                "paragraph_id": 17,
                "start": 20977,
                "end": 23606
            }
        ]
    },

    "q2": {
        "question": "Quels sont les différents types de réchauffement stratosphérique ?",
        "references": [
            {
                "paragraph_id": 17,
                "start": 20977,
                "end": 23606
            },
            {
                "paragraph_id": 19,
                "start": 24299,
                "end": 26038
            },
            {
                "paragraph_id": 95,
                "start": 97548,
                "end": 100839
            }
        ]
    },

    "q3": {
        "question": "Comment évolue la température dans la stratosphère lors d'un réchauffement stratosphérique ?",
        "references": [
            {
                "paragraph_id": 15,
                "start": 17778,
                "end": 19885
            },
            {
                "paragraph_id": 17,
                "start": 20977,
                "end": 23606
            }
        ]
    },

    "q4": {
        "question": "Quels phénomènes physiques influencent la température de la stratosphère ?",
        "references": [
            {
                "paragraph_id": 2,
                "start": 1875,
                "end": 2642
            },
            {
                "paragraph_id": 4,
                "start": 2974,
                "end": 4210
            },
            {
                "paragraph_id": 12,
                "start": 11597,
                "end": 14252
            },
            {
                "paragraph_id": 54,
                "start": 62730,
                "end": 65757
            }
        ]
    },

    "q5": {
        "question": "Quel est le rôle des ondes planétaires dans les réchauffements stratosphériques ?",
        "references": [
            {
                "paragraph_id": 17,
                "start": 20977,
                "end": 23606
            },
            {
                "paragraph_id": 95,
                "start": 97548,
                "end": 100839
            },
            {
                "paragraph_id": 156,
                "start": 164399,
                "end": 166726
            }
        ]
    }
}

for data in GROUND_TRUTH.values():
    for ref in data["references"]:
        ref["text"] = text[ref["start"]:ref["end"]].strip()

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH.write_text(
    json.dumps(GROUND_TRUTH, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("=" * 80)
print("VERITE TERRAIN CORRIGEE")
print("=" * 80)

for qid, data in GROUND_TRUTH.items():
    print(f"\n{qid.upper()} - {data['question']}")
    print(f"Nombre de passages de référence : {len(data['references'])}")

    for i, ref in enumerate(data["references"], 1):
        preview = " ".join(ref["text"].split())

        if len(preview) > 180:
            preview = preview[:180] + "..."

        print(
            f"  Reference {i} | "
            f"paragraph_id={ref['paragraph_id']} | "
            f"span={ref['start']}:{ref['end']}"
        )
        print(f"  {preview}")

print("\n" + "=" * 80)
print(f"Fichier : {OUTPUT_PATH}")
print("=" * 80)
