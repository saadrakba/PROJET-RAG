import json
from pathlib import Path

OUTPUT_PATH = Path("backend/evaluation/questions.json")

questions = [
    {
        "id": 1,
        "question": "Qu'est-ce qu'un réchauffement stratosphérique majeur ?"
    },
    {
        "id": 2,
        "question": "Quels sont les différents types de réchauffement stratosphérique ?"
    },
    {
        "id": 3,
        "question": "Comment évolue la température dans la stratosphère lors d'un réchauffement stratosphérique ?"
    },
    {
        "id": 4,
        "question": "Quels phénomènes physiques influencent la température de la stratosphère ?"
    },
    {
        "id": 5,
        "question": "Quel est le rôle des ondes planétaires dans les réchauffements stratosphériques ?"
    }
]

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH.write_text(
    json.dumps(
        questions,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)

print("=" * 60)
print("QUESTIONS D'EVALUATION CRÉÉES")
print("=" * 60)

for item in questions:
    print(f"{item['id']}. {item['question']}")

print()
print(f"Fichier : {OUTPUT_PATH}")
