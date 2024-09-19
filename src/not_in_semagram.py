import query_semagram as qs
from eval_measures import read_generated_concepts
from pathlib import Path
import json
import glob
import os

main_folder = "../results"
folders = ["zero_shot_v1", "zero_shot_v2", "zero_shot_v3"]

for folder in folders:
    f = os.path.join(main_folder, folder)
    output_folder = os.path.join(os.path.join(main_folder, "semagram"), folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in glob.glob(os.path.join(f, "**/*.jsonl"), recursive=True):
        f_name = Path(file).stem

        with open(file, "r") as reader, open(
            os.path.join(output_folder, f"{f_name}"), "w"
        ) as writer:
            for line in reader:
                data = json.loads(line.strip())
                concepts = read_generated_concepts(data["result"])
                concepts_category = qs.get_concepts_by_category(data["cat"])
                concepts_eval = qs.get_concept_with_slot_value(
                    concepts_category, data["slot"], data["value"]
                )
                not_in_semagram = [c for c in concepts if c not in concepts_eval]
                d = {
                    "cat": data["cat"],
                    "slot": data["slot"],
                    "value": data["value"],
                    "prompt": data["prompt"],
                    "not_in_semagram": not_in_semagram,
                }

                writer.write(f"{json.dumps(d)}\n")
