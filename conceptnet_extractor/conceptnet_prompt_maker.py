import csv
import ujson as json

# Change this to the path of your CSV file with related concepts
input_csv_path = "usedfor_concepts.csv"
output_json_path = "prompts.json"

prompts = []

# A dictionary to hold the list of related concepts for each unique concept1
related_concepts_dict = {}

# Read the CSV file and populate the dictionary
with open(input_csv_path, "r", encoding="utf-8") as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        concept1, concept2 = row
        if concept1 not in related_concepts_dict:
            related_concepts_dict[concept1] = []
        related_concepts_dict[concept1].append(concept2)

# Create prompts for each concept1 with at least 10 related concepts
for concept1, related_concepts in related_concepts_dict.items():
    if len(related_concepts) >= 10:
        # Take the first 10 unique related concepts
        related_concepts_list = related_concepts[:10]
        prompt_text = f"### Instruction\nGiven the concept '{concept1}', list 10 concepts for which {concept1} is used for, in the form of a comma-separated list.\n### Output\nResponse:"
        # prompt_text = f"### Instruction\nGiven the concept '{concept1}' (source: ConceptNet), list 10 ConceptNet concepts for which {concept1} is used for, in the form of a comma-separated list.\n### Output\nResponse:"
        # prompt_text = f'### Instruction\nGiven the concept \'{concept1}\', list 10 concepts for which {concept1} is used for, in the form of a comma-separated list.\n### Example for the concept \'Apple\'\nResponse: eating, making apple pie, sate hunger, bait a trap, computing, dessert, enjoy the fruit, getting good carbs, growing apple trees, munching\n### Output for the concept \'{concept1}\'\nResponse:'
        # prompt_text = f"### Instruction\nGiven the concept '{concept1}', list 10 concepts for which {concept1} is used for, in the form of a python-like list.\n### Output\nResponse:"
        prompt = {"concept": concept1, "prompt": prompt_text}
        prompts.append(prompt)

# Write the prompts to a JSON file
with open(output_json_path, "w", encoding="utf-8") as jsonfile:
    json.dump(prompts, jsonfile, ensure_ascii=False, indent=4)

print(f"Prompts have been written to {output_json_path}")
