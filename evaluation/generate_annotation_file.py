import glob
import json
import os
import random
import regex as re
from pathlib import Path
from evaluation import category_collection as cc


def load_dict():

    d = dict()
    results = cc.find({})
    for result in results:
        d.setdefault(result['category'], set())
        d[result['category']].add(result['concept'])

    return d


def read_results(file):
    with open(file, "r", encoding='utf8') as f:
        results = []
        for line in f:
            results.append(json.loads(line.strip()))

    return results


def extract_concepts(results, category_concept_dict):

    cleaned_res = []
    for result in results:
        concepts_category = category_concept_dict[result['cat']]
        words = [x.lower().strip() for x in re.findall(r'[a-zA-Z ]+', result["result"])]
        words = [x for x in words if x and len(x) <= 20]
        words = [x for x in words if x not in concepts_category]
        result["concepts"] = words
        cleaned_res.append(result)

    return cleaned_res


def extract_criteria(prompt_instruction):
    criteria_regex = r"Criteria: .*"
    match = re.search(criteria_regex, prompt_instruction)
    return match.group()


def get_model_name(f_name):
    return Path(f_name).stem.split('__')[0].lower()


def get_random_concept(list_of_concepts, extracted_concepts):

    if len(list_of_concepts) == 0:
        return None

    list_of_possible_concepts = [x for x in list_of_concepts if x not in extracted_concepts]
    if len(list_of_possible_concepts) > 0:
        return random.choice(list_of_possible_concepts)

    return random.choice(list_of_concepts)


def __core(json_file, cc_dict, extracted_concepts):

    selected = []

    count = 0
    gen_flag = False
    spec_flag = False
    m_name = get_model_name(json_file)
    if m_name == 'to_annotate': return None

    results = read_results(json_file)
    results = extract_concepts(results, cc_dict)
    random.shuffle(results)

    for result in results:
        if count == 100:
            break

        slot = result["slot"]
        selected_concept = get_random_concept(result['concepts'], extracted_concepts)
        if selected_concept:

            if slot == 'generalization' and gen_flag: continue
            if slot == 'specialization' and spec_flag: continue

            if slot == 'specialization':
                spec_flag = True
            if slot == 'generalization':
                gen_flag = True

            count += 1

            criteria = extract_criteria(result['prompt'])
            extracted_concepts.add(selected_concept)
            result['selected'] = selected_concept
            result['model'] = m_name
            result['criteria'] = criteria
            selected.append(result)

    return selected


def read_all_files(folder):

    extracted_concepts = set()
    selected = []

    cc_dict = load_dict()

    for json_file in glob.glob(os.path.join(folder, '**/*.jsonl'), recursive=True):

        selected += __core(json_file, cc_dict, extracted_concepts)

    random.shuffle(selected)
    return selected


def read_all_models(models):

    extracted_concepts = set()
    selected = []

    cc_dict = load_dict()

    for model in models:

        selected += __core(model, cc_dict, extracted_concepts)

    random.shuffle(selected)
    return selected



def write_json_file(selected_elements, output_file):

    with open(output_file, 'w', encoding='utf-8') as writer:
        for elem in selected_elements:
            writer.write(f"{json.dumps(elem)}\n")


if __name__ == '__main__':

    input_file = '../results/zero_shot - v1 prompt/7B'
    output_file = '../manual_evaluation/7B_to_annotate.jsonl'

    selected_concepts_for_annotation = read_all_files(input_file)
    print(f"# elements to annotate: {len(selected_concepts_for_annotation)}")
    write_json_file(selected_concepts_for_annotation, output_file)
