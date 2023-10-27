import json
import regex as re
from evaluation import query_semagram as qs
from pathlib import Path
from glob import glob
import os
from tqdm import tqdm


def read_generated_concepts(concept_string):
    if '\n\n' in concept_string:
        concept_string = concept_string.split('\n\n')[0]
    elif '###' in concept_string:
        concept_string = concept_string.split('###')[0]

    words = [x.lower().strip() for x in re.findall(r'[a-zA-Z ]+',concept_string)]
    words = [x for x in words if x]
    return words


def precision_at_k(target_list, gold_list, k):

    target_list = target_list[:k]
    found_k = [elem for elem in target_list if elem in gold_list]

    num_found = len(found_k)
    num_target = len(target_list)

    if (num_found == 0) or (num_target == 0):
        return 0

    return num_found / num_target


def hits_at_k(target_list, gold_list, k):

    target_list = target_list[:k]
    found_k = [elem for elem in target_list if elem in gold_list]

    num_found = len(found_k)
    num_gold = len(gold_list)

    if (num_found == 0) or (num_gold == 0):
        return 0

    return num_found / num_gold


def MRR(target_list, gold_list):

    index = 0
    for rank, elem in enumerate(target_list):
        if elem in gold_list:
            index = rank
            break

    if index == 0:
        return 0

    return 1./index


def AP(target_list, gold_list, k):

    if k == 1:
        return precision_at_k(target_list, gold_list, 1)

    return precision_at_k(target_list, gold_list, k) + AP(target_list, gold_list, k-1)


def get_model_name(f_name):
    return Path(f_name).stem.lower() #.split('__')[0].lower()


def compute_model_scores(model_name, model_file):

    p_1 = 0.
    p_2 = 0.
    p_5 = 0.
    p_10 = 0.

    h_1 = 0.
    h_2 = 0.
    h_5 = 0.
    h_10 = 0.

    #ap_1 = 0.
    #ap_2 = 0.
    #ap_5 = 0.
    #ap_10 = 0.

    mrr = 0.

    num_q = 0.

    with open(model_file, 'r', encoding='utf-8') as reader:
        for line in reader:
            num_q += 1
            data = json.loads(line.strip())
            concepts = read_generated_concepts(data['result'])
            concepts_category = qs.get_concepts_by_category(data['cat'])
            concepts_eval = qs.get_concept_with_slot_value(concepts_category, data['slot'], data['value'])

            if len(concepts) > 0 and concepts_eval and len(concepts_eval) > 0:

                p_1 += precision_at_k(concepts, concepts_eval, 1)
                p_2 += precision_at_k(concepts, concepts_eval, 2)
                p_5 += precision_at_k(concepts, concepts_eval, 5)
                p_10 += precision_at_k(concepts, concepts_eval, 10)

                h_1 += hits_at_k(concepts, concepts_eval, 1)
                h_2 += hits_at_k(concepts, concepts_eval, 2)
                h_5 += hits_at_k(concepts, concepts_eval, 5)
                h_10 += hits_at_k(concepts, concepts_eval, 10)

                #ap_1 += AP(concepts, concepts_eval, 1)
                #ap_2 += AP(concepts, concepts_eval, 2) / 2.
                #ap_5 += AP(concepts, concepts_eval, 5) / 5.
                #ap_10 += AP(concepts, concepts_eval, 10) / 10.

                mrr += MRR(concepts, concepts_eval)

    p_1 /= num_q
    p_2 /= num_q
    p_5 /= num_q
    p_10 /= num_q

    h_1 /= num_q
    h_2 /= num_q
    h_5 /= num_q
    h_10 /= num_q

    #map_1 = ap_1 / num_q
    #map_2 = ap_2 / num_q
    #map_5 = ap_5 / num_q
    #map_10 = ap_10 / num_q

    mrr /= num_q

    with open(f'../../data/prompts/zero_shot_v1/results/{model_name}.txt', 'w') as writer:
        writer.write(f'P@1: {p_1}\n')
        writer.write(f'P@2: {p_2}\n')
        writer.write(f'P@5: {p_5}\n')
        writer.write(f'P@10: {p_10}\n')
        writer.write('\n============================\n')
        writer.write(f'H@1: {h_1}\n')
        writer.write(f'H@2: {h_2}\n')
        writer.write(f'H@5: {h_5}\n')
        writer.write(f'H@10: {h_10}\n')
        writer.write('\n============================\n')
        #writer.write(f'MAP@1: {map_1}\n')
        #writer.write(f'MAP@2: {map_2}\n')
        #writer.write(f'MAP@5: {map_5}\n')
        #writer.write(f'MAP@10: {map_10}\n')
        #writer.write('\n============================\n')
        writer.write(f'MRR: {mrr}\n')


def read_model_files_and_write_results(folder):

    t = tqdm()
    output_file = '../../data/prompts/zero_shot_v1/results/{0}.txt'

    for json_file in glob(os.path.join(folder, '**/*.jsonl'), recursive=True):
        model_name = get_model_name(json_file)
        if model_name == 'to_annotate' or os.path.exists(output_file.format(model_name)):
            continue

        compute_model_scores(model_name, json_file)

        t.update(1)


if __name__ == '__main__':

    for size in ['7B', '13B', '30B']:
        main_folder = f'../../data/prompts/zero_shot_v1/{size}'
        read_model_files_and_write_results(main_folder)