import os
import tqdm
import glob
import ujson
import csv
import eval_measures as em
from nltk.corpus import framenet as fn

import nltk
nltk.download('framenet_v17')


def get_concepts_eval(c_name):
    name = c_name.split(' ')[0]
    frame = fn.frame(name)
    values = frame.lexUnit.values()
    values = [x.name.lower() for x in values]
    values = [x.split('.')[0] if (x.endswith('v') or x.endswith('a') or x.endswith('n')) else x for x in values]
    return values


def compute_model_scores(model_name, model_file, output_folder):

    p_1 = 0.
    p_2 = 0.
    p_5 = 0.
    p_10 = 0.

    h_1 = 0.
    h_2 = 0.
    h_5 = 0.
    h_10 = 0.

    acc_1 = 0.
    acc_2 = 0.
    acc_5 = 0.
    acc_10 = 0.

    rec_1 = 0.
    rec_2 = 0.
    rec_5 = 0.
    rec_10 = 0.

    as1 = 0.
    as2 = 0.
    as5 = 0.
    as10 = 0.

    mrr = 0.

    num_q = 0.

    with open(model_file, 'r', encoding='utf-8') as reader:
        for line in reader:
            data = ujson.loads(line.strip())
            concepts = em.read_generated_concepts(data['result'])
            concepts_eval = get_concepts_eval(data['frame_name'])

            if len(concepts) > 0 and concepts_eval and len(concepts_eval) > 0:
                num_q += 1

                p_1 += em.precision_at_k(concepts, concepts_eval, 1)
                p_2 += em.precision_at_k(concepts, concepts_eval, 2)
                p_5 += em.precision_at_k(concepts, concepts_eval, 5)
                p_10 += em.precision_at_k(concepts, concepts_eval, 10)

                h_1 += em.hits_at_k(concepts, concepts_eval, 1)
                h_2 += em.hits_at_k(concepts, concepts_eval, 2)
                h_5 += em.hits_at_k(concepts, concepts_eval, 5)
                h_10 += em.hits_at_k(concepts, concepts_eval, 10)

                acc_1 += em.accuracy_at_k(concepts, concepts_eval, 1)
                acc_2 += em.accuracy_at_k(concepts, concepts_eval, 2)
                acc_5 += em.accuracy_at_k(concepts, concepts_eval, 5)
                acc_10 += em.accuracy_at_k(concepts, concepts_eval, 10)

                rec_1 += em.recall_at_k(concepts, concepts_eval, 1)
                rec_2 += em.recall_at_k(concepts, concepts_eval, 2)
                rec_5 += em.recall_at_k(concepts, concepts_eval, 5)
                rec_10 += em.recall_at_k(concepts, concepts_eval, 10)

                mrr += em.MRR(concepts, concepts_eval)

                as1 += em.asintoto(concepts_eval, 1)
                as2 += em.asintoto(concepts_eval, 2)
                as5 += em.asintoto(concepts_eval, 5)
                as10 += em.asintoto(concepts_eval, 10)

    p_1 /= num_q
    p_2 /= num_q
    p_5 /= num_q
    p_10 /= num_q

    h_1 /= num_q
    h_2 /= num_q
    h_5 /= num_q
    h_10 /= num_q

    acc_1 /= num_q
    acc_2 /= num_q
    acc_5 /= num_q
    acc_10 /= num_q

    rec_1 /= num_q
    rec_2 /= num_q
    rec_5 /= num_q
    rec_10 /= num_q

    as1 /= num_q
    as2 /= num_q
    as5 /= num_q
    as10 /= num_q

    mrr /= num_q

    output_file = os.path.join(output_folder, f'{model_name}.txt')
    with open(output_file, 'w') as writer:
        writer.write(f'As@1: {as1}\n')
        writer.write(f'As@2: {as2}\n')
        writer.write(f'As@5: {as5}\n')
        writer.write(f'As@10: {as10}\n')
        writer.write('\n============================\n')

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

        writer.write(f'ACC@1: {acc_1}\n')
        writer.write(f'ACC@2: {acc_2}\n')
        writer.write(f'ACC@5: {acc_5}\n')
        writer.write(f'ACC@10: {acc_10}\n')
        writer.write('\n============================\n')

        writer.write(f'R@1: {rec_1}\n')
        writer.write(f'R@2: {rec_2}\n')
        writer.write(f'R@5: {rec_5}\n')
        writer.write(f'R@10: {rec_10}\n')
        writer.write('\n============================\n')

        writer.write(f'MRR: {mrr}\n')


def read_model_files_and_write_results(folder, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    t = tqdm.tqdm()
    output_file = os.path.join(output_folder, '{0}.txt')

    for json_file in glob.glob(os.path.join(folder, '**/*.jsonl'), recursive=True):
        model_name = em.get_model_name(json_file)
        if model_name == 'to_annotate' or os.path.exists(output_file.format(model_name)):
            continue

        compute_model_scores(model_name, json_file, output_folder)

        t.update(1)


if __name__ == '__main__':

    for type in ['gloss', 'nogloss']:
        for model in ['c4ai-command-r-plus',
                      'gemma-2-27b-it',
                      'Jamba-v0.1',
                      'L3-8B-Stheno-v3.2',
                      'Meta-Llama-3-70B-AWQ',
                      'Meta-Llama-3-70B-Instruct-FP8',
                      'Mistral-7B-Instruct-v0.3',
                      'Phi-3-medium-4k-instruct']:

            inpt_file = f'/Users/giovanni/PycharmProjects/LLM-Semagram/expanding_horizons/framenet/{type}/{model}'
            otpt_file = f'/Users/giovanni/PycharmProjects/LLM-Semagram/expanding_horizons/scores/framenet/{type}/{model}'

            read_model_files_and_write_results(inpt_file, otpt_file)
