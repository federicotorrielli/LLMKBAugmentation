import glob
import os
import json


def read_file(fname):

    with open(fname, 'r', encoding='utf-8') as reader:
        data = json.load(reader)

    return data


def extract_info(data):

    d = {
        'num_el': data['i'],
        'models': data['models'],
        'timeDiffs': data['timeDiffs'],
        'answers': data['answers'],
        'wrong': data['isWrong'],
        'concepts': [d['selected'] for d in data['dataset']],
        'categories': [d['cat'] for d in data['dataset']],
        'criterias': [d['criteria'] for d in data['dataset']],
    }

    return data['name'], d


def _reduce_for_debugging(data, max_len):

    d = {
        'num_el': data['num_el'],
        'models': data['models'][:max_len],
        'timeDiffs': data['timeDiffs'][:max_len],
        'answers': data['answers'][:max_len],
        'wrong': data['wrong'][:max_len],
        'concepts': data['concepts'][:max_len],
        'categories': data['categories'][:max_len],
        'criterias': data['criterias'][:max_len]
    }

    return d


def read_all_data(folder, debug=False):

    data_dict = dict()
    min_i = 500
    for fname in glob.glob(os.path.join(folder, '**/*.json'), recursive=True):
        name, annotation = extract_info(read_file(fname))
        num_annotated = annotation['num_el']
        if (not debug) and num_annotated < 500:
            print(f'{name} has annotated {num_annotated}/{500} elements. Skipping it.')
            continue

        if debug and num_annotated < min_i:
            min_i = num_annotated

        data_dict[name] = annotation

    if debug:
        for name in data_dict.keys():
            data_dict[name] = _reduce_for_debugging(data_dict[name], min_i)

    return min_i, data_dict
