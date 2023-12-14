import os


def get_num_rows(file):
    with open(file, 'r') as reader:
        num_rows = 0
        for _ in reader:
            num_rows += 1

        return num_rows


def check_files(f1, f2):
    n_rows_1 = get_num_rows(f1)
    n_rows_2 = get_num_rows(f2)

    assert n_rows_1 == n_rows_2, f'{n_rows_1} != {n_rows_2}'


if __name__ == '__main__':

    prompt_list = ['./prompts/zero_shot_prompt_v1.json',
                   './prompts/zero_shot_prompt_v2.json',
                   './prompts/zero_shot_prompt_v3.json',
                   './prompts/zero_shot_prompt_v4.json'
                   ]

    for p1 in prompt_list:
        for p2 in prompt_list:
            if p1 != p2:
                try:
                    check_files(p1, p2)
                except Exception as e:
                    print(e)
