import gc
import os
import requests
import torch
import ujson
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

# Define the common model names for all tasks
model_names = [
    # "mistralai/Mistral-7B-Instruct-v0.3",
    # "TheBloke/Yi-34B-AWQ",
    # "TheBloke/tulu-2-70B-AWQ",
    # "TheBloke/Aetheria-L2-70B-AWQ",
    "microsoft/Phi-3-medium-4k-instruct"
    # "google/gemma-2-27b-it",
]

# Define input files for different tasks
tasks = {
    "conceptnet": [
        "conceptnet_zero_shot_prompt_v1.json",
        "conceptnet_zero_shot_prompt_v2.json",
        "conceptnet_zero_shot_prompt_v3.json",
        "conceptnet_one_shot_prompt_v1.json",
    ],
    "multialignet": [
        "multialignet_adjectives_oneshot_v1.json",
        "multialignet_adjectives_oneshot_v2.json",
        "multialignet_adjectives_oneshot_v3.json",
        "multialignet_adjectives_zeroshot_v1.json",
        "multialignet_adjectives_zeroshot_v2.json",
        "multialignet_adjectives_zeroshot_v3.json",
        "multialignet_nouns_oneshot_v1.json",
        "multialignet_nouns_oneshot_v2.json",
        "multialignet_nouns_oneshot_v3.json",
        "multialignet_nouns_zeroshot_v1.json",
        "multialignet_nouns_zeroshot_v2.json",
        "multialignet_nouns_zeroshot_v3.json",
        "multialignet_verbs_oneshot_v1.json",
        "multialignet_verbs_oneshot_v2.json",
        "multialignet_verbs_oneshot_v3.json",
        "multialignet_verbs_zeroshot_v1.json",
        "multialignet_verbs_zeroshot_v2.json",
        "multialignet_verbs_zeroshot_v3.json",
    ],
    "semagram": [
        "zero_shot_prompt_v1.json",
        "zero_shot_prompt_v2.json",
        "zero_shot_prompt_v3.json",
        "zero_shot_prompt_v4.json",
        "one_shot_prompt_v1.json",
    ],
}

sampling_params = SamplingParams(top_p=0.95, temperature=0.4, max_tokens=100)
t = sampling_params.temperature
top_p = sampling_params.top_p
max_new_tokens = sampling_params.max_tokens


def get_number_prompts(file_name):
    with open(file_name, "r") as reader:
        all_data = ujson.load(reader)
        return len(all_data)


def prompt_generator(file_name, task_type, model_name):
    with open(file_name, "r") as reader:
        all_data = ujson.load(reader)
        for data in all_data:
            if "gemma" in model_name.lower() and "-it" in model_name.lower():
                modified_prompt = f"<bos><start_of_turn>user\n{data['prompt']}<end_of_turn>\n<start_of_turn>model\n"
            elif "phi" in model_name.lower():
                modified_prompt = f"<|user|>\n{data['prompt']}<|end|>\n<|assistant|>\n"
            else:
                modified_prompt = data["prompt"]

            if task_type == "conceptnet":
                yield data["concept"], modified_prompt
            elif task_type == "multialignet":
                yield data["count"], data["lemmas"], data["wordnet_id"], modified_prompt
            elif task_type == "semagram":
                yield data["cat"], data["slot"], data["value"], modified_prompt


def get_only_prompts(file_name, model_name):
    with open(file_name, "r") as reader:
        all_data = ujson.load(reader)
        if "gemma" in model_name.lower() and "-it" in model_name.lower():
            return [
                f"<bos><start_of_turn>user\n{data['prompt']}<end_of_turn>\n<start_of_turn>model\n"
                for data in all_data
            ]
        if "phi" in model_name.lower():
            return [
                f"<|user|>\n{data['prompt']}<|end|>\n<|assistant|>\n"
                for data in all_data
            ]
        return [data["prompt"] for data in all_data]


def get_simple_model_name(m_name):
    return m_name.split("/")[-1] if "/" in m_name else m_name


def download_file(url, file_name):
    r = requests.get(url, allow_redirects=True)
    open(file_name, "wb").write(r.content)


def run_inference(model_names, tasks):
    downloaded_files = []

    for model_name in model_names:
        f_m_name = get_simple_model_name(model_name)
        print(f"Running model {model_name}")

        model = LLM(
            model_name,
            trust_remote_code=True,
            tensor_parallel_size=torch.cuda.device_count(),
            dtype="auto",
            quantization="awq" if "awq" in model_name.lower() else None,
        )

        for task_type, input_files in tasks.items():
            for file_name in input_files:
                if not os.path.exists(file_name):
                    download_file(
                        f"https://www.evilscript.eu/upload/files/{file_name}", file_name
                    )
                    downloaded_files.append(file_name)

                num_prompts = get_number_prompts(file_name)

                output_dir = os.path.join(task_type, f_m_name)
                os.makedirs(output_dir, exist_ok=True)
                file_output = os.path.join(
                    output_dir,
                    f"t_{t}__top_p_{top_p}__max_new_tokens_{max_new_tokens}__file_{file_name.replace('.json', '')}.jsonl",
                )

                print(
                    f"Processing {file_name} with {num_prompts} prompts for task {task_type}"
                )

                with open(file_output, "w") as writer:
                    generator = prompt_generator(file_name, task_type, model_name)
                    full_list_prompts = get_only_prompts(file_name, model_name)
                    model_outputs = model.generate(
                        full_list_prompts, sampling_params=sampling_params
                    )
                    outputs_dict = {
                        output.prompt: output.outputs[0]
                        .text.replace("<bos>", "")
                        .replace("<start_of_turn>", "")
                        .replace("<end_of_turn>", "")
                        .replace("<eos>", "")
                        .replace("[INST]", "")
                        .replace("[/INST]", "")
                        .replace("<|user|>", "")
                        .replace("<|assistant|>", "")
                        .replace("<|end|>", "")
                        .replace("\n", "")
                        .strip()
                        for output in model_outputs
                    }

                    for prompts in tqdm(generator, total=num_prompts):
                        prompt = prompts[-1]
                        model_output = outputs_dict[prompt]
                        result = {"result": model_output, "prompt": prompt}

                        if task_type == "conceptnet":
                            result.update({"concept": prompts[0]})
                        elif task_type == "multialignet":
                            result.update(
                                {
                                    "count": prompts[0],
                                    "lemmas": prompts[1],
                                    "wordnet_id": prompts[2],
                                }
                            )
                        elif task_type == "semagram":
                            result.update(
                                {
                                    "cat": prompts[0],
                                    "slot": prompts[1],
                                    "value": prompts[2],
                                }
                            )

                        writer.write(ujson.dumps(result) + "\n")

        with torch.no_grad():
            destroy_model_parallel()
            del model
            gc.collect()
            torch.cuda.empty_cache()
            torch.distributed.destroy_process_group()

    # Delete downloaded files
    for file in downloaded_files:
        os.remove(file)


# Run inference for each model and each task
run_inference(model_names, tasks)
