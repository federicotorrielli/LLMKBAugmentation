import gc
import os
import re

import requests
import torch
import ujson
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

MODEL_NAMES = [
    "TechxGenus/Meta-Llama-3-70B-AWQ",  # foundational | 70B | AWQ
    "TechxGenus/Meta-Llama-3-70B-Instruct-AWQ",  # instruct | 70B | AWQ
    "Sao10K/L3-8B-Stheno-v3.2",  # storytelling/instruct | 8B | unquantized
    "microsoft/Phi-3-medium-4k-instruct",  # instruct | 14B | unquantized
    "mistralai/Mistral-7B-Instruct-v0.3",  # instruct | 7B | unquantized
    "google/gemma-2-27b-it"  # instruct | 27B | unquantized - need to be run with VLLM_ATTENTION_BACKEND=FLASHINFER
]

TASKS = {
    "conceptnet": [
        "conceptnet_zero_shot_prompt_usedfor_v1.json",
        "conceptnet_zero_shot_prompt_usedfor_v2.json",
        "conceptnet_zero_shot_prompt_usedfor_v3.json",
        "conceptnet_one_shot_prompt_usedfor_v1.json",
        "conceptnet_zero_shot_prompt_relatedto_v1.json",
        "conceptnet_zero_shot_prompt_relatedto_v2.json",
        "conceptnet_zero_shot_prompt_relatedto_v3.json",
        "conceptnet_one_shot_prompt_relatedto_v1.json",
    ],
    "multialignet": [
        f"multialignet_{word_type}_{shot_type}_v{v}.json"
        for word_type in ["adjectives", "nouns", "verbs"]
        for shot_type in ["oneshot", "zeroshot"]
        for v in range(1, 4)
    ],
    "semagram": [
        f"{'zero' if i < 4 else 'one'}_shot_prompt_v{i}.json" for i in range(1, 6)
    ],
}

SAMPLING_PARAMS = SamplingParams(top_p=0.95, temperature=0.4, max_tokens=100)


def get_modified_prompt(prompt, model_name):
    if "gemma" in model_name.lower() and "-it" in model_name.lower():
        return (
            f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        )
    elif "phi" in model_name.lower():
        return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    elif ("meta" in model_name.lower() and "instruct" in model_name.lower()) or "stheno" in model_name.lower():
        return f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt


def process_file(file_name, task_type, model_name):
    with open(file_name, "r") as reader:
        data = ujson.load(reader)
        for item in data:
            prompt = get_modified_prompt(item["prompt"], model_name)
            if task_type == "conceptnet":
                yield item["concept"], prompt
            elif task_type == "multialignet":
                yield item["count"], item["lemmas"], item["wordnet_id"], prompt
            elif task_type == "semagram":
                yield item["cat"], item["slot"], item["value"], prompt


def get_prompts(file_name, model_name):
    with open(file_name, "r") as reader:
        return [
            get_modified_prompt(item["prompt"], model_name)
            for item in ujson.load(reader)
        ]


def download_file(url, file_name):
    r = requests.get(url, allow_redirects=True)
    with open(file_name, "wb") as f:
        f.write(r.content)


def clean_output(text):
    # Remove everything between <|...|>
    text = re.sub(r"<\|.*?\|>", "", text)

    # Remove other specific patterns
    patterns = ["<bos>", "<eos>", "[INST]", "[/INST]", "\n"]
    for pattern in patterns:
        text = text.replace(pattern, "")

    return text.strip()


def run_inference(model_names, tasks):
    downloaded_files = []

    for model_name in model_names:
        print(f"Running model {model_name}")
        if "gemma" in model_name.lower():
            os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
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

                output_dir = os.path.join(task_type, model_name.split("/")[-1])
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(
                    output_dir,
                    f"t_{SAMPLING_PARAMS.temperature}__top_p_{SAMPLING_PARAMS.top_p}__max_new_tokens_{SAMPLING_PARAMS.max_tokens}__file_{file_name.replace('.json', '')}.jsonl",
                )

                print(f"Processing {file_name} for task {task_type}")

                with open(output_file, "w") as writer:
                    prompts = get_prompts(file_name, model_name)
                    outputs = model.generate(prompts, sampling_params=SAMPLING_PARAMS)
                    output_dict = {
                        output.prompt: clean_output(output.outputs[0].text)
                        for output in outputs
                    }

                    for item in tqdm(
                            process_file(file_name, task_type, model_name),
                            total=len(prompts),
                    ):
                        prompt = item[-1]
                        result = {"result": output_dict[prompt], "prompt": prompt}

                        if task_type == "conceptnet":
                            result["concept"] = item[0]
                        elif task_type == "multialignet":
                            result.update(
                                {
                                    "count": item[0],
                                    "lemmas": item[1],
                                    "wordnet_id": item[2],
                                }
                            )
                        elif task_type == "semagram":
                            result.update(
                                {"cat": item[0], "slot": item[1], "value": item[2]}
                            )

                        writer.write(ujson.dumps(result) + "\n")

        with torch.no_grad():
            destroy_model_parallel()
            del model
            gc.collect()
            torch.cuda.empty_cache()
            torch.distributed.destroy_process_group()

    for file in downloaded_files:
        os.remove(file)


run_inference(MODEL_NAMES, TASKS)
