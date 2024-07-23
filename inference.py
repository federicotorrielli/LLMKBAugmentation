import gc
import os
import re
from typing import Dict, Generator, List, Tuple

import requests
import torch
import ujson
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

MODEL_NAMES = [
    # All the models are in the top 20 of the LLM HF open leaderboard
    # Tests were run on 8xA100-80G or one A100-80G depending on the model size
    "meta-llama/Meta-Llama-3-70B",  # pretrained | 70B
    "meta-llama/Meta-Llama-3-70B-Instruct",  # instruct | 70B
    "Sao10K/L3-8B-Stheno-v3.2",  # storytelling/instruct | 8B
    "microsoft/Phi-3-medium-4k-instruct",  # instruct | 14B
    "mistralai/Mistral-7B-Instruct-v0.3",  # instruct | 7B
    "CohereForAI/c4ai-command-r-plus",  # instruct | 104B
    "ai21labs/Jamba-v0.1",  # pretrained MoE | 52B | unquantized | Mamba Architecture
    "google/gemma-2-27b-it",  # instruct | 27B | needs to be run with VLLM_ATTENTION_BACKEND=FLASHINFER
]

TASKS = {
    "conceptnet": {
        "RelatedTo": [
            "conceptnet_oneshot_relatedto_commasep.json",
            "conceptnet_oneshot_relatedto_json.json",
            "conceptnet_zeroshot_relatedto_commasep.json",
            "conceptnet_zeroshot_relatedto_json.json",
        ],
        "UsedFor": [
            "conceptnet_oneshot_usedfor_commasep.json",
            "conceptnet_oneshot_usedfor_json.json",
            "conceptnet_zeroshot_usedfor_commasep.json",
            "conceptnet_zeroshot_usedfor_json.json",
        ],
    },
    "framenet": {
        "gloss": [
            f"framenet_{word_type}_{shot_type}_{format_type}_GLOSS.json"
            for word_type in ["adjectives", "nouns", "verbs"]
            for shot_type in ["oneshot", "zeroshot"]
            for format_type in ["commasep", "json"]
        ],
        "nogloss": [
            f"framenet_{word_type}_{shot_type}_{format_type}.json"
            for word_type in ["adjectives", "nouns", "verbs"]
            for shot_type in ["oneshot", "zeroshot"]
            for format_type in ["commasep", "json"]
        ],
    },
    "multialignet": [
        f"multialignet_{word_type}_{shot_type}_{format_type}.json"
        for word_type in ["adjectives", "nouns", "verbs"]
        for shot_type in ["oneshot", "zeroshot"]
        for format_type in ["commasep", "json"]
    ],
    "semagram": [
        f"semagram_{shot_type}_{format_type}.json"
        for shot_type in ["oneshot", "zeroshot"]
        for format_type in ["commasep", "json"]
    ],
}

SAMPLING_PARAMS = SamplingParams(top_p=0.95, temperature=0.4, max_tokens=100)


def get_modified_prompt(prompt: str, model_name: str) -> str:
    if "gemma" in model_name.lower() and "-it" in model_name.lower():
        return (
            f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        )
    elif "phi" in model_name.lower():
        return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    elif (
            "meta" in model_name.lower() and "instruct" in model_name.lower()
    ) or "stheno" in model_name.lower():
        return f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "c4ai" in model_name.lower():
        return f"<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
    elif "jamba" in model_name.lower() or "meta" in model_name.lower():
        return f"{prompt}\n"
    return f"[INST]{prompt}[/INST]\n"


def process_file(
        file_path: str, task_type: str, model_name: str
) -> Generator[Tuple, None, None]:
    with open(file_path, "r") as reader:
        data = ujson.load(reader)
        for item in data:
            prompt = get_modified_prompt(item["prompt"], model_name)
            if task_type in ["conceptnet", "multialignet"]:
                yield item["concept_no"], item["concept_name"], prompt
            elif task_type == "framenet":
                yield item["frame_no"], item["frame_name"], prompt
            elif task_type == "semagram":
                yield (
                    item["concept_no"],
                    item["concept_name"],
                    item["concept_criterion"],
                    prompt,
                )


def get_prompts(file_path: str, model_name: str) -> List[str]:
    with open(file_path, "r") as reader:
        return [
            get_modified_prompt(item["prompt"], model_name)
            for item in ujson.load(reader)
        ]


def download_file(url: str, file_path: str) -> None:
    r = requests.get(url, allow_redirects=True)
    with open(file_path, "wb") as f:
        f.write(r.content)


def clean_output(text: str) -> str:
    text = re.sub(r"<\|.*?\|>", "", text)
    patterns = ["<bos>", "<eos>", "[INST]", "[/INST]", "\n"]
    for pattern in patterns:
        text = text.replace(pattern, "")
    return text.strip()


def determine_quantization(model_name):
    model_name_lower = model_name.lower()
    if "awq" in model_name_lower:
        return "awq"
    elif "fp8" in model_name_lower:
        return "fp8"
    else:
        return None


def run_inference(model_names: List[str], tasks: Dict) -> None:
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
            quantization=determine_quantization(model_name),
        )

        for task_type, task_data in tasks.items():
            if isinstance(task_data, dict):
                for subtask, input_files in task_data.items():
                    process_task(
                        model,
                        model_name,
                        task_type,
                        subtask,
                        input_files,
                        downloaded_files,
                    )
            else:
                process_task(
                    model, model_name, task_type, None, task_data, downloaded_files
                )

        destroy_model_parallel()
        del model.llm_engine.model_executor
        del model
        gc.collect()
        torch.cuda.empty_cache()
        # torch.distributed.destroy_process_group()

    for file in downloaded_files:
        os.remove(file)


def process_task(
        model: LLM,
        model_name: str,
        task_type: str,
        subtask: str,
        input_files: List[str],
        downloaded_files: List[str],
) -> None:
    for file_name in input_files:
        file_path = os.path.join("prompts", task_type, subtask or "", file_name)
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            download_file(
                f"https://www.evilscript.eu/upload/files/{file_name}", file_path
            )
            downloaded_files.append(file_path)

        output_dir = os.path.join(
            "results", task_type, subtask or "", model_name.split("/")[-1]
        )
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir,
            f"t_{SAMPLING_PARAMS.temperature}__top_p_{SAMPLING_PARAMS.top_p}__max_new_tokens_{SAMPLING_PARAMS.max_tokens}__file_{file_name.replace('.json', '')}.jsonl",
        )

        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print(
                f"Skipping {file_name} for task {task_type} as it is already processed"
            )
            continue

        print(
            f"Processing {file_name} for task {task_type}"
            + (f" subtask {subtask}" if subtask else "")
        )

        with open(output_file, "w") as writer:
            prompts = get_prompts(file_path, model_name)
            outputs = model.generate(prompts, sampling_params=SAMPLING_PARAMS)
            output_dict = {
                output.prompt: clean_output(output.outputs[0].text)
                for output in outputs
            }

            for item in tqdm(
                    process_file(file_path, task_type, model_name), total=len(prompts)
            ):
                prompt = item[-1]
                result = {"result": output_dict[prompt], "prompt": prompt}

                if task_type == "conceptnet":
                    result["concept_no"], result["concept_name"] = item[:2]
                elif task_type == "framenet":
                    result["frame_no"], result["frame_name"] = item[:2]
                elif task_type == "multialignet":
                    result["concept_no"], result["concept_name"] = item[:2]
                elif task_type == "semagram":
                    (
                        result["concept_no"],
                        result["concept_name"],
                        result["concept_criterion"],
                    ) = item[:3]

                writer.write(ujson.dumps(result) + "\n")


if __name__ == "__main__":
    run_inference(MODEL_NAMES, TASKS)
