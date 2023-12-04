import ujson
import torch
import time
import gc
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

model_names = [
    "TheBloke/Yi-34B-AWQ",
    "TheBloke/tulu-2-70B-AWQ",
    "TheBloke/Aetheria-L2-70B-AWQ",
]

input_f = [
    "zero_shot_prompt_v1.json",
    "zero_shot_prompt_v2.json",
    "zero_shot_prompt_v3.json",
]

sampling_params = SamplingParams(top_p=0.9, temperature=0.35, max_tokens=50)

t = sampling_params.temperature
top_p = sampling_params.top_p
max_new_tokens = sampling_params.max_tokens


def get_number_prompts(file_name):
    with open(file_name, "r") as reader:
        all_data = ujson.load(reader)
        counter = 0
        for _ in all_data:
            counter += 1
        return counter


def prompt_generator(file_name):
    with open(file_name, "r") as reader:
        all_data = ujson.load(reader)
        for data in all_data:
            yield data["cat"], data["slot"], data["value"], data["prompt"]


def get_only_prompts(file_name):
    with open(file_name, "r") as reader:
        all_data = ujson.load(reader)
        output = []
        for data in all_data:
            output.append(data["prompt"])
        return output


def get_all_prompts(file_name):
    with open(file_name, "r") as reader:
        all_data = ujson.load(reader)
        output = []
        for data in all_data:
            output.append((data["cat"], data["slot"], data["value"], data["prompt"]))
        return output


def get_simple_model_name(m_name):
    if "/" in m_name:
        m_name = m_name.split("/")[-1]
    return m_name


for file_name in input_f:
    num_prompts = get_number_prompts(file_name)
    for model_name in model_names:
        f_m_name = get_simple_model_name(model_name)
        file_output = f"{f_m_name}__t_{t}__top_p_{top_p}__max_new_tokens_{max_new_tokens}__file_{file_name}.jsonl"

        print(f"Running {model_name} with {num_prompts} prompts")

        if "awq" in model_name.lower():
            model = LLM(
                model_name,
                trust_remote_code=True,
                tensor_parallel_size=torch.cuda.device_count(),
                dtype="auto",
                quantization="awq",
            )
        else:
            model = LLM(
                model_name,
                trust_remote_code=True,
                tensor_parallel_size=torch.cuda.device_count(),
                dtype="auto",
            )

        with open(file_output, "w") as writer:
            generator = prompt_generator(file_name)
            full_list_prompts = get_only_prompts(file_name)
            model_outputs = model.generate(
                full_list_prompts, sampling_params=sampling_params
            )
            outputs_dict = {}
            for output in model_outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                outputs_dict[prompt] = generated_text
            for cat, slot, value, prompt in tqdm(generator, total=num_prompts):
                model_output = outputs_dict[prompt]
                json_dump = ujson.dumps(
                    {
                        "cat": cat,
                        "slot": slot,
                        "value": value,
                        "prompt": prompt,
                        "result": model_output,
                    }
                )
                writer.write(json_dump + "\n")
            with torch.no_grad():
                destroy_model_parallel()
                del model
                gc.collect()
                torch.cuda.empty_cache()
                torch.distributed.destroy_process_group()
