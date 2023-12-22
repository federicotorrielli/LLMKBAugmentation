import gc

import torch
import ujson
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

model_names = [
    "TheBloke/Yi-34B-AWQ",
    "TheBloke/tulu-2-70B-AWQ",
    "TheBloke/Aetheria-L2-70B-AWQ",
]

input = "output.json"

sampling_params = SamplingParams(top_p=0.95, temperature=0.4, max_tokens=20)

t = sampling_params.temperature
top_p = sampling_params.top_p
max_new_tokens = sampling_params.max_tokens

for model_name in model_names:
    model = LLM(
        model_name,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype="auto",
        quantization="awq",
    )

    prompt_list = []
    with open(input, "r") as reader:
        all_data = ujson.load(reader)
        for data in all_data:
            prompt_list.append(data["prompt"])

    model_outputs = model.generate(prompt_list, sampling_params=sampling_params)

    outputs_dict = {}
    for output in model_outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        outputs_dict[prompt] = generated_text

    with open("output.jsonl", "w") as writer:
        for prompt in prompt_list:
            model_output = outputs_dict[prompt]
            json_dump = ujson.dumps(
                {
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
