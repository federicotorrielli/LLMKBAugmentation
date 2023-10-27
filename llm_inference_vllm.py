import ujson
import torch
import gc
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

# TODO: use tiktoken instead of the huggingface tokenizer

###### CONFIGURATION SECTION ######

model_names = [#"TheBloke/Wizard-Vicuna-7B-Uncensored-HF",
               "meta-llama/Llama-2-7b-hf",
               "tiiuae/falcon-7b",
               "mosaicml/mpt-7b-instruct",
               ]

sampling_params = SamplingParams(top_p=0.7, temperature=0.35, max_tokens=32)

###### DO NOT TOUCH HERE ######

t = sampling_params.temperature
top_p = sampling_params.top_p
max_new_tokens = sampling_params.max_tokens

###### CODE SECTION ######

def get_number_prompts(file):
  with open(file, 'r') as reader:
    all_data = ujson.load(reader)
    counter = 0
    for _ in all_data:
      counter += 1
    return counter

def prompt_generator(file):
  with open(file, 'r') as reader:
    all_data = ujson.load(reader)
    for data in all_data:
      yield data["cat"], data["slot"], data["value"], data["prompt"]
      
def get_only_prompts(file):
  with open(file, 'r') as reader:
    all_data = ujson.load(reader)
    output = []
    for data in all_data:
      output.append(data["prompt"])
    return output

def get_all_prompts(file):
  with open(file, 'r') as reader:
    all_data = ujson.load(reader)
    output = []
    for data in all_data:
      output.append((data["cat"], data["slot"], data["value"], data["prompt"]))
    return output

def get_simple_model_name(m_name):
  if '/' in m_name:
    m_name = m_name.split('/')[-1]
  return m_name

input_f = 'zero_shot_prompt_v1.json'
num_prompts = get_number_prompts(input_f)

# Purtroppo VLLM ha un bug ancora non fixato (https://github.com/vllm-project/vllm/issues/565) che non permette
# Di runnare 2 modelli in sequenza. Per questo motivo, per ogni modello, bisogna cancellare la memoria della GPU

for model_name in model_names:
  f_m_name = get_simple_model_name(model_name)
  file_output = f'{f_m_name}__t_{t}__top_p_{top_p}__max_new_tokens_{max_new_tokens}.jsonl'

  model = LLM(model_name, trust_remote_code=True, gpu_memory_utilization=0.93)
  
  with open(file_output, 'w') as writer:
    generator = prompt_generator(input_f)
    full_list_prompts = get_only_prompts(input_f)
    model_outputs = model.generate(full_list_prompts, sampling_params=sampling_params)
    destroy_model_parallel()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    outputs_dict = {}
    for output in model_outputs:
      prompt = output.prompt
      generated_text = output.outputs[0].text
      outputs_dict[prompt] = generated_text
    for (cat, slot, value, prompt) in tqdm(generator, total=num_prompts):
      model_output = outputs_dict[prompt]
      json_dump = ujson.dumps({"cat": cat,
                              "slot": slot,
                              "value": value,
                              "prompt": prompt,
                              "result": model_output})
      writer.write(json_dump + '\n')