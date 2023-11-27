import json
import transformers
from accelerate import infer_auto_device_map, init_empty_weights, PartialState
import torch
from tqdm import tqdm

model_names = ["01-ai/Yi-34B"]


def load_llm_model(model_name):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    config = transformers.AutoConfig.from_pretrained(
        model_name, trust_remote_code=True, init_device="meta"
    )

    with init_empty_weights():
        model = transformers.AutoModelForCausalLM.from_config(
            config, trust_remote_code=True
        )
        model.tie_weights()

    device_map = infer_auto_device_map(
        model, no_split_module_classes=["OPTDecoderLayer"], dtype="bfloat16"
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device_map,
    )

    model.to("cuda")

    return tokenizer, model


t = 0.35
top_p = 0.9
max_new_tokens = 50


def generate_from_model(model, tokenizer, text):
    encoded_input = tokenizer(
        text, truncation=True, return_tensors="pt"
    )
    output_sequences = model.generate(
        input_ids=encoded_input["input_ids"].to("cuda"),
        temperature=t,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    decoded_seq = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    decoded_seq = decoded_seq.replace(text, "")
    return decoded_seq


def get_number_prompts(file):
    with open(file, "r") as reader:
        all_data = json.load(reader)
        counter = 0
        for _ in all_data:
            counter += 1
        return counter


def prompt_generator(file):
    with open(file, "r") as reader:
        all_data = json.load(reader)
        for data in all_data:
            yield data["cat"], data["slot"], data["value"], data["prompt"]


def get_all_prompts(file):
    with open(file, "r") as reader:
        all_data = json.load(reader)
        output = []
        for data in all_data:
            output.append((data["cat"], data["slot"], data["value"], data["prompt"]))
        return output


def get_simple_model_name(m_name):
    if "/" in m_name:
        m_name = m_name.split("/")[-1]

    return m_name


input_f = "zero_shot_prompt_v1.json"
num_prompts = get_number_prompts(input_f)

for model_name in model_names:
    f_m_name = get_simple_model_name(model_name)
    file_output = (
        f"{f_m_name}__t_{t}__top_p_{top_p}__max_new_tokens_{max_new_tokens}.jsonl"
    )

    tokenizer, model = load_llm_model(model_name)

    with open(file_output, "w") as writer:
        generator = prompt_generator(input_f)
        for cat, slot, value, prompt in tqdm(generator, total=num_prompts):
            model_output = generate_from_model(model, tokenizer, prompt)
            json_dump = json.dumps(
                {
                    "cat": cat,
                    "slot": slot,
                    "value": value,
                    "prompt": prompt,
                    "result": model_output,
                }
            )
            writer.write(json_dump + "\n")

    del tokenizer
    del model
    torch.cuda.empty_cache()
