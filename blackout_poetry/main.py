import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    OPTForCausalLM,
)

device = "cpu"


model_name = "luodian/llama-7b-hf"

model = AutoModelForCausalLM.from_pretrained(
    model_name, cache_dir=None, torch_dtype=torch.float32
)

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=None, use_fast=False)


prompt = "Hello, how are you?"

inputs = tokenizer(prompt, padding=False, truncation=False)


breakpoint()

most_likely_generations = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask.to(device),
    num_beams=1,
    do_sample=False,
).cpu()

breakpoint()
