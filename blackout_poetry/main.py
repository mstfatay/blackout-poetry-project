import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM
)

device = "cpu"


model_name = "luodian/llama-7b-hf"

#model = AutoModelForCausalLM.from_pretrained(
#    model_name, cache_dir=None, torch_dtype=torch.float32
#)
model = LlamaForCausalLM.from_pretrained(model_name, return_dict_in_generate=True)

#tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=None, use_fast=False)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Hello, how are you?"

#inputs = tokenizer(prompt, padding=False, truncation=False)

inputs = tokenizer(prompt, return_tensors="pt")


#generate_ids = model.generate(
#    inputs.input_ids,
#    attention_mask=inputs.attention_mask,
#    num_beams=1,
#    do_sample=False,
#)

output_window = 5
generation_length = len(inputs.input_ids[0]) + output_window

generation_output = model.generate(inputs.input_ids, max_length=generation_length, output_scores=True)

# create an empty torch tensor to hold the selected token ids
output_ids = torch.empty(0, dtype=torch.int64)

# generation_output['scores']->shape = (num_generated_tokens, num_generations, vocab_size)
for i in range(len(generation_output['scores'])):
    logits: torch.tensor = generation_output['scores'][i][0]
    sorted_logits, sorted_inds = logits.sort(descending=True)

    selected_token_id = sorted_inds[0]
    output_ids = torch.cat([output_ids, selected_token_id.unsqueeze(0)])


# output_ids: add selected_token_id to the end of the input_ids

output: str = tokenizer.batch_decode(output_ids.unsqueeze(0), skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


next_word = output.strip().split(maxsplit = 1)[0]
print(output)
print(next_word)

breakpoint()