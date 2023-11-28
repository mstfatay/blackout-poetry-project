import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from .corpus import BlackoutPoetryCorpus, BaseCorpus

class CorpusSearch:
    output_window: int = 5
    model_name: str = "luodian/llama-7b-hf"

    def __init__(self, corpus: BaseCorpus, prompt: str):
        self.prompt = prompt
        self.model = LlamaForCausalLM.from_pretrained(self.model_name, return_dict_in_generate=True)
        self.tokenizer =AutoTokenizer.from_pretrained(self.model_name)
        self.corpus = corpus

        self.generated_words: list[str] = []

    def corpus_search(self, prompt: str, word_count: int, verbose: bool = True):
        for _ in range(word_count):
            result = self.corpus_search_iteration()
            if verbose:
                print(result)

        return self.generated_output

    def corpus_search_iteration(self):
        inputs = self.tokenizer(self.prompt, return_tensors="pt")
        generation_length = len(inputs.input_ids[0]) + self.output_window

        generation_output = self.model.generate(inputs.input_ids, max_length=generation_length, output_scores=True)

        # create an empty torch tensor to hold the selected token ids
        output_ids = torch.empty(0, dtype=torch.int64)

        # generation_output['scores']->shape = (num_generated_tokens, num_generations, vocab_size)
        for i in range(len(generation_output['scores'])):
            logits: torch.tensor = generation_output['scores'][i][0]
            sorted_logits, sorted_inds = logits.sort(descending=True)

            selected_token_id = sorted_inds[0]
            output_ids = torch.cat([output_ids, selected_token_id.unsqueeze(0)])

        output: str = self.tokenizer.batch_decode(output_ids.unsqueeze(0), skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        next_word = output.strip().split(maxsplit = 1)[0]

        self.prompt = self.prompt + " " + next_word

        self.generated_words.append(next_word)

        return {'next_word': next_word, 'generated_sequence': output_ids, 'generated_output': output}
    
    @property
    def generated_output(self):
        return " ".join(self.generated_words)
    

if __name__ == "__main__":
    prompt = "Hello, how are you?"

    corpus = BlackoutPoetryCorpus()
    result = CorpusSearch(prompt).corpus_search(prompt, 10)
    print(result)