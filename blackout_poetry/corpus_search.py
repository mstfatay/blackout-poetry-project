import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from .corpus import BlackoutPoetryCorpus, BaseCorpus
from json import dump

class CorpusSearch:
    output_window: int = 1
    model_name: str = "luodian/llama-7b-hf"
    best_tokens_count: int = 1000

    def __init__(self, corpus: BaseCorpus):
        self.model = LlamaForCausalLM.from_pretrained(self.model_name, return_dict_in_generate=True)
        self.tokenizer =AutoTokenizer.from_pretrained(self.model_name)
        self.corpus = corpus

        self.generated_words: list[str] = []

    def corpus_search(self, prompt: str, word_count: int, verbose: bool = True, output_file_path: str = None):
        self.prompt = prompt
        self.generated_words: list[str] = []

        data = []
        for _ in range(word_count):
            result = self._corpus_search_iteration()
            data.append(result)
            if verbose:
                print(result['next_word'])

            if result['next_word'] is None:
                break

        if output_file_path:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                dump({
                    "generated_output": self.generated_output,
                    "details": data,
                }, f, indent=2)
        
        return self.generated_output

    def _corpus_search_iteration(self):
        inputs = self.tokenizer(self.prompt, return_tensors="pt")
        generation_length = len(inputs.input_ids[0]) + self.output_window

        generation_output = self.model.generate(inputs.input_ids, max_length=generation_length, output_scores=True, num_beams=1, do_sample=False)

        # create an empty torch tensor to hold the tensor of selected token ids
        output_ids = torch.empty(0, dtype=torch.int64) #  dtype=torch.tensor([]).dtype)

        # generation_output['scores']->shape = (num_generated_tokens, num_generations, vocab_size)
        for i in range(len(generation_output['scores'])):
            logits: torch.tensor = generation_output['scores'][i][0]
            sorted_logits, sorted_inds = logits.sort(descending=True)

            # transpose it
            selected_token_ids = sorted_inds[0:self.best_tokens_count].unsqueeze(1)
            output_ids = torch.cat([output_ids, selected_token_ids], dim=1)

        outputs: list[str] = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        possible_next_words = outputs #[self._clean_word(output) for output in outputs]



        next_word = None
        for possible_next_word in possible_next_words:
            if self.corpus.find(possible_next_word) >= 0:
                self.corpus.retrieve(possible_next_word)
                next_word = possible_next_word
                break
        
        if next_word is None:
            return {
                'next_word': None,
                #'possible_next_words': possible_next_words,
                'outputs': outputs,
            }

        self.prompt = self.prompt.strip() + " " + next_word

        self.generated_words.append(next_word)

        return {
            'next_word': next_word, 
            #'possible_next_words': possible_next_words,
            'outputs': outputs,
        }
    
    def _clean_word(self, word: str) -> str:
        return word.strip().strip('.,?!`"\':;{}[]()“‘’')


    def set_corpus(self, corpus: BaseCorpus):
        self.corpus = corpus

    @property
    def generated_output(self):
        return " ".join(self.generated_words)
    

