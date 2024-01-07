import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from .token_corpus import BlackoutPoetryTokenCorpus
from .base_corpus import BaseCorpus
from json import dump


class TokenCorpusSearch:
    output_window: int = 1
    model_name: str = "luodian/llama-7b-hf"
    best_tokens_count: int = 32000
    treshold_prob: float = 0.0001
    max_output_length: int = 10

    def __init__(self, corpus: BlackoutPoetryTokenCorpus, model_name: str = None):
        if model_name:
            self.model_name = model_name
        self.model = LlamaForCausalLM.from_pretrained(
            self.model_name, return_dict_in_generate=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.corpus = corpus

    def _generate_next_best_tokens(self, input_token_ids: torch.tensor) -> list[int]:
        unsequezed_input_token_ids = input_token_ids.unsqueeze(0)
        generation_length = len(input_token_ids) + 1
        generation_output = self.model.generate(
            unsequezed_input_token_ids,
            max_length=generation_length,
            output_scores=True,
            num_beams=1,
            do_sample=False,
        )

        # only one token is generated
        logits: torch.tensor = generation_output["scores"][0][0]
        probs = torch.nn.Softmax(dim=0)(logits)
        sorted_probs, sorted_inds = probs.sort(descending=True)

        selected_token_ids = sorted_inds[0 : self.best_tokens_count]
        selected_probs = sorted_probs[0 : self.best_tokens_count]

        return {
            "best_next_tokens": selected_token_ids,
            "best_next_tokens_probs": selected_probs,
        }

    def corpus_search(
        self,
        prompt: str,
        max_iterations: int = 50,
        verbose: bool = True,
        output_file_path: str = None,
    ):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        return self.corpus_search_with_tokens(
            inputs.input_ids[0], max_iterations, verbose
        )

    def corpus_search_with_tokens(
        self, input_ids: torch.tensor, max_iterations, verbose
    ):
        stack = []
        texts_trace = []
        for i in range(max_iterations):
            stack = self._corpus_search_one_iteration(input_ids, stack)
            stack_text = self.get_stack_text(stack)
            texts_trace.append(stack_text)

            if verbose:
                print(f"iteration {i}:")
                print(stack_text)

            # stack_token_ids = self._get_stack_token_ids(stack)
            # new_ids = torch.concat([input_ids, stack_token_ids], dim=0)
            # curr_text = self.tokenizer.decode(new_ids.tolist())

        if verbose:
            print(texts_trace)

        generations = []
        for text in self.texts_trace:
            for i in range(len(text), 0, -1):
                if text[i] in [".", "?", "!"]:
                    generation = text[: i + 1].strip()
                    generations.append(generation)
                    break

        # remove duplicates
        generations = list(set(generations))
        if verbose:
            print(generations)
        return generations

    def get_stack_text(self, stack) -> str:
        if len(stack) == 0:
            return "[empty stack]"
        stack_token_ids = self._get_stack_token_ids(stack)
        curr_text = self.tokenizer.decode(stack_token_ids.tolist())
        return curr_text

    def _get_stack_token_ids(self, stack):
        token_ids = []
        for stack_obj_list in stack:
            for stack_obj in stack_obj_list:
                token_ind = stack_obj["token_ind"]
                token_ids.append(stack_obj["best_next_tokens"][token_ind])
        return torch.tensor(token_ids, dtype=torch.int64)

    def _get_last_corpus_ind_from_stack(self, stack):
        if len(stack) == 0:
            return -1
        return stack[-1][-1]["corpus_ind"]

    def is_viable_node(self, stack_obj, stack):
        token_ind = stack_obj["token_ind"]
        corpus_ind = stack_obj["corpus_ind"]

        if len(stack) >= self.max_output_length:
            return False

        if token_ind == -1 or corpus_ind == -1:
            return False
        if stack_obj["best_next_tokens_probs"][token_ind] < self.treshold_prob:
            return False

        if len(stack) != 0 and stack_obj["is_new_word_list"][token_ind]:
            word_seq = self._create_one_token_seq(stack)
            previous_word_corpus_ind = stack[-1][-1]["corpus_ind"]

            if stack_obj["corpus_ind"] == previous_word_corpus_ind:
                return False

            # Previous word should be complete
            find_exact_ind = self.corpus.find_exact(word_seq, previous_word_corpus_ind)

            if find_exact_ind != previous_word_corpus_ind:
                return False

        return True

    def create_is_new_word_list(self, stack_obj, stack) -> list[bool]:
        if len(stack) == 0:
            return [True] * len(stack_obj["best_next_tokens"])

        tokens = stack_obj["best_next_tokens"]

        token_strings = self.tokenizer.convert_ids_to_tokens(tokens.tolist())

        space = "▁"
        return [token_string[0] == space for token_string in token_strings]

    def _back_track(self, stack):
        if len(stack) == 0:
            raise Exception("Cannot backtrack from empty stack")
        back_tracked_node = stack[-1].pop()

        # remove empty lists from the end of the stack
        if len(stack[-1]) == 0:
            stack.pop()

        stack = self._corpus_search_add_new_node(stack, back_tracked_node)
        return stack

    def _corpus_search_add_new_node(self, stack, stack_obj):
        corpus_ind = self._get_last_corpus_ind_from_stack(stack)

        is_new_word_list = self.create_is_new_word_list(stack_obj, stack)
        stack_obj["is_new_word_list"] = is_new_word_list

        from_ind = stack_obj.get("token_ind", -1) + 1
        token_seqs = self._create_token_seqs(stack, stack_obj)
        token_ind, new_corpus_ind = self.corpus.find_first_in_list(
            token_seqs, corpus_ind, from_ind
        )

        stack_obj["token_ind"] = token_ind
        stack_obj["corpus_ind"] = new_corpus_ind

        is_viable_new_node = self.is_viable_node(stack_obj, stack)

        if is_viable_new_node:
            is_curr_new_word = is_new_word_list[token_ind]

            if is_curr_new_word:
                stack.append([stack_obj])
            else:
                stack[-1].append(stack_obj)
        else:
            stack = self._back_track(stack)
        return stack

    def _corpus_search_one_iteration(self, input_ids: torch.tensor, stack):
        stack_token_ids = self._get_stack_token_ids(stack)
        new_ids = torch.concat([input_ids, stack_token_ids], dim=0)

        best_next_tokens_obj = self._generate_next_best_tokens(new_ids)
        best_next_tokens = best_next_tokens_obj["best_next_tokens"]
        best_next_tokens_probs = best_next_tokens_obj["best_next_tokens_probs"]

        stack_obj = {
            "best_next_tokens": best_next_tokens,
            "best_next_tokens_probs": best_next_tokens_probs,
        }

        stack = self._corpus_search_add_new_node(stack, stack_obj)
        return stack

    def _create_one_token_seq(self, stack) -> list[int]:
        if len(stack) == 0:
            return []

        last_word_objs = stack[-1]
        last_word_token_seq = []
        for last_word_obj in last_word_objs:
            token_ind = last_word_obj["token_ind"]
            last_word_token_seq.append(
                int(last_word_obj["best_next_tokens"][token_ind])
            )
        return last_word_token_seq

    def _create_token_seqs(self, stack, stack_obj) -> list[list[int]]:
        best_next_tokens = stack_obj["best_next_tokens"]

        terminal_tokens_list = best_next_tokens.tolist()

        if len(stack) == 0:
            return [[token] for token in terminal_tokens_list]

        last_word_objs = stack[-1]
        last_word_token_seq = []
        for last_word_obj in last_word_objs:
            token_ind = last_word_obj["token_ind"]
            last_word_token_seq.append(
                int(last_word_obj["best_next_tokens"][token_ind])
            )

        token_seqs = []
        for i in range(len(terminal_tokens_list)):
            curr_best_next_token = terminal_tokens_list[i]
            if stack_obj["is_new_word_list"][i]:
                token_seqs.append([curr_best_next_token])
            else:
                token_seqs.append(last_word_token_seq + [curr_best_next_token])
        return token_seqs
