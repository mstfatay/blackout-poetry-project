from .base_corpus import BaseCorpus
from transformers import AutoTokenizer


class BlackoutPoetryTokenCorpus(BaseCorpus):
    punctions = ".,?!`\"':;{}[]()“‘’"
    end_punctions = ".,?!`\"':;}])’"
    start_punctions = "`\"'{[(“‘"
    model_name: str = "luodian/llama-7b-hf"

    def __init__(self, text: str, model_name: str = None):
        if model_name:
            self.model_name = model_name
        self.text = text

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        tmp_text = text.lower()
        for punction in self.punctions:
            tmp_text = tmp_text.replace(punction, " ")
        cleaned_text = tmp_text.strip()

        self.words = [word for word in cleaned_text.split() if word]

        # We preserve the order of words
        self.word_combinations: list[list[str]] = []
        for word in self.words:
            self.word_combinations.append(self._get_combinations_of_word(word))

        self.tokens_of_combinations: list[list[list[int]]] = []
        for word_comb in self.word_combinations:
            curr_tokens = self._tokenize_words(word_comb)
            self.tokens_of_combinations.append(curr_tokens)

    def _get_combinations_of_word(self, word: str) -> list[str]:
        # There are currently a total of 88 combinations of each word
        word_itself = word
        upper_cased = word.capitalize()

        fund_combs = [word_itself, upper_cased]

        combs_with_punc = [*fund_combs]
        for punction in self.end_punctions:
            for comb in fund_combs:
                with_punction = f"{comb}{punction}"
                combs_with_punc.append(with_punction)
        for punction in self.start_punctions:
            for comb in fund_combs:
                with_punction = f"{punction}{comb}"
                combs_with_punc.append(with_punction)

        return combs_with_punc

        all_combs = [*combs_with_punc]
        for word in combs_with_punc:
            all_combs.append(f" {word}")

        return all_combs

    def _tokenize_words(self, words: list[str]) -> list[int]:
        # token_strings = self.tokenizer.convert_ids_to_tokens(token_ids)
        # space == '▁'
        tokenization_result = self.tokenizer(
            words, return_tensors="pt", add_special_tokens=False, padding=True
        )
        input_ids = tokenization_result["input_ids"].tolist()
        attention_mask = tokenization_result["attention_mask"].tolist()

        result: list[list[int]] = []
        for i in range(len(input_ids)):
            curr_input_ids = input_ids[i]
            curr_attention_mask = attention_mask[i]
            curr_result = []
            for j in range(len(curr_input_ids)):
                if curr_attention_mask[j] == 1:
                    curr_result.append(curr_input_ids[j])
            result.append(curr_result)
        return result

    def find(self, token: int) -> int:
        if token in self.tokens:
            return self.tokens.index(token)
        else:
            return -1

    def retrieve(self, token: int) -> int:
        index = self.find(token)
        if index >= 0:
            selected_token = self.tokens[index]
            self.tokens = self.tokens[index + 1 :]
            return selected_token
        else:
            return None

    def retrieve_first_from_list(self, tokens: list[int]) -> int:
        for token in tokens:
            selected_token = self.retrieve(token)
            if selected_token:
                return selected_token
        return None


if __name__ == "__main__":
    text = "Solomon Eliot Asch (September 14, 1907 \u2013 February 20, 1996) was a Polish and American Gestalt psychologist.\n\nEarly life\nAsch was born in Warsaw, the capital of Poland, to a Jewish family that considered themselves Polish. He grew up in a small town called \u0141owicz. In 1920, Asch and his family moved to the United States. They lived on the Lower East Side of New York. He did not speak English well, so he found school difficult. He learned English by reading Charles Dickens. Asch got his degree from the College of the City of New York in 1928. In 1930, he married Florence Miller.\n\nCareer\nIn the 1930s, Asch started to do research. In 1951, Asch did an experiment that is known as the Asch Experiment. Asch's experiment tested how much humans are influenced by other people's opinions. A textbook that Asch wrote, Social Psychology, was published in 1952. In 1966 Asch started the Institute for Cognitive Studies at Rutgers University. In 1972 he moved to the University of Pennsylvania. He stayed there as a professor of psychology until he retired in 1979. He started all of this research because he thought that there was a problem with the Sherif's conformity test.\n\nReferences\n\n1907 births\n1996 deaths\nAmerican psychologists\nAmerican Jews\nPolish psychologists\nPolish Jews"
    corpus = BlackoutPoetryTokenCorpus(text)

    breakpoint()
