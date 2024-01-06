from .base_corpus import BaseCorpus


class BlackoutPoetryTokenCorpus(BaseCorpus):
    punctions = ".,?!`\"':;{}[]()“‘’"
    end_punctions = ".,?!`\"':;}])’"
    start_punctions = "`\"'{[(“‘"

    def __init__(self, text: str):
        self.text = text

        tmp_text = text.lower()
        for punction in self.punctions:
            tmp_text = tmp_text.replace(punction, " ")
        cleaned_text = tmp_text.strip()

        self.words = [word for word in cleaned_text.split() if word]

        # Good thing is the order of words is preserved
        self.word_combinations = []
        for word in self.words:
            self.word_combinations.extend(self._get_combinations_of_word(word))

        # The order of tokens is preserved
        self.tokens = self._tokenize_words(self.word_combinations)

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

        all_combs = [*combs_with_punc]
        for word in combs_with_punc:
            all_combs.append(f" {word}")

        return all_combs

    def _tokenize_words(self, words: list[str]) -> list[int]:
        return []

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
