class BaseCorpus:
    def __init__(self):
        pass

    def find(self, word: str) -> int:
        raise NotImplementedError

    def retrieve(self, word: str):
        raise NotImplementedError

    def retrieve_first_from_list(self, words: list[str]):
        raise NotImplementedError


class BlackoutPoetryWordCorpus(BaseCorpus):
    punctions = ".,?!`\"':;{}[]()“‘’"

    def __init__(self, text: str):
        self.text = text
        self.words = list(map(lambda x: x.lower(), text.split()))

    def find(self, word: str) -> int:
        word = self._clean_word(word)
        if word.lower() in self.words:
            return self.words.index(word.lower())
        else:
            return -1

    def retrieve(self, word: str) -> str:
        word = self._clean_word(word)
        index = self.find(word)
        if index >= 0:
            selected_word = self.words[index]
            self.words = self.words[index + 1 :]
            return selected_word
        else:
            return None

    def retrieve_first_from_list(self, words: list[str]) -> str:
        for word in words:
            selected_word = self.retrieve(word)
            if selected_word:
                return selected_word
        return None

    def _clean_word(self, word: str) -> str:
        if word in self.punctions:
            return word
        return word.strip().strip(".,?!`\"':;{}[]()“‘’")
