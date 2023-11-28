


class BaseCorpus:
    def __init__(self):
        pass

    def find(self, word: str) -> int:
        raise NotImplementedError

    def retrieve(self, word: str):
        raise NotImplementedError

    def retrieve_first_from_list(self, words: list[str]):
        raise NotImplementedError
    

class BlackoutPoetryCorpus(BaseCorpus):
    def __init__(self, text: str):
        self.text = text
        self.words = text.split()

    def find(self, word: str) -> int:
        if word in self.words:
            return self.words.index(word)
        else:
            return -1

    def retrieve(self, word: str) -> str:
        index = self.find(word)
        if index >= 0:
            selected_word = self.words[index]
            self.words = self.words[index+1:]
            return selected_word
        else:
            return None
        
    def retrieve_first_from_list(self, words: list[str]) -> str:
        for word in words:
            selected_word = self.retrieve(word)
            if selected_word:
                return selected_word
        return None