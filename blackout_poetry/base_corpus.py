class BaseCorpus:
    def __init__(self):
        pass

    def find(self, word: str) -> int:
        raise NotImplementedError

    def retrieve(self, word: str):
        raise NotImplementedError

    def retrieve_first_from_list(self, words: list[str]):
        raise NotImplementedError
