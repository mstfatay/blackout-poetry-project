from blackout_poetry.token_corpus import BlackoutPoetryTokenCorpus
from blackout_poetry.token_corpus_search import TokenCorpusSearch

if __name__ == "__main__":
    prompt = "Here is a poem:"

    with open("blackout_poetry/corpus/1.txt", "r", encoding="utf-8") as f:
        corpus_text = f.read()

    corpus_text = " ".join(corpus_text.split())

    corpus = BlackoutPoetryTokenCorpus(corpus_text)
    corpus_search = TokenCorpusSearch(corpus)
    result = corpus_search.corpus_search(
        prompt, 50, output_file_path="blackout_poetry/output/run.json"
    )
