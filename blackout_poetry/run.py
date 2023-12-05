from blackout_poetry.corpus import BlackoutPoetryCorpus
from blackout_poetry.corpus_search import CorpusSearch

if __name__ == "__main__":
    prompt = "Here is a poem: \n\n"

    with open(f"corpus/1.txt", "r", encoding="utf-8") as f:
        corpus_text = f.read()

    corpus = BlackoutPoetryCorpus(corpus_text)
    corpus_search = CorpusSearch(corpus)
    result = corpus_search.corpus_search(
        prompt, 50, output_file_path=f"output/run.json"
    )
