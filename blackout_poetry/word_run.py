from blackout_poetry.word_corpus import BlackoutPoetryWordCorpus
from blackout_poetry.word_corpus_search import WordCorpusSearch

if __name__ == "__main__":
    prompt = "Here is a poem: \n\n"

    with open("blackout_poetry/corpus/1.txt", "r", encoding="utf-8") as f:
        corpus_text = f.read()

    corpus = BlackoutPoetryWordCorpus(corpus_text)
    corpus_search = WordCorpusSearch(corpus)
    result = corpus_search.corpus_search(
        prompt, 50, output_file_path="blackout_poetry/output/run.json"
    )
