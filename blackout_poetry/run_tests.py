from blackout_poetry.corpus import BlackoutPoetryCorpus
from blackout_poetry.corpus_search import CorpusSearch
from blackout_poetry.data import get_wikipedia_dataset
from json import dump
import os

tests1 = [
    {
        "name": "wikipedia_treshold_001",
        "treshold_prob": 0.001,
        "prompt": "Here is a poem: \n\n",
    },
    {
        "name": "wikipedia_treshold_0003",
        "treshold_prob": 0.0003,
        "prompt": "Here is a poem: \n\n",
    },
    {
        "name": "wikipedia_treshold_0001",
        "treshold_prob": 0.0001,
        "prompt": "Here is a poem: \n\n",
    },
]

tests2 = [
    {
        "name": "wikipedia_prompt_1",
        "treshold_prob": 0.001,
        "prompt": "Here is a poem: \n\n",
    },
    {
        "name": "wikipedia_prompt_2",
        "treshold_prob": 0.001,
        "prompt": "Here is a poem: \n",
    },
    {
        "name": "wikipedia_prompt_3",
        "treshold_prob": 0.001,
        "prompt": "Here is a random sentence: \n\n",
    },
    {
        "name": "wikipedia_prompt_4",
        "treshold_prob": 0.001,
        "prompt": "Here is a random sentence: \n",
    },
]

tests3 = [
    {
        "name": "wikipedia_llama_13b",
        "treshold_prob": 0.001,
        "prompt": "Here is a poem: \n\n",
        "model_name": "luodian/llama-13b-hf",
    },
    {
        "name": "wikipedia_llama_7b",
        "treshold_prob": 0.001,
        "prompt": "Here is a poem: \n\n",
        "model_name": "luodian/llama-7b-hf",
    },
]

if __name__ == "__main__":
    test_name = "wikipedia1"

    dataset = get_wikipedia_dataset()

    for test in tests3:
        test_name = test["name"]
        prompt = test["prompt"]
        model_name = test["model_name"]
        # create directory
        os.makedirs(f"evals/{test_name}/outputs", exist_ok=True)
        with open(f"evals/{test_name}/data.json", "w", encoding="utf-8") as f:
            dump(list(dataset), f, indent=2)

        corpus_search = CorpusSearch(
            None,
            model_name=model_name,
        )
        results = []

        for data_point in dataset:
            corpus_text = data_point["text"]
            title = data_point["title"]

            corpus = BlackoutPoetryCorpus(corpus_text)
            corpus_search.set_corpus(corpus)
            corpus_search.treshold_prob = test["treshold_prob"]
            result = corpus_search.corpus_search(
                prompt, 50, output_file_path=f"evals/{test_name}/outputs/{title}.json"
            )
            results.append(result)

        with open(f"evals/{test_name}/results.json", "w", encoding="utf-8") as f:
            dump(results, f, indent=2)
