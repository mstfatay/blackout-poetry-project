from blackout_poetry.token_corpus import BlackoutPoetryTokenCorpus
from blackout_poetry.token_corpus_search import TokenCorpusSearch
from blackout_poetry.data import get_wikipedia_dataset
from json import dump
import os

tests1 = [
    {
        "name": "wikipedia_treshold_01",
        "treshold_prob": 0.01,
        "prompt": "Here is a poem:",
    },
    {
        "name": "wikipedia_treshold_003",
        "treshold_prob": 0.003,
        "prompt": "Here is a poem:",
    },
    {
        "name": "wikipedia_treshold_001",
        "treshold_prob": 0.001,
        "prompt": "Here is a poem:",
    },
    {
        "name": "wikipedia_treshold_0003",
        "treshold_prob": 0.0003,
        "prompt": "Here is a poem:",
    },
    {
        "name": "wikipedia_treshold_0001",
        "treshold_prob": 0.0001,
        "prompt": "Here is a poem:",
    },
]

tests2 = [
    {
        "name": "wikipedia_prompt_1",
        "treshold_prob": 0.001,
        "prompt": "Here is a poem:",
    },
    {
        "name": "wikipedia_prompt_2",
        "treshold_prob": 0.001,
        "prompt": "Here is a random sentence:",
    },
    {
        "name": "wikipedia_prompt_3",
        "treshold_prob": 0.001,
        "prompt": "Here is a sentence:",
    },
]

tests3 = [
    {
        "name": "wikipedia_llama_7b",
        "treshold_prob": 0.001,
        "prompt": "Here is a poem:",
        "model_name": "luodian/llama-7b-hf",
    },
    {
        "name": "wikipedia_llama_13b",
        "treshold_prob": 0.001,
        "prompt": "Here is a poem:",
        "model_name": "luodian/llama-13b-hf",
    },
]


tests4 = [
    {
        "name": "wikipedia_llama_13b_prompt2_treshold_003",
        "treshold_prob": 0.003,
        "prompt": "Here is a random sentence:",
        "model_name": "luodian/llama-13b-hf",
    },
]

if __name__ == "__main__":
    dataset = get_wikipedia_dataset()

    corpus_search = TokenCorpusSearch(
        None,
    )

    for test in tests4:
        test_name = test["name"]
        prompt = test["prompt"]
        # create directory
        os.makedirs(f"evals_token/{test_name}", exist_ok=True)
        with open(f"evals_token/{test_name}/data.json", "w", encoding="utf-8") as f:
            dump(list(dataset), f, indent=2)

        if test.get("model_name", None):
            if corpus_search.model_name != test["model_name"]:
                print("Changing model to", test["model_name"])
                corpus_search.change_model(test["model_name"])

        results = []

        for i, data_point in enumerate(dataset):
            print("---------------------------------------")
            print(f"Running {i+1}/{len(dataset)}")
            corpus_text = data_point["text"]
            corpus_text = " ".join(corpus_text.split())
            title = data_point["title"]

            corpus = BlackoutPoetryTokenCorpus(
                corpus_text, test.get("model_name", None)
            )
            corpus_search.set_corpus(corpus)
            corpus_search.treshold_prob = test["treshold_prob"]
            result = corpus_search.corpus_search(
                prompt,
                50,
                # output_file_path=f"evals_token/{test_name}/outputs/{title}.json",
            )
            results.append({"title": title, "generations": result})

        with open(f"evals_token/{test_name}/results.json", "w", encoding="utf-8") as f:
            dump(results, f, indent=2)
