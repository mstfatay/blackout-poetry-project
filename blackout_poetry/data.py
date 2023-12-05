from datasets import load_dataset, Audio


def get_wikipedia_dataset():
    dataset = load_dataset("wikipedia", "20220301.simple", split="train")

    # shuffle the dataset
    dataset = dataset.shuffle(seed=42)

    # select a subset of the dataset
    dataset = dataset.select(range(10))

    return dataset


dataset = get_wikipedia_dataset()
