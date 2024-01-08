from pathlib import Path
from json import load, dump
from blackout_poetry.evaluator_chain import evaluator_chain
from asyncio import run, gather


def get_experiment_names():
    path = Path("evals_token")

    experiment_names = []
    for file in path.iterdir():
        if file.is_dir():
            experiment_names.append(file.parts[-1])
    return experiment_names


def get_experiment_data_and_result(experiment_name):
    data_path = f"evals_token/{experiment_name}/data.json"
    results_path = f"evals_token/{experiment_name}/results.json"

    with open(data_path, "r", encoding="utf-8") as f:
        data = load(f)
    with open(results_path, "r", encoding="utf-8") as f:
        results = load(f)
    return data, results


def find_data_point_by_title(data, title):
    for data_point in data:
        if data_point["title"] == title:
            return data_point
    return None


async def run_tasks(tasks):
    return await gather(*tasks)


def run_evaluations():
    experiment_names = get_experiment_names()

    for experiment_name in experiment_names:
        print("Running:", experiment_name)
        data, results = get_experiment_data_and_result(experiment_name)

        task_inputs = []
        tasks = []
        for result in results:
            title = result["title"]
            generations = result["generations"]

            data_point = find_data_point_by_title(data, title)
            text = data_point["text"]

            if len(generations) > 0:
                selected_generation = generations[0]
                for generation in generations:
                    if len(generation) > len(selected_generation):
                        selected_generation = generation
            else:
                selected_generation = None

            if selected_generation:
                task_inputs.append(
                    {"title": title, "text": text, "generation": selected_generation}
                )
                task = evaluator_chain.ainvoke(
                    {
                        "original_poem": text,
                        "blackout_poetry": selected_generation,
                    }
                )
                tasks.append(task)

        results = run(run_tasks(tasks))

        for i, result in enumerate(results):
            results[i] = {
                "title": task_inputs[i]["title"],
                "generation": task_inputs[i]["generation"],
                **result,
            }

        with open(
            f"evals_token/{experiment_name}/evaluations.json", "w", encoding="utf-8"
        ) as f:
            dump(results, f, indent=2)

        point_count = len(results)
        human_count = len([result for result in results if result["answer"] == "human"])
        machine_count = len(
            [result for result in results if result["answer"] == "machine"]
        )
        unsure_count = len(
            [result for result in results if result["answer"] == "unsure"]
        )

        statistics = {
            "point_count": point_count,
            "human_count": human_count,
            "machine_count": machine_count,
            "unsure_count": unsure_count,
            "human_ratio": human_count / point_count,
            "machine_ratio": machine_count / point_count,
            "unsure_ratio": unsure_count / point_count,
        }

        with open(
            f"evals_token/{experiment_name}/statistics.json", "w", encoding="utf-8"
        ) as f:
            dump(statistics, f, indent=2)


if __name__ == "__main__":
    run_evaluations()
