# Greedy Restricted Corpus Search

## Setup

`pip install -r requirements.txt`

## Run Demo

`python -m blackout_poetry.demo`

## Test Results

Test results are documented under `evals_token` folder.

## Regenerate Test Results

`python -m blackout_poetry.token_run_tests`
`python -m blackout_poetry.test_evaluator`

Note: in order to run evaluator script, you need to create `blackout_poetry/.env` file from `blackout_poetry/.env.sample` and fill the `OPENAI_API_KEY` environment variable.
