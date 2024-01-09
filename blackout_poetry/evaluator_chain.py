from langchain.prompts import PromptTemplate
from langchain.chat_models.openai import ChatOpenAI
from langchain.schema import StrOutputParser, OutputParserException
from dotenv import load_dotenv

load_dotenv()


class EvaluatorOutputParser(StrOutputParser):
    def parse(self, text: str) -> str:
        if "Reasoning:" not in text or "Answer:" not in text:
            raise OutputParserException("Nonparsable output")

        tmp_text0 = text.split("Reasoning:", 1)[1].strip()

        tmp_texts1 = tmp_text0.split("Answer:", 1)
        reasoning_text = tmp_texts1[0].strip()
        answer_text = tmp_texts1[1].strip().lower()

        if answer_text not in ["human", "machine", "unsure"]:
            raise OutputParserException("Invalid label")

        return {
            "reasoning": reasoning_text,
            "answer": answer_text,
        }


PROMPT = """You are a human in a Turing test. Some blackout poetries are shown to you. Humans write half of the blackout poetries and a machine creates the other half. Here is your next blackout poetry to state if a human or a machine writes the blackout poetry.

Original Poem:
{original_poem}

Blackout Poetry:
{blackout_poetry}

Answer in the following format:
Reasoning: step by step reasoning here
Answer: [human, machine, or unsure]
"""

prompt_template = PromptTemplate.from_template(PROMPT)
prompt = prompt_template.format(
    original_poem="I am a human",
    blackout_poetry="I am a human",
)

chat = ChatOpenAI(
    model_name="gpt-3.5-turbo", temperature=0, max_retries=10, request_timeout=15.0
)


evaluator_chain = prompt_template | chat | EvaluatorOutputParser()


if __name__ == "__main__":
    input = {
        "original_poem": "I am a human",
        "blackout_poetry": "I am a human",
    }

    result = evaluator_chain.invoke(input)
    print(result)
