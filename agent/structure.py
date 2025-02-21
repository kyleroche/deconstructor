import argparse
import os
import json
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from griptape.configs import Defaults
from griptape.configs.drivers import DriversConfig
from griptape.structures import Agent
from griptape.rules import Rule
from griptape.drivers import GriptapeCloudEventListenerDriver
from griptape.events import EventBus, EventListener


class WordPart(BaseModel):
    id: str = Field(description="Lowercase identifier, unique across parts and combinations")
    text: str = Field(description="Exact section of input word")
    originalWord: str = Field(description="Oldest word/affix this part comes from")
    origin: str = Field(description="Brief origin (Latin, Greek, etc)")
    meaning: str = Field(description="Concise meaning of this part")


class Combination(BaseModel):
    id: str = Field(description="Unique lowercase identifier")
    text: str = Field(description="Combined text segments")
    definition: str = Field(description="Clear definition of combined parts")
    sourceIds: List[str] = Field(description="Array of part/combination ids used")


class WordOutput(BaseModel):
    thought: str = Field(description="Think about the word/phrase, it's origins, and how it's put together")
    parts: List[WordPart] = Field(description="Array of word parts that combine to form the word")
    combinations: List[List[Combination]] = Field(description="Layers of combinations forming a DAG to the final word")


def is_running_in_managed_environment() -> bool:
    return "GT_CLOUD_STRUCTURE_RUN_ID" in os.environ


def get_listener_api_key() -> str:
    api_key = os.environ.get("GT_CLOUD_API_KEY", "")
    if is_running_in_managed_environment() and not api_key:
        pass
    return api_key


def setup_config():
    if is_running_in_managed_environment():
        event_driver = GriptapeCloudEventListenerDriver(api_key=get_listener_api_key())
        EventBus.add_event_listener(EventListener(event_listener_driver=event_driver))
    else:
        load_dotenv('../.env.local')


def create_word_agent() -> Agent:
    return Agent(
        rules=[
            Rule("You are a linguistic expert that deconstructs words into their meaningful parts and explains their etymology."),
            Rule("You must ONLY analyze the exact input word provided, never substitute it with a different word."),
            Rule("Create multiple layers of combinations to form the final meaning of the word."),
            Rule("The final combination must be the complete input word."),
            Rule("All IDs must be unique across both parts and combinations."),
            Rule("The parts must combine exactly to form the input word."),
            Rule("Respond with a JSON object that matches this schema exactly:"),
            Rule(json.dumps(WordOutput.model_json_schema(), indent=2))
        ]
    )


def deconstruct_word(agent: Agent, word: str, previous_attempts: list = None) -> dict:
    prompt = f"""Your task is to deconstruct this EXACT word: '{word}'
Do not analyze any other word. Focus only on '{word}'.
Break down '{word}' into its etymological components."""

    if previous_attempts:
        prompt += f"\n\nPrevious attempts:\n{json.dumps(previous_attempts, indent=2)}\n\nPlease fix all the issues and try again."

    response = agent.run(prompt)
    try:
        response_text = str(response.output)
        return WordOutput.model_validate_json(response_text)
    except Exception as e:
        raise ValueError(f"Failed to parse agent response as JSON: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w",
        "--word",
        required=True,
        help="The word to deconstruct",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed output",
    )

    args = parser.parse_args()
    
    setup_config()
    agent = create_word_agent()
    
    try:
        result = deconstruct_word(agent, args.word)
        if args.verbose:
            print(json.dumps(result.model_dump(), indent=2))
        else:
            parts = ", ".join(f"{p.text} ({p.meaning})" for p in result.parts)
            print(f"Word: {args.word}")
            print(f"Parts: {parts}")
            print(f"Definition: {result.combinations[-1][0].definition}")
    except Exception as e:
        print(f"Error deconstructing word: {e}") 