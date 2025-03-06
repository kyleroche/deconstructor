# region Imports
"""
External dependencies and type definitions
Core libraries for CLI, environment, and data handling
Griptape framework components for AI agent functionality
"""

import argparse
import os
import json
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from griptape.structures import Agent
from griptape.drivers import GriptapeCloudRulesetDriver
from griptape.rules import Ruleset
from griptape.drivers.event_listener.griptape_cloud import (
    GriptapeCloudEventListenerDriver,
)
from griptape.events import (
    EventBus,
    EventListener,
    FinishStructureRunEvent,
    FinishTaskEvent,
    BaseEvent,
)
# endregion

# region Data Model
"""
Pydantic models defining the structure for word parts and combinations
Used for type validation and JSON schema generation
"""


class WordPart(BaseModel):
    id: str = Field(
        description="Lowercase identifier, unique across parts and combinations"
    )
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
    thought: str = Field(
        description="Think about the word/phrase, it's origins, and how it's put together"
    )
    parts: List[WordPart] = Field(
        description="Array of word parts that combine to form the word"
    )
    combinations: List[List[Combination]] = Field(
        description="Layers of combinations forming a DAG to the final word"
    )


# endregion

# region Environment Setup
"""
Functions for configuring the runtime environment
Handles cloud vs local execution and API key management
If this is running in Griptape Cloud, we will publish events to the event bus
"""


def is_running_in_managed_environment() -> bool:
    return "GT_CLOUD_STRUCTURE_RUN_ID" in os.environ


def get_listener_api_key() -> str:
    api_key = os.environ.get("GT_CLOUD_API_KEY", "")
    if is_running_in_managed_environment() and not api_key:
        pass
    return api_key


def setup_config():
    if is_running_in_managed_environment():

        def event_handler(event: BaseEvent):
            if isinstance(event, FinishStructureRunEvent):
                if event.output_task_output is not None and isinstance(
                    event.output_task_output.value, BaseModel
                ):
                    event.output_task_output.value = (
                        event.output_task_output.value.model_dump()
                    )
            if isinstance(event, FinishTaskEvent):
                if event.task_output is not None and isinstance(
                    event.task_output.value, BaseModel
                ):
                    event.task_output.value = event.task_output.value.model_dump()

            return event

        event_driver = GriptapeCloudEventListenerDriver(api_key=get_listener_api_key())

        event_listener = EventListener(
            on_event=event_handler,
            event_listener_driver=event_driver,
        )

        EventBus.add_event_listener(event_listener)
    else:
        load_dotenv("../.env.local")


# endregion

# region Agent Configuration
"""
Creating and configuring the linguistic analysis agent.
Uses the pydantic model for structured output.
Rules are loaded from Griptape Cloud.
"""


def create_word_agent() -> Agent:
    setup_config()
    ruleset = Ruleset(
        name="Etymology Ruleset",
        ruleset_driver=GriptapeCloudRulesetDriver(
            ruleset_id="f887ffcf-7729-4a10-a685-ba3c3d78b5ef",
            api_key=os.environ.get("GT_CLOUD_API_KEY", ""),
        ),
    )

    return Agent(output_schema=WordOutput, rulesets=[ruleset])


# endregion

# region Word Deconstruction
"""
Functions for deconstructing words using the linguistic analysis agent. 
Handles prompt construction and result parsing
"""


def deconstruct_word(agent: Agent, word: str, previous_attempts: list = None) -> dict:
    prompt = f"""Your task is to deconstruct this EXACT word: '{word}'
Do not analyze any other word. Focus only on '{word}'.
Break down '{word}' into its etymological components."""

    if previous_attempts:
        prompt += f"\n\nPrevious attempts:\n{json.dumps(previous_attempts, indent=2)}\n\nPlease fix all the issues and try again."

    response = agent.run(prompt)
    try:
        output = response.output.value

        if isinstance(output, WordOutput):
            result = output.model_dump()
            return result
        return output
    except Exception as e:
        raise ValueError(f"Failed to parse agent response: {e}")


# endregion

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
            print(json.dumps(result, indent=2))
        else:
            # Handle result as dict now
            parts = ", ".join(f"{p['text']} ({p['meaning']})" for p in result["parts"])
            print(f"Word: {args.word}")
            print(f"Parts: {parts}")
            print(f"Definition: {result['combinations'][-1][0]['definition']}")
    except Exception as e:
        print(f"Error deconstructing word: {e}")
