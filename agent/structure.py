# region Imports
"""
External dependencies and type definitions
Core libraries for CLI, environment, and data handling
Griptape framework components for AI agent functionality
"""

import argparse
import json
from typing import List
from griptape.utils import GriptapeCloudStructure
from pydantic import BaseModel, Field
from griptape.structures import Agent
from griptape.drivers.ruleset.griptape_cloud import GriptapeCloudRulesetDriver
from griptape.rules import Ruleset
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

# region Agent Configuration
"""
Creating and configuring the linguistic analysis agent.
Uses the pydantic model for structured output.
Rules are loaded from Griptape Cloud.
"""


def create_word_agent() -> Agent:
    ruleset = Ruleset(
        name="Etymology Ruleset",
        ruleset_driver=GriptapeCloudRulesetDriver(
            ruleset_id="f887ffcf-7729-4a10-a685-ba3c3d78b5ef"
        ),
    )

    return Agent(output_schema=WordOutput, rulesets=[ruleset])


# endregion

# region Word Deconstruction
"""
Functions for deconstructing words using the linguistic analysis agent. 
Handles prompt construction and result parsing
"""


def deconstruct_word(
    agent: Agent, word: str, previous_attempts: list | None = None
) -> WordOutput:
    prompt = f"""Your task is to deconstruct this EXACT word: '{word}'
Do not analyze any other word. Focus only on '{word}'.
Break down '{word}' into its etymological components."""

    if previous_attempts:
        prompt += f"\n\nPrevious attempts:\n{json.dumps(previous_attempts, indent=2)}\n\nPlease fix all the issues and try again."

    agent.run(prompt)
    if isinstance(agent.output.value, WordOutput):
        return agent.output.value
    else:
        raise ValueError("Agent output is not a WordOutput")


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

    # region Environment Setup
    with GriptapeCloudStructure(observe=True):
        agent = create_word_agent()

        # endregion
        try:
            result = deconstruct_word(agent, args.word)
            if args.verbose:
                print(json.dumps(result.model_dump(), indent=2))
            else:
                # Handle result as dict now
                parts = ", ".join(f"{p.text} ({p.meaning})" for p in result.parts)
                print(f"Word: {args.word}")
                print(f"Parts: {parts}")
                print(f"Definition: {result.combinations[-1][0].definition}")
        except Exception as e:
            print(f"Error deconstructing word: {e}")
