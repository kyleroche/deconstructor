# region Imports
"""External dependencies and type definitions.

Core libraries for CLI, environment, and data handling
Griptape framework components for AI agent functionality
"""

import argparse
import json
import logging

from griptape.drivers.observability.open_telemetry import (
    OpenTelemetryObservabilityDriver,
)
from griptape.drivers.ruleset.griptape_cloud import GriptapeCloudRulesetDriver
from griptape.observability import Observability
from griptape.rules import Ruleset
from griptape.structures import Agent
from griptape.utils import GriptapeCloudStructure
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic import BaseModel, Field

# endregion

logger = logging.getLogger(__name__)
# region Data Model
"""
Pydantic models defining the structure for word parts and combinations
Used for type validation and JSON schema generation
"""


class WordPart(BaseModel):
    id: str = Field(description="Lowercase identifier, unique across parts and combinations")
    text: str = Field(description="Exact section of input word")
    originalWord: str = Field(description="Oldest word/affix this part comes from")  # noqa: N815
    origin: str = Field(description="Brief origin (Latin, Greek, etc)")
    meaning: str = Field(description="Concise meaning of this part")


class Combination(BaseModel):
    id: str = Field(description="Unique lowercase identifier")
    text: str = Field(description="Combined text segments")
    definition: str = Field(description="Clear definition of combined parts")
    sourceIds: list[str] = Field(description="Array of part/combination ids used")  # noqa: N815


class WordOutput(BaseModel):
    thought: str = Field(description="Think about the word/phrase, it's origins, and how it's put together")
    parts: list[WordPart] = Field(description="Array of word parts that combine to form the word")
    combinations: list[list[Combination]] = Field(description="Layers of combinations forming a DAG to the final word")


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
        ruleset_driver=GriptapeCloudRulesetDriver(ruleset_id="f887ffcf-7729-4a10-a685-ba3c3d78b5ef"),
    )

    return Agent(output_schema=WordOutput, rulesets=[ruleset])


# endregion

# region Word Deconstruction
"""
Functions for deconstructing words using the linguistic analysis agent.
Handles prompt construction and result parsing
"""


def deconstruct_word(agent: Agent, word: str, previous_attempts: list | None = None) -> WordOutput:
    prompt = f"""Your task is to deconstruct this EXACT word: '{word}'
Do not analyze any other word. Focus only on '{word}'.
Break down '{word}' into its etymological components."""

    if previous_attempts:
        prompt += (
            f"\n\nPrevious attempts:\n{json.dumps(previous_attempts, indent=2)}\n\n"
            "Please fix all the issues and try again."
        )

    agent.run(prompt)
    if isinstance(agent.output.value, WordOutput):
        return agent.output.value
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
    with GriptapeCloudStructure(
        observe=True,
        observability=Observability(
            observability_driver=OpenTelemetryObservabilityDriver(
                service_name="etymology",
                span_processor=BatchSpanProcessor(OTLPSpanExporter()),
            )
        ),
    ):
        agent = create_word_agent()

        # endregion
        try:
            result = deconstruct_word(agent, args.word)
            if args.verbose:
                logger.info(json.dumps(result.model_dump(), indent=2))
            else:
                # Handle result as dict now
                parts = ", ".join(f"{p.text} ({p.meaning})" for p in result.parts)
                logger.info("Word: %s", args.word)
                logger.info("Parts: %s", parts)
                logger.info("Definition: %s", {result.combinations[-1][0].definition})
        except Exception:
            logger.exception("Error deconstructing word")
