"""
Here, we define functions using the LLM to check the conditions in the 
flowchart. Each function name must match the name given for the 
condition in the flowchart
"""

import guidance
from guidance import gen, select, system, user, assistant, with_temperature
import sqlite3

import typing as ty


def prompted_flowchart_step_bool(
    llm: guidance.models.Model,
    article_text: str,
    step_prompt: str,
    temperature_reasoning: ty.Optional[float] = 0.4,
    temperature_selection: ty.Optional[float] = 0.1,
    store_reasoning: ty.Optional[sqlite3.Connection] = None,
) -> bool:
    """
    Use the given prompt on the article text to answer a yes/no question,
    returning a boolean
    """

    with user():
        llm += f"You will be asked a yes/no question about the following text: \n{article_text}\n\n"

        llm += f"Question: {step_prompt}\n"

        llm += "Explain your reasoning step by step, and answer yes or no"

    with assistant():
        llm += (
            with_temperature(gen("reasoning", max_tokens=512), temperature_reasoning)
            + "\n"
        )

    with assistant():
        llm += "So the final answer is " + with_temperature(
            select(["yes", "no"], name="answer"), temperature_selection
        )

    # print(llm)

    return llm["answer"] == "yes"


def prompted_flowchart_terminal():
    """
    Use the LLM to find the targets and AEs for the GO annotation

    """
    pass
