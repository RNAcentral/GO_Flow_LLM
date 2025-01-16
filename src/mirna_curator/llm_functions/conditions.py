"""
Here, we define functions using the LLM to check the conditions in the 
flowchart. Each function name must match the name given for the 
condition in the flowchart
"""

import guidance
from guidance import gen, select, system, user, assistant, with_temperature
import sqlite3

import typing as ty

@guidance
def prompted_flowchart_step_bool(
    llm: guidance.models.Model,
    article_text: str,
    load_article_text: bool,
    step_prompt: str,
    rna_id: str,
    temperature_reasoning: ty.Optional[float] = 0.4,
    temperature_selection: ty.Optional[float] = 0.4,
) -> guidance.models.Model:
    """
    Use the given prompt on the article text to answer a yes/no question,
    returning a boolean
    """

    with user():
        if load_article_text:
            llm += f"You will be asked a yes/no question about the following text: \n{article_text}\n\n"
        else:
            llm += "\n\n"
        llm += f"Question: {step_prompt}\nRestrict your considerations to {rna_id} if there are multiple RNAs mentioned\n"

        llm += "Explain your reasoning step by step, be concise.\n"

    with assistant():
        llm += (
            with_temperature(gen("reasoning", max_tokens=512, stop=["<|end|>", "<|eot_id|>", "<|eom_id|>"]), temperature_reasoning)
            + "\n"
        )
    
    # with user():
    #     llm += ("Based on the reasoning above, what is your final answer to the question? "
    #             "Ensure that the final answer you give is consistent with any answer at the end of your reasoning.\n"
    #     )

    with assistant():
        llm += f"The final answer, based on my reasoning above is: " + with_temperature(
            select(["yes", "no"], name="answer"), temperature_selection
        )

    return llm
@guidance
def prompted_flowchart_terminal(llm: guidance.models.Model,
                                article_text: str,
                                load_article_text: bool,
                                detector_prompt: str,
                                rna_id: str,
                                temperature_reasoning: ty.Optional[float] = 0.4,
                                temperature_selection: ty.Optional[float] = 0.1,):
    """
    Use the LLM to find the targets and AEs for the GO annotation

    """
    with user():
        if load_article_text:
            llm += f"You will be asked a question about the following text: \n{article_text}\n\n"
        else:
            llm += "\n\n"
        
        llm += (f"Question: {detector_prompt}. Restrict your answer to the target of {rna_id}. "
                "Give some reasoning for your answer, then state the protein name as it appears in the paper.\n"
                "When stating the protein name, do not add additional formatting or an explanation of the name.\n"
        )
    with assistant():
        llm += ( "Reasoning: " 
            + with_temperature(gen("reasoning", max_tokens=512, stop=["<|end|>", "<|eot_id|>", "<|eom_id|>"]), temperature_reasoning)
            + "\n"
        )
    with assistant():
        llm += "Protein name: " + with_temperature(
            gen(max_tokens=10, name="protein_name", stop=["<|end|>", "<|eot_id|>"]), temperature_selection
        )

    return llm

    pass
