"""
Here, we define functions using the LLM to check the conditions in the 
flowchart. Each function name must match the name given for the 
condition in the flowchart
"""

import guidance
from guidance import gen, select, system, user, assistant, with_temperature, substring
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
            llm += f"You will be asked a yes/no question. The answer could be in following text, or it could be in some text you have already seen: \n{article_text}\n\n"
        else:
            llm += "\n\n"
        llm += f"Question: {step_prompt}\nRestrict your considerations to {rna_id} if there are multiple RNAs mentioned\n"

        llm += ("Explain your reasoning step-by-step, using these guidelines:\n"
            "Use mathematical notation where possible\n"
            "Skip obvious steps\n"
            "Use brief variable names\n"
            "Structure as 'If A then B because C'\n"
            "Maximum 10 words per step\n"
            "Use symbols (→, =, ≠, etc.) instead of words\n"
            "Abbreviate common terms (prob/probability, calc/calculate)\n"
            "Your response should be clear but minimal. Show key logical steps only.\n"
        )

    with assistant():
        llm += (
            with_temperature(gen("reasoning", max_tokens=512, stop=["<|end|>", "<|eot_id|>", "<|eom_id|>"]), temperature_reasoning)
            + "\n"
        )

    with assistant():
        llm += f"The final answer, based on my reasoning above is: " + with_temperature(
            select(["yes", "no"], name="answer"), temperature_selection
        )
    
    with user():
        llm += "Give a piece of evidence from the text that supports your answer. Choose the most relevant sentence or two.\n"

    with assistant():
        llm += f"The most relevant piece of evidence is: '{substring(article_text, name='evidence')}'"
    

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
            + with_temperature(gen("detector_reasoning", max_tokens=512, stop=["<|end|>", "<|eot_id|>", "<|eom_id|>"]), temperature_reasoning)
            + "\n"
        )
    with assistant():
        llm += f"Protein name: {substring(article_text, name='protein_name')}"
        # with_temperature(
        #     gen(max_tokens=10, name="protein_name", stop=["<|end|>", "<|eot_id|>"]), temperature_selection
        # )
    with user():
        llm += "Give a piece of evidence from the text that supports your answer. Choose the most relevant piece of evidence.\n"
    with assistant():
        llm += f"The most relevant piece of evidence is: '{substring(article_text, name='detector_evidence')}'"

    return llm

    pass
