"""
Here, we define functions using the LLM to check the conditions in the 
flowchart. Each function name must match the name given for the 
condition in the flowchart
"""

import guidance
from guidance import gen, select, system, user, assistant, with_temperature, substring

from mirna_curator.llm_functions.evidence import extract_evidence
from mirna_curator.apis import epmc
from mirna_curator.model.llm import STOP_TOKENS
import typing as ty

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            logger.info(
                f"Appending {len(llm.engine.tokenizer.encode(article_text.encode('utf-8')))} tokens (internal node)"
            )
            llm += f"You will be asked a yes/no question. The answer could be in following text, or it could be in some text you have already seen: \n{article_text}\n\n"
        else:
            llm += "\n\n"
        llm += f"Question: {step_prompt}\nRestrict your considerations to {rna_id} if there are multiple RNAs mentioned\n"

        llm += "Explain your reasoning step-by-step. Try to be concise\n"
            
    logger.info(f"LLM input tokens: {llm.engine.metrics.engine_input_tokens}")
    logger.info(f"LLM generated tokens: {llm.engine.metrics.engine_output_tokens}")
    logger.info(
        f"LLM total tokens: {llm.engine.metrics.engine_input_tokens + llm.engine.metrics.engine_output_tokens}"
    )
    with assistant():
        llm += (
            with_temperature(
                gen(
                    "reasoning",
                    max_tokens=1024,
                    stop=STOP_TOKENS,
                ),
                temperature_reasoning,
            )
            + "\n"
        )

    with assistant():
        llm += f"The final answer, based on my reasoning above is: " + with_temperature(
            select(["yes", "no"], name="answer"), temperature_selection
        )

    llm += extract_evidence(article_text, mode="recursive-paragraph")

    return llm


@guidance
def prompted_flowchart_terminal(
    llm: guidance.models.Model,
    article_text: str,
    load_article_text: bool,
    detector_prompt: str,
    rna_id: str,
    paper_id: str,
    temperature_reasoning: ty.Optional[float] = 0.4,
    temperature_selection: ty.Optional[float] = 0.1,
):
    """
    Use the LLM to find the targets and AEs for the GO annotation

    """
    epmc_annotated_genes = epmc.get_gene_name_annotations(paper_id)
    with user():
        if load_article_text:
            logger.info(
                f"Appending {len(llm.engine.tokenizer.encode(article_text.encode('utf-8')))} tokens (terminal node)"
            )
            llm += f"You will be asked a question about the following text: \n{article_text}\n\n"
        else:
            llm += "\n\n"
        llm += (
            f"Question: {detector_prompt}. Restrict your answer to the target of {rna_id}. "
            "Give some reasoning for your answer, then state the miRNA's target protein name as it appears in the paper.\n"
        )
    logger.info(f"LLM input tokens: {llm.engine.metrics.engine_input_tokens}")
    logger.info(f"LLM generated tokens: {llm.engine.metrics.engine_output_tokens}")
    logger.info(
        f"LLM total tokens: {llm.engine.metrics.engine_input_tokens + llm.engine.metrics.engine_output_tokens}"
    )
    with assistant():
        llm += (
            "Reasoning: "
            + with_temperature(
                gen(
                    "detector_reasoning",
                    max_tokens=128,
                    stop=STOP_TOKENS,
                ),
                temperature_reasoning,
            )
            + "\n"
        )
    with assistant():
        llm += f"Protein name: {select(epmc_annotated_genes, name='protein_name')}"
        # with_temperature(
        #     gen(max_tokens=10, name="protein_name", stop=["<|end|>", "<|eot_id|>"]), temperature_selection
        # )

    llm += extract_evidence(article_text, mode="recursive-paragraph")

    return llm
