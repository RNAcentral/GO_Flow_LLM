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
from mirna_curator.llm_functions.tools import safe_import
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
    config: ty.Optional[ty.Dict[str, ty.Any]] = {},
    temperature_reasoning: ty.Optional[float] = 0.6,
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

        llm += "Explain your reasoning step-by-step. Be concise\n"

    logger.info(f"LLM input tokens: {llm.engine.metrics.engine_input_tokens}")
    logger.info(f"LLM generated tokens: {llm.engine.metrics.engine_output_tokens}")
    logger.info(
        f"LLM total tokens: {llm.engine.metrics.engine_input_tokens + llm.engine.metrics.engine_output_tokens}"
    )
    with assistant():
        llm += (
            "Reasoning:\n"
        )
        if config['deepseek_mode']:
            llm += "<think>\n"
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
        logger.info("Generated reasoning ok")

    with assistant():
        llm += f"The final answer, based on my reasoning above is: " + with_temperature(
            select(["yes", "no"], name="answer"), temperature_selection
        )
        logger.info("Selected answer ok")

    llm += extract_evidence(article_text, mode=config.get("evidence_mode", "single-sentence"))
    logger.info("Evidence extracted, ready to return")

    return llm


@guidance
def prompted_flowchart_step_tool(
    llm: guidance.models.Model,
    article_text: str,
    load_article_text: bool,
    step_prompt: str,
    rna_id: str,
    config: ty.Optional[ty.Dict[str, ty.Any]] = {},
    tools: ty.Optional[ty.List[str]] = [],
    temperature_reasoning: ty.Optional[float] = 0.6,
    temperature_selection: ty.Optional[float] = 0.4,
) -> guidance.models.Model:
    """
    Use the given prompt on the article text to answer a yes/no question,
    returning a boolean

    Args:
        lm: guidance.models.Model: A guidance model that's ready to go
        article_text: str: The article text we need to work on
        load_article_text: bool : Flag for whether we are going to load the text into the context
        step_prompt: str: The prompt read from the flowchart json file
        rna_id: str: The RNA id we are working on
        tools: ty.Optional[ty.List[str]] = []: A list of tools for the LLM to use
        temperature_reasoning: ty.Optional[float] = 0.6: The reasoning temperature (0.6 is R1 reccomended)
        temperature_selection: ty.Optional[float] = 0.4: The yes/no selection temperature
    """

    ## build the tool description string.
    tool_dict = safe_import(tools)
    tools_string = (
        "To help me answer this question, I have access to some tools to look"
        " up some information. The tools are described here:\n"
        "===========================\n"
    )

    for tool_name, tool in tool_dict.items():
        tools_string += f"Name: {tool_name}\n+++++++++++\n"
        tools_string += f"Description:\n{tool.__doc__}"
        tools_string += "\n+++++++++++\n"

    tools_string += f"Name: finish\n+++++++++++\n"
    tools_string += (
        f"Description:\nEnd the searching process and move on to answering the question"
    )
    tools_string += "\n+++++++++++\n"

    tools_string += "===========================\n"

    _tools = tools
    _tools.append("finish")
    with user():
        if load_article_text:
            logger.info(
                f"Appending {len(llm.engine.tokenizer.encode(article_text.encode('utf-8')))} tokens (internal node)"
            )
            llm += f"You will be asked a yes/no question. The answer could be in following text, or it could be in some text you have already seen: \n{article_text}\n\n"
        else:
            llm += "\n\n"

        llm += f"Question: {step_prompt}\n"

    ## Make a tiny little ReAct agent loop
    i = 0
    max_steps = 5
    with assistant():
        llm += tools_string
        while True:
            llm += f"Thought {i}: " + gen(suffix="\n")
            llm += f"Act {i}: " + select(_tools, name="act")
            llm += "[" + gen(name="arg", suffix="]") + "\n"
            if llm["act"].lower() == "finish" or i > max_steps:
                break
            else:
                logger.info(f"calling {llm['act']} with argument {llm['arg']}")
                tool_output = tool_dict[llm["act"]](llm["arg"])
                llm += f"Observation {i}: {tool_output}\n"
            i += 1
        # Restrict your considerations to {rna_id} if there are multiple RNAs mentioned\n"

        llm += "Explain your reasoning step-by-step. Be concise\n"

    logger.info(f"LLM input tokens: {llm.engine.metrics.engine_input_tokens}")
    logger.info(f"LLM generated tokens: {llm.engine.metrics.engine_output_tokens}")
    logger.info(
        f"LLM total tokens: {llm.engine.metrics.engine_input_tokens + llm.engine.metrics.engine_output_tokens}"
    )
    with assistant():
        llm += (
            "Reasoning:\n"
        )
        if config['deepseek_mode']:
            llm += "<think>\n"
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

    llm += extract_evidence(article_text, mode=config.get("evidence_mode", "single-sentence"))

    return llm


@guidance
def prompted_flowchart_terminal(
    llm: guidance.models.Model,
    article_text: str,
    load_article_text: bool,
    detector_prompt: str,
    rna_id: str,
    paper_id: str,
    config: ty.Optional[ty.Dict[str, ty.Any]] = {},
    temperature_reasoning: ty.Optional[float] = 0.6,
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
            "Remember: we are looking for the target of the miRNA mentioned in this paper, do not recall what you know about the miRNA.\n"
        )
    logger.info(f"LLM input tokens: {llm.engine.metrics.engine_input_tokens}")
    logger.info(f"LLM generated tokens: {llm.engine.metrics.engine_output_tokens}")
    logger.info(
        f"LLM total tokens: {llm.engine.metrics.engine_input_tokens + llm.engine.metrics.engine_output_tokens}"
    )
    with assistant():
        llm += (
            "Reasoning:\n"
        )
        if config['deepseek_mode']:
            llm += "<think>\n"
        llm += (
            with_temperature(
                gen(
                    "detector_reasoning",
                    max_tokens=1024,
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

    llm += extract_evidence(article_text, mode=config.get("evidence_mode", "single-sentence"))

    return llm
