import guidance
from guidance import user, assistant, gen, select, with_temperature
import typing as ty
from mirna_curator.model.llm import STOP_TOKENS

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prompted_filter(
    llm: guidance.models.Model,
    article_text: str,
    _load_article_text: bool,  ## Always load it, but keep this argument for signature compatibility
    filter_prompt: str,
    rna_id: str,
    config: ty.Optional[ty.Dict[str, ty.Any]] = {},
    temperature_reasoning: ty.Optional[float] = 0.6,
    temperature_selection: ty.Optional[float] = 0.1,
) -> str:
    """
    This is not a guidance function, so the results of this do not get persisted in model state

    """
    with user():
        logger.info(
            f"Appending {len(llm.engine.tokenizer.encode(article_text.encode('utf-8')))} tokens (filter node)"
        )
        llm += f"You will be asked a question about the following text: \n{article_text}\n\n"
        llm += f"Question: {filter_prompt}. Restrict your answer to the target of {rna_id}. "
    logger.info(f"LLM input tokens: {llm._get_usage().input_tokens}")
    logger.info(f"LLM generated tokens: {llm._get_usage().output_tokens}")
    logger.info(
        f"LLM total tokens: {llm._get_usage().input_tokens + llm._get_usage().output_tokens}"
    )
    with assistant():
        llm += (
            "Reasoning: "
            + with_temperature(
                gen(
                    "reasoning",
                    max_tokens=1024,
                    stop=STOP_TOKENS,
                ),
                temperature_reasoning,
            )
            + "\n"
        )
        llm += f"The final answer, based on my reasoning above is: " + with_temperature(
            select(["yes", "no"], name="answer"), temperature_selection
        )
        logger.debug("Selected answer ok")

    return llm["answer"], llm["reasoning"]
