import guidance
from guidance import user, assistant, gen, select, with_temperature
import typing as ty
from mirna_curator.model.llm import STOP_TOKENS, log_usage
from mirna_curator.llm_functions.reasoning import reasoning_block

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
    temperature_reasoning: ty.Optional[float] = None,  ## None means "take it from the run config"
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
    with assistant():
        llm += reasoning_block(config, "reasoning", temperature=temperature_reasoning)
        llm += f"The final answer, based on my reasoning above is: " + with_temperature(
            select(["yes", "no"], name="answer"), temperature_selection
        )
        logger.debug("Selected answer ok")

    log_usage(llm)

    return llm["answer"], llm["reasoning"]
