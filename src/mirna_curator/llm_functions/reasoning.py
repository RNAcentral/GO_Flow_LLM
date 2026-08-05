"""
Model-agnostic handling of the reasoning (chain-of-thought) channel.

The chain of thought is part of the curation record, so it is always captured into its
own named guidance variable regardless of which model is loaded. What varies between
models is only how the thinking region is *delimited* in the prompt, which is what
REASONING_STYLES describes.

Note that guidance never renders a model's jinja chat template - it drives the token
stream itself and only takes static role markers from its ChatTemplate classes. So
llama.cpp's own `--jinja` / `--reasoning-format` handling is unreachable from here and
the markers have to be injected explicitly. Entries for this table are derived from a
model's real chat template offline by `scripts/extract_reasoning_style.py`.
"""

import re
import guidance
from guidance import gen, special_token, with_temperature
import typing as ty

from mirna_curator.model.llm import STOP_TOKENS

import logging

logger = logging.getLogger(__name__)


## Reasoning temperature of 0.6 is the DeepSeek-R1 recommendation, and was what the
## original QwQ-32B runs used. Only applied if the run config doesn't set one.
DEFAULT_REASONING_TEMPERATURE = 0.6


## `open` and `close` bracket the generated reasoning; `stop` are the extra stop strings
## that end the thinking region for this style. `stop_special` is the same thing for a
## terminator that is a single *special* token (Gemma 4's `<channel|>`): those cannot be
## expressed as text stops, so they go through guidance's `special_token()` instead, and
## guidance allows exactly one of them and won't mix it with text stops. A special-token
## stop is *generated*, so it lands in the transcript by itself and `close` must not
## repeat it. `close` is emitted unconditionally, which
## is what guarantees the block is balanced - guidance consumes stop text without writing
## it back into model state, so a generation that ends on `</think>` would otherwise leave
## the tag dangling.
##
## `gate` is text that has to appear in the *system* turn to switch thinking on at all.
## Only Gemma 4 needs it: `<think>`-style models are gated by the assistant prefill, which
## `open` already covers, but Gemma 4 reads a `<|think|>` token from the system turn and
## keeps the reasoning itself in a named channel inside the assistant turn.
## `reasoning_format` and `reasoning_structure` fill the same-named placeholders in a
## flowchart's system prompt. They exist because the system prompt used to hard-code QwQ's
## `<think>`/`\boxed{}` contract, which told every other model to ignore its own thinking
## markers - so the style's `stop` never fired and every reasoning block ran to max_tokens.
## A style must describe the output contract it actually expects.
REASONING_STYLES = {
    "none": {
        "open": "Reasoning: ",
        "close": "\n",
        "stop": [],
        "gate": "",
        "reasoning_format": "",
        "reasoning_structure": "",
    },
    "think": {
        "open": "<think>\n",
        "close": "\n</think>\n\n",
        "stop": ["</think>"],
        "gate": "",
        ## Reproduces the original QwQ system prompt exactly, so those runs are unchanged
        "reasoning_format": (
            "MANDATORY FORMAT:\n"
            "\n"
            "Present your reasoning in <think></think> tags\n"
            "Provide your final answer in \\boxed{answer} format\n"
        ),
        "reasoning_structure": (
            "RESPONSE STRUCTURE:\n"
            "<think>\n"
            "[Your analytical reasoning about the question]\n"
            "</think>\n"
            "\\boxed{[Your definitive answer]}\n"
            "\n"
            "Follow this format without exception for every response."
        ),
    },
    "gemma4": {
        "open": "<|channel>thought\n",
        ## Nothing to close with: the stop token is part of the grammar body, so it is
        ## emitted into the transcript by the generation itself - forced by the parser even
        ## if `max_tokens` runs out first - and writing it again here would double it.
        "close": "",
        ## `<channel|>` is a single *special* token (id 101). Text stops are compiled into a
        ## regex over bytes, which a special token has no representation in, so it goes in
        ## `stop_special` instead - see the comment on that key below.
        "stop": [],
        "stop_special": "<channel|>",
        "gate": "<|think|>\n",
        ## Otherwise deliberately empty - Gemma 4 handles its own channel, and imposing a
        ## foreign format contract is what made it run away.
        "reasoning_format": "",
        "reasoning_structure": "",
    },
}


def get_reasoning_style(config: ty.Dict[str, ty.Any]) -> ty.Dict[str, ty.Any]:
    """
    Look up the reasoning style for a run configuration.

    Arguments:
        config: dict - the run configuration. A missing `reasoning_style` means "none",
                       so callers that pass no config at all still work.

    Returns:
        style: dict - the open/close/stop definition for that style
    """
    name = config.get("reasoning_style", "none")
    if name not in REASONING_STYLES:
        raise ValueError(
            f"Unknown reasoning_style {name!r}. Known styles: {sorted(REASONING_STYLES)}. "
            "Derive a new one from the model's chat template with "
            "scripts/extract_reasoning_style.py"
        )
    return REASONING_STYLES[name]


def get_reasoning_gate(config: ty.Dict[str, ty.Any]) -> str:
    """
    Text that must lead the system turn for this style to think at all.

    Empty for every style except gemma4. Callers should emit a system turn containing
    just this if the run has no system prompt of its own, otherwise thinking stays off
    and the channel markers never appear.

    Arguments:
        config: dict - the run configuration

    Returns:
        str - the gate text, possibly empty
    """
    return get_reasoning_style(config).get("gate", "")


def apply_reasoning_format(system_prompt: str, config: ty.Dict[str, ty.Any]) -> str:
    """
    Fill a flowchart's system prompt with the output contract for this reasoning style.

    Substitutes the `{reasoning_format}` and `{reasoning_structure}` placeholders. Plain
    str.replace rather than str.format, because the QwQ contract contains literal braces
    (`\\boxed{answer}`) that format() would try to interpret as fields.

    A prompt with no placeholders is returned unchanged, so flowcharts that never had a
    system prompt, or have a model-agnostic one, keep working untouched.

    Arguments:
        system_prompt: str - the raw prompt from the flowchart's prompts.json
        config: dict - the run configuration, read for `reasoning_style`

    Returns:
        str - the prompt with the style's format contract substituted in
    """
    style = get_reasoning_style(config)
    filled = system_prompt
    for key in ("reasoning_format", "reasoning_structure"):
        filled = filled.replace("{" + key + "}", style.get(key, ""))

    ## Styles that supply nothing leave blank runs behind where the blocks used to be
    return re.sub(r"\n{3,}", "\n\n", filled).strip()


@guidance
def reasoning_block(
    llm: guidance.models.Model,
    config: ty.Dict[str, ty.Any],
    name: str,
    max_tokens: int = 1024,
    temperature: ty.Optional[float] = None,
) -> guidance.models.Model:
    """
    Generate a block of reasoning into its own capture, delimited per the run's
    reasoning style.

    Arguments:
        llm: the guidance model
        config: dict - the run configuration, read for `reasoning_style` and `temperature`
        name: str - the capture name, e.g. "reasoning" or "detector_reasoning". This is
                    what ends up in the NDJSON curation trace, so it must not change
                    casually.
        max_tokens: int - generation cap for the reasoning
        temperature: float - overrides the run config if given

    Returns:
        llm: the model, with the reasoning captured under `name`
    """
    style = get_reasoning_style(config)
    if temperature is None:
        temperature = config.get("temperature", DEFAULT_REASONING_TEMPERATURE)

    special = style.get("stop_special")
    stop = special_token(special) if special else STOP_TOKENS + style["stop"]

    llm += style["open"]
    llm += with_temperature(
        gen(name, max_tokens=max_tokens, stop=stop),
        temperature,
    )
    llm += style["close"]
    logger.debug("Generated %s ok", name)
    return llm
