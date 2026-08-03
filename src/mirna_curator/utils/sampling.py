from guidance._schema import SamplingParams


## The single source of truth for sampling defaults - the CLI seeds its defaults from
## this, and model loading falls back to it for the parameters guidance doesn't carry.
## Only top_p/top_k/min_p/repetition_penalty fit in guidance's SamplingParams; the rest
## are passed to LlamaCpp directly.
DEFAULT_SAMPLING_PARAMS = {
    "temperature": 0.6,
    "min_p": 0.00,
    "top_k": 40,
    "top_p": 0.95,  # This configuration from danhanchen of Unsloth, should
    "repetition_penalty": 1.1,  # reduce the repetition on reasoning
    "dry_multiplier": 0.5,
}


def get_sampling_params(config: dict) -> SamplingParams:
    """
    Convert our sampling parameters dictionary into a guidance sampling parameters typedDict to
    support calling `with_sampling_params`

    For now, only these parameters are supported

    Arguments:
        config: dict - The run configuration dictionary, should contain the sampling parameters too

    Returns:
        sampling_params: SamplingParams - a guidance sampling parameters TypedDict
    """

    merged = {**DEFAULT_SAMPLING_PARAMS, **config}

    sampling_params = SamplingParams(
        top_p = merged["top_p"],
        top_k = merged["top_k"],
        min_p = merged["min_p"],
        repetition_penalty = merged["repetition_penalty"],
    )

    return sampling_params
