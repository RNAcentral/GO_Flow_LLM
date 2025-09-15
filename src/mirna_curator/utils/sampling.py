from guidance._schema import SamplingParams


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

    sampling_params = SamplingParams(
        top_p = config.get("top_p", 0.95),
        top_k = config.get("top_k", 40),
        min_p = config.get("min_p", 0.0),
        repetition_penalty = config.get("repetition_penalty", 1.0)
    )

    return sampling_params