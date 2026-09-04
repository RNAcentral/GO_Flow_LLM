from guidance.models import LlamaCpp
from guidance.chat import (
    ChatTemplate,
    UnsupportedRoleException,
    ChatMLTemplate,
    Llama2ChatTemplate,
    Llama3ChatTemplate,
    Phi3MiniChatTemplate,
    Phi3SmallMediumChatTemplate,
    Mistral7BInstructChatTemplate,
    Gemma29BInstructChatTemplate,
    Qwen2dot5ChatTemplate,
    Qwen3ChatTemplate
)

class Gemma4ChatTemplate(ChatTemplate):
    """
    Chat template for Gemma 4, which guidance does not ship a class for.

    Nothing carries over from Gemma 2 - the turn markers changed from
    `<start_of_turn>`/`<end_of_turn>` to `<|turn>`/`<turn|>`, so
    Gemma29BInstructChatTemplate produces tokens that are not in Gemma 4's vocabulary
    and the model never sees a turn boundary. Gemma 4 does support a system role, which
    Gemma 2 did not. The assistant is still called `model`.

    Markers extracted from google/gemma-4-E2B-it by scripts/extract_reasoning_style.py.
    A ChatTemplate subclass is all guidance needs - load_template_class accepts one
    directly - so this does not depend on anything landing upstream.
    """

    def get_role_start(self, role_name):
        if role_name == "assistant":
            return "<|turn>model\n"
        elif role_name in ("system", "user"):
            return f"<|turn>{role_name}\n"
        else:
            raise UnsupportedRoleException(role_name, self)

    def get_role_end(self, role_name=None):  # noqa ARG002
        return "<turn|>\n"


TEMPLATE_LOOKUP = {
    "chatml": ChatMLTemplate,
    "llama2": Llama2ChatTemplate,
    "llama3": Llama3ChatTemplate,
    "phi3-mini": Phi3MiniChatTemplate,
    "phi3-med": Phi3SmallMediumChatTemplate,
    "mistral": Mistral7BInstructChatTemplate,
    "gemma": Gemma29BInstructChatTemplate,
    "qwen25": Qwen2dot5ChatTemplate,
    "qwen3": Qwen3ChatTemplate,
    "gemma4": Gemma4ChatTemplate,
}

from huggingface_hub import HfFileSystem, hf_hub_download
from pathlib import Path
import re
import logging

from mirna_curator.utils.sampling import get_sampling_params


logger = logging.getLogger(__name__)


## Model-independent turn-enders only. Anything specific to a reasoning style (e.g.
## `</think>`) belongs in REASONING_STYLES in llm_functions/reasoning.py, so that a
## non-thinking model doesn't carry stop strings it will never emit.
STOP_TOKENS = ["<|end|>", "<|eot_id|>", "<|eom_id|>", "<|im_end|>", "<|endoftext|>"]


def get_chat_template(name):
    """
    Look up a guidance ChatTemplate class by short name.

    This raises rather than falling back to ChatML, because the fallback is silent and
    a wrong template is not obviously wrong at runtime - it just quietly degrades the
    output. A typo here previously sent a whole Qwen3 run through the ChatML template.

    Arguments:
        name: str - a key of TEMPLATE_LOOKUP

    Returns:
        The guidance ChatTemplate subclass for that model family
    """
    if name not in TEMPLATE_LOOKUP:
        raise ValueError(
            f"Unknown chat_template {name!r}. Known templates: {sorted(TEMPLATE_LOOKUP)}. "
            "Role markers for a model guidance has no class for can be derived with "
            "scripts/extract_reasoning_style.py"
        )
    return TEMPLATE_LOOKUP[name]


def log_usage(llm):
    """
    Log how full the context is, and how much has been generated.

    guidance's usage API is private, so it is wrapped here to give one place to fix when
    guidance is upgraded.

    Careful with `usage.input_tokens`: it is incremented per *forward pass*, so it
    re-counts the whole KV cache for every token generated and reaches millions against
    a 32k context. It is a billing-style "tokens processed" counter, not a prompt size.
    The distinct tokens actually in the context are `input_tokens - cached_input_tokens`,
    which is what matters for staying under n_ctx, and is free to compute (no
    re-tokenising of the transcript).

    Arguments:
        llm: the guidance model to report usage for
    """
    usage = llm._get_usage()
    prompt_tokens = usage.input_tokens - usage.cached_input_tokens

    ## n_ctx comes off the underlying llama_cpp model; don't let a private-API change
    ## take a run down over a log line
    try:
        n_ctx = llm.engine.model_obj.n_ctx()
        fill = f" ({100 * prompt_tokens / n_ctx:.0f}% of {n_ctx})"
    except Exception:  # noqa BLE001
        fill = ""

    logger.info(
        "Context %d tokens%s; generated %d tokens so far (%d processed incl. cache re-reads)",
        prompt_tokens,
        fill,
        usage.output_tokens,
        usage.input_tokens,
    )


def download_split_file(repo_id, filenames):
    """
    Large models are split on hf-hub, this downloads and reconsitutes them for loading

    We download each file in turn to a local dir, thenconcatenate it to one big file
    Each downloaded file will be immediately deleted to save space

    Parameters
    ----------
    repo_id : str
        The name of the huggingface repo we will be pulling from
    filenames : List[str]
        The list of chunks to download

    Returns
    -------
    List[str]
        List of locally downloaded shards

    Raises
    ------
        ValueError
            When the number of downloaded shards does not match how many shard the
            filenames claim there should be

    """
    expected_file_count = int(filenames[0].split("-of-")[-1].replace(".gguf", ""))
    local_filenames = []
    for remote_filename in filenames:
        remote_filepath = Path(remote_filename)
        if len(remote_filepath.parts) > 3:
            subdir = remote_filepath.parts[-2]
        else:
            subdir = None
        local_path = hf_hub_download(
            repo_id=repo_id, filename=Path(remote_filename).name, subfolder=subdir
        )
        local_filenames.append(local_path)

    if len(local_filenames) != expected_file_count:
        raise ValueError(
            "Number of downloaded shards does not match expected shards based on filename!"
        )

    return local_filenames


def get_model(
    model_name: str,
    chat_template: str = None,
    quantization: str = None,
    context_length: int = 16384,
    run_config_options: dict | None = None,
):
    """
    Load a llama.cpp model, either locally or by downloading from huggingface

    Note - this will cache the models, so make sure the HF_HOME environment
    variable is set appropriately.

    Parameters:
        model_name: str
            The local filepath, or huggingface hub ID of the model to use

        chat_template (optional): str
            The chat template to use when formatting interactions with the model.
            Defaults to chatml. For best results ensure this is set correctly

        quantization (optional): str
            What quantization type/level to use. This is required when loading from
            a hf hub repo that contains multiple models.

        context_length (optional): int
            The context length to use when interacting with the model. Defaults to 16384

    Returns:
        model: guidance.LlamaCpp
            A guidance-wrapped Llama.cpp model instance

    Raises:
        FileNotFoundError:
            When:
                - the local file doesn not exist
                - the model repo on huggingface does not exist
                - The model repo contains no gguf files
        ValueError:
            When:
                - When no quant type specified for a repo with multiple ggufs
                - When the requested quant tyoe was not found in the repo



    """
    fs = HfFileSystem()

    if Path(model_name).exists():
        logging.debug("Loading local model from path %s", model_name)
        # Don't need to do anything really
        model_path = model_name
    elif fs.exists(model_name):
        logging.debug("Downloading a gguf file from hub, then loading it")
        # Search the repo in hub for gguf files
        gguf_files = fs.glob(f"{model_name}/**/*.gguf")
        if len(gguf_files) == 0:
            logging.error(
                "There are no gguf files in the provided repo! Can't load anything"
            )
            raise FileNotFoundError(
                "There are no gguf files in the provided repo! Can't load anything"
            )
        elif len(gguf_files) == 1:
            remote_filename = Path(gguf_files[0]).name
            logging.debug("Only one gguf file in the repo, loading %s", remote_filename)
        else:
            if quantization is None:
                logging.error(
                    "Must provide quantization type if you want to load from a repo with multiple quants!"
                )
                raise ValueError(
                    "Must provide quantization type if you want to load from a repo with multiple quants!"
                )
            # Find the right quantization file types
            matching_ggufs = list(
                filter(
                    lambda x: re.search(f".*{quantization.lower()}.*", x.lower()),
                    gguf_files,
                )
            )
            if len(matching_ggufs) == 0:
                logging.error(
                    "Quantization %s was not found in repo %s. Can't load the model!",
                    quantization,
                    model_name,
                )
                raise ValueError(
                    f"Quantization {quantization} was not found in repo {model_name}. Can't load the model!"
                )
            # If there's more than one matching the quantization, we have a split file
            elif len(matching_ggufs) > 1:
                logging.debug(
                    "Right quantisation found, looks like a sharded file. Downloading shards..."
                )
                local_filenames = download_split_file(model_name, matching_ggufs)
                ## Giving the first split as local path should work
                model_path = list(filter(lambda x: "01-of" in x, local_filenames))[0]
            else:
                # Only one match, load directly
                remote_filepath = Path(matching_ggufs[0])
                if remote_filepath.is_dir():
                    shard_files = list(sorted(remote_filepath.glob("*.gguf")))
                    remote_filename = remote_filepath / shard_files[0]

                logging.debug(
                    "Found the right quantisation, loading %s", remote_filepath
                )
                model_path = hf_hub_download(
                    repo_id=model_name, filename=remote_filepath.name
                )
    else:
        logging.error("Local model file does not exist, and is not a huggingface repo!")
        raise FileNotFoundError(
            "Local model file does not exist, and is not a huggingface repo!"
        )

    ## Callers that don't set sampling parameters fall back to get_sampling_params' defaults
    run_config_options = run_config_options or {}
    sampling_params = get_sampling_params(run_config_options)

    ## Only `sampling_params` and real llama_cpp.Llama arguments have any effect here.
    ## LlamaCpp forwards anything it doesn't recognise to llama_cpp.Llama, whose __init__
    ## ends in `**kwargs, # type: ignore` and silently discards them - which is how
    ## `temperature`, `flash_attention` (the real spelling is `flash_attn`),
    ## `dry_multiplier` and `samplers` sat here doing nothing. Temperature is applied by
    ## guidance at each gen() site instead, via reasoning_block.
    model = LlamaCpp(
        model=model_path,
        echo=False,
        n_gpu_layers=-1,
        n_ctx=context_length,
        flash_attn=True,
        chat_template=get_chat_template(chat_template),
        seed=-1,
        sampling_params=sampling_params,
    )

    return model
