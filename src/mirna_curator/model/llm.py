from guidance.models import LlamaCpp
from guidance.chat import (
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
}

from huggingface_hub import HfFileSystem, hf_hub_download
from pathlib import Path
import re
import logging

from mirna_curator.utils.sampling import get_sampling_params


logger = logging.getLogger(__name__)


STOP_TOKENS = ["<|end|>", "<|eot_id|>", "<|eom_id|>", "</think>", "<|im_end|>", "<|endoftext|>", '}']


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
    run_config_options: dict = None,
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
    
    sampling_params = get_sampling_params(run_config_options)

    model = LlamaCpp(
        model=model_path,
        echo=False,
        n_gpu_layers=-1,
        n_ctx=context_length,
        flash_attention=True,
        temperature=0.6,
        chat_template=TEMPLATE_LOOKUP.get(chat_template, ChatMLTemplate),
        seed=-1,
        dry_multiplier=run_config_options.get("dry_multiplier", 1.0),
        samplers="top_k;top_p;min_p;temperature;dry;typ_p;xtc",
        sampling_params=sampling_params
    )

    return model
