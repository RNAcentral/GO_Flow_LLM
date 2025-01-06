from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class Prompt(BaseModel):
    name: str
    type: Literal[
        "condition_prompt_boolean",
        "terminal_short_circuit",
        "terminal_full",
        "terminal_bp_only",
        "terminal_conditional",
    ]
    prompt: str = ""
    target_section: Optional[str] = None


class Detector(BaseModel):
    name: str
    type: Literal["AE"]
    prompt: str = ""


class CurationPrompts(BaseModel):
    prompts: List[Prompt]
    detector: Detector
