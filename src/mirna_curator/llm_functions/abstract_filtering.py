from guidance import gen, select, system, user, assistant
import guidance
from mirna_curator.model.prompts import system_prompt_general

import logging

logger = logging.getLogger(__name__)


def assess_abstract(llm: guidance.models.LlamaCpp, abstract: str) -> bool:
    """
    Use the LLM to reason about an abstract and, based on the GO
    curation criteria, forward it for inclusion in the processing,
    or do not curate it.

    Articles that should be curated contain things like:
        - Functional Analyses
        - Luciferase assay
        - qtPCR (quantitative real-time PCR)



    """

    curation_decision_prompt = (
        "Identify research papers where the abstract contains ALL of the following elements:\n"
        "- Specific mention of one or more microRNAs with their exact names (e.g., miR-X, miR-Y)\n"
        "- Explicit identification of protein targets or genes\n"
        "- Description of experimental validation methods (such as luciferase assays, western blots, or reporter assays)\n"
        "- Clear statements about changes in expression levels or protein abundance\n"
        "- Indication of direct targeting (terms like 'directly targets', 'binds to', 'targets the 3`UTR')\n"
        "The abstract should suggest that the paper contains experimental evidence of miRNA-mediated regulation of protein expression, rather than just computational predictions or correlative studies. Prioritize abstracts that describe mechanistic details of how the miRNA affects its target(s).Exclude papers that:\n\n"
        "- Only describe computational predictions without experimental validation\n"
        "- Focus solely on expression profiles without mechanistic insights\n"
        "- Only show correlations without functional validation\n"
        "- Exclusively discuss therapeutic applications without mechanistic details\n"
        "Consider the abstract below:\n\n"
        "Abstract:\n{abstract}\n"
        "Step by step, consider how this abstract matches the above criteria\n"
    )

    try:
        with system():
            llm += system_prompt_general
    except Exception:
        logger.warning(
            "This model does not support a system prompt, forwarding as user instead..."
        )
        with user():
            llm += system_prompt_general

    with user():
        llm += curation_decision_prompt.format(abstract=abstract)

    with assistant():
        llm += "Reasoning: " + gen("reasoning", max_tokens=512)

    with user():
        llm += "Therefore, would you reccomend this abstract be forwarded to a curator for further investigation? Answer yes or no\n"

    with assistant():
        llm += select(["yes", "no"], name="decision")

    print(llm)
    return llm["decision"] == "yes"
