import typing as ty
from dataclasses import dataclass
from guidance.models._base._model import Model
from guidance import user, assistant, select, gen, with_temperature
import guidance
from epmc_xml.article import Article
from mirna_curator.utils.tracing import curation_tracer

from mirna_curator.flowchart.curation import NodeType, CurationFlowchart
from mirna_curator.flowchart.flow_prompts import CurationPrompts

from mirna_curator.llm_functions.conditions import (
    prompted_flowchart_step_bool,
    prompted_flowchart_terminal,
    prompted_flowchart_step_tool,
)
from mirna_curator.llm_functions.filtering import prompted_filter
from mirna_curator.model.llm import STOP_TOKENS
from time import time
from functools import partial
import logging

logger = logging.getLogger(__name__)


def find_section_heading(llm, target, possibles):
    """
    Finds the most likely section heading given the ones found in the paper.

    This is not a guidance function, so the state of the LLM is not modified.
    I think that means we can clear/reset the LLM with no ill effects outside this function



    """
    # print(dir(llm))
    # llm.reset()
    try:
        augmentations = {
            "methods": (
                "Bear in mind this section is likely to contain details on the experimental "
                "techniques used."
            ),
            "results": (
                "Bear in mind this section is likely to contain the results of the experiments, "
                "but may also contain the discussion of those results."
            ),
        }
        with user():
            llm += (
                f"We are looking for the closest section heading to '{target}' from "
                f"the following possbilities: {','.join(possibles)}. "
                "Which of the available headings most likely to contain the information "
                f"we would expect from a section titled '{target}'? "
                f"{augmentations.get(target, '')}"
            )
            llm += "\nThink about it briefly, then make a selection.\n"
        with assistant():
            llm += (
                f"The section heading {target} implies "
                + with_temperature(
                    gen("reasoning", max_tokens=512, stop=STOP_TOKENS), 0.6
                )
                + " therefore the most likely section heading is: "
            )
            llm += select(possibles, name="target_section_name")
        target_section_name = llm["target_section_name"]
        curation_tracer.log_event(
            "flowchart_section_choice",
            step="choose_section",
            evidence="",
            result=target_section_name,
            reasoning=llm["reasoning"],
            loaded_sections=[],
            timestamp=time(),
        )
    except Exception as e:
        print(e)
        print(llm)
        exit()
    return target_section_name


@dataclass
class ComputationNode:
    function: ty.Callable
    transitions: ty.Dict[ty.Any, "ComputationNode"]
    prompt_name: ty.Optional[str]
    node_type: ty.Literal["filter", "internal", "terminal"]
    tools: ty.Optional[ty.List[str]]
    name: str


class ComputationGraph:
    def __init__(self, flowchart: CurationFlowchart, run_config: ty.Dict = None):
        self.construct_nodes(flowchart)
        self.loaded_sections = []
        self.run_config = run_config
        self.current_node = None

    def construct_nodes(self, flowchart: CurationFlowchart) -> None:
        """
        Constructs the nodes we will use, but does not link anything together yet
        """
        self._nodes = {}
        # first pass, construct the nodes without transitions
        for flow_node_name, flow_node_props in flowchart.nodes.items():
            if flow_node_props.type == NodeType("conditional_prompt_boolean"):
                function = prompted_flowchart_step_bool
                prompt = flow_node_props.data.prompt_name
                node_type = "internal"
            elif flow_node_props.type == NodeType("conditional_tool_use"):
                function = partial(
                    prompted_flowchart_step_tool, tools=flow_node_props.data.tools
                )
                prompt = flow_node_props.data.prompt_name
                node_type = "internal"
            elif flow_node_props.type == NodeType("terminal_full"):
                function = prompted_flowchart_terminal
                prompt = flow_node_props.data.terminal_name
                node_type = "terminal"
            elif flow_node_props.type == NodeType("terminal_short_circuit"):
                function = prompted_flowchart_terminal
                prompt = flow_node_props.data.terminal_name
                node_type = "terminal"
            elif flow_node_props.type == NodeType("terminal_conditional"):
                function = prompted_flowchart_terminal
                prompt = flow_node_props.data.terminal_name
                node_type = "terminal"
            elif flow_node_props.type == NodeType("filter"):
                function = prompted_filter
                node_type = "filter"
                prompt = flow_node_props.data.prompt_name

            ## Initialise node with empty transitions dict
            this_node = ComputationNode(
                function=function,
                name=flow_node_name,
                node_type=node_type,
                transitions={},
                prompt_name=prompt,
                tools=flow_node_props.data.tools,
            )
            self._nodes[flow_node_name] = this_node

        ## Next pass to link the nodes together correctly
        for flow_node_name, flow_node_props in flowchart.nodes.items():
            flow_transition = flow_node_props.transitions
            if flow_transition is not None:
                if flow_transition.true is not None:
                    self._nodes[flow_node_name].transitions[True] = self._nodes[
                        flow_transition.true
                    ]
                if flow_transition.false is not None:
                    self._nodes[flow_node_name].transitions[False] = self._nodes[
                        flow_transition.false
                    ]
                if flow_transition.next is not None:
                    self._nodes[flow_node_name].transitions["next"] = self._nodes[
                        flow_transition.next
                    ]

        self.start_node = self._nodes[flowchart.startNode]

    def infer_target_section_name(self, llm, prompt, article):
        """
        Check the dsection names in the dictionary first, then fall back to asking the LLM
        for help if there's no direct match

        This is used in a few places, so makes sense to factor out
        """
        ## sometimes, the section we want is named differently, so need to use the LLM to figure it out
        if not prompt.target_section in article.sections.keys():
            check_subtitles = [
                prompt.target_section in section_name
                for section_name in article.sections.keys()
            ]
            if not any(check_subtitles):
                target_section_name = find_section_heading(
                    llm, prompt.target_section, list(article.sections.keys())
                )
            else:
                target_section_name = list(article.sections.keys())[
                    check_subtitles.index(True)
                ]
        else:
            target_section_name = prompt.target_section

        return target_section_name

    def run_filters(self, llm, article, prompts, rna_id):
        """
        Run filters in the flowchart

        This will update the current node, and other node recording things, but does not
        update the LLM state. That's what we want for the filtering step, but we do have to
        handle the terminal node correctly so it doesn't lose the annotation from filtering
        """
        while self.current_node.node_type == "filter":
            logger.info(f"Applying filter node {self.current_node.name}")
            self.visited_nodes.append(self.current_node.name)

            ## Have to filter to get the prompt named by the flowchart node
            prompt = list(
                filter(
                    lambda p: p.name == self.current_node.prompt_name, prompts.prompts
                )
            )[0]

            try:
                ## Find and load the relevant article section
                target_section_name = self.infer_target_section_name(
                    llm, prompt, article
                )

                filter_decision, filter_reasoning = self.current_node.function(
                    llm,
                    article.get_section(
                        target_section_name,
                        include_figures=True,
                        figures_placement="end",
                    ),
                    True,
                    prompt.prompt,
                    rna_id,
                    config=self.run_config,
                )

                node_result = filter_decision
                node_evidence = ""
                node_reasoning = filter_reasoning

                curation_tracer.log_event(
                    "flowchart_filter",
                    step=self.current_node.name,
                    evidence=node_evidence,
                    result=filter_decision,
                    reasoning=node_reasoning,
                    loaded_sections=self.loaded_sections,
                    timestamp=time(),
                )

                self.visit_results.append(node_result)
                self.visit_evidences.append(node_evidence)
                self.visit_reasonings.append(node_reasoning)

                ## Only move to next node after updating everything else
                self.current_node = self.current_node.transitions[node_result == "yes"]
                ## It should pretty much always be the case that if we go to a terminal
                ## node from a filter, there will be no annotation. This just allows us to
                ## record the reason as an annotation and skip to the next paper.
                ## Actual handling of the terminal node will be done elsewhere
                if self.current_node.node_type == "terminal":
                    break
                self.node_idx += 1

            ## TODO: this can be improved with some more specific exception handling
            except Exception as e:
                logger.error(f"Hit error: {e} while filtering, aborting")
                logger.error(f"LLM state: {str(llm)}")
                logger.error(filter_decision)
                logger.error(filter_reasoning)
                exit(1)

    @guidance
    def run_nodes(self, llm, article, prompts, rna_id):
        """
        Runs the core logic of the flowchart
        This will just keep advancing the state until it hits a terminal node
        Therefore, it doesn't return anything
        """
        while self.current_node.node_type == "internal":
            print(self.current_node.name)
            ## Have to filter to get the prompt named by the flowchart node
            prompt = list(
                filter(
                    lambda p: p.name == self.current_node.prompt_name, prompts.prompts
                )
            )[0]

            self.visited_nodes.append(self.current_node.name)
            logger.info(f"Processing node {self.current_node.name}")

            ## see if we already have the target section loaded - this should speed things up provided we can reuse the context
            if not prompt.target_section in self.loaded_sections:
                logger.info(f"Loading section {prompt.target_section} into context")
                ## sometimes, the section we want is named differently, so need to use the LLM to figure it out
                target_section_name = self.infer_target_section_name(
                    llm, prompt, article
                )

            try:
                ## Now we load a section to the context only once, we have to get the node result here.
                if target_section_name in self.loaded_sections:
                    logger.info("Running condition function, not loading context")
                    llm += self.current_node.function(
                        article.get_section(
                            target_section_name,
                            include_figures=True,
                            figures_placement="end",
                        ),
                        False,
                        prompt.prompt,
                        rna_id,
                        config=self.run_config,
                    )
                else:
                    logger.info("Running condition function, loading context")
                    llm += self.current_node.function(
                        article.get_section(
                            target_section_name,
                            include_figures=True,
                            figures_placement="end",
                        ),
                        True,
                        prompt.prompt,
                        rna_id,
                        config=self.run_config,
                    )
                    self.loaded_sections.append(target_section_name)

            ## TODO: improve specificity of exception handling here
            except Exception as e:
                logger.error("Hit an exception when trying to run conditions")
                logger.error(f"Exception: {e}")
                error_count += 1
                if error_count > 3:
                    print("Too many errors, exiting")
                    logger.fatal("Too many errors, exiting")
                    exit()
                continue

            node_result = llm["answer"].lower().replace("*", "") == "yes"
            node_evidence = llm["evidence"]
            node_reasoning = llm["reasoning"]

            curation_tracer.log_event(
                "flowchart_internal",
                step=self.current_node.name,
                evidence=node_evidence,
                result=llm["answer"].lower().replace("*", ""),
                reasoning=node_reasoning,
                loaded_sections=self.loaded_sections,
                timestamp=time(),
            )

            self.visit_results.append(node_result)
            self.visit_evidences.append(node_evidence)
            self.visit_reasonings.append(node_reasoning)

            ## Move to the next node...
            if self.current_node.transitions.get(node_result, None) is not None:
                self.current_node = self.current_node.transitions[node_result]
            else:
                annotation = None
                aes = None
                break
            self.node_idx += 1
            ## Terminal node handling -
            if self.current_node.node_type == "terminal":
                logger.info(f"Hit terminal node {self.current_node.name}")
                break
        return llm

    def terminal_node_check(self, llm, article, prompts, rna_id, paper_id):
        """
        This checks if we are on a terminal node, and if we are it figures out what to do. This is
        either:
            - Short circuit for no annotation by recording the annotation note and moving on, or
            - Running the detector prompt to get the aes for the annotation, and recording the result

        The LLM will have the needed sections in context (probably) if we have got to a terminal node
        as a result of sucessful curation. If we're short circuiting after filtering, then it doesn't
        matter.
        """
        aes = None  ## Default is no extensions
        annotation = None
        if self.current_node.node_type == "terminal":
            self.visited_nodes.append(self.current_node.name)
            if self.current_node.prompt_name is None:
                prompt = None
            else:
                prompt = list(
                    filter(
                        lambda p: p.name == self.current_node.prompt_name,
                        prompts.prompts,
                    )
                )[0]
            if prompt is None:
                annotation = None
                node_reasoning = ""
                node_evidence = ""
                target_name = ""
            elif prompt.name == "no_annotation":
                annotation = prompt.annotation
                node_reasoning = ""
                node_evidence = ""
                target_name = ""

            else:
                ## Only lookup the target section name if we are actually going to use it
                target_section_name = self.infer_target_section_name(
                    llm, prompt, article
                )

                annotation = prompt.annotation
                detector = list(
                    filter(lambda d: d.name == prompt.detector, prompts.detectors)
                )[0]
                ## Now we load a section to the context only once, we have to get the node result here.
                if target_section_name in self.loaded_sections:
                    llm += self.current_node.function(
                        article.sections[target_section_name],
                        False,
                        detector.prompt,
                        rna_id,
                        paper_id,
                        config=self.run_config,
                    )
                else:
                    llm += self.current_node.function(
                        article.sections[target_section_name],
                        True,
                        detector.prompt,
                        rna_id,
                        paper_id,
                        config=self.run_config,
                    )
                    self.loaded_sections.append(target_section_name)

                ## extract results from the LLM
                aes = {detector.name : llm["protein_name"].strip()}
                target_name = llm["protein_name"].strip()
                node_reasoning = llm["detector_reasoning"]
                node_evidence = llm["evidence"]

            self.visit_results.append(target_name)
            self.visit_evidences.append(node_evidence)
            self.visit_reasonings.append(node_reasoning)

            curation_tracer.log_event(
                "flowchart_terminal",
                step=self.current_node.name,
                evidence=node_evidence,
                result=target_name,
                reasoning=node_reasoning,
                loaded_sections=self.loaded_sections,
                timestamp=time(),
            )
        self.node_idx += 1
        ## These will only have something in if the node was a terminal
        return annotation, aes

    def execute_graph(
        self,
        paper_id: str,
        llm: Model,
        article: Article,
        rna_id: str,
        prompts: CurationPrompts,
    ):
        curation_tracer.set_paper_id(paper_id)
        self.current_node = self._nodes[self.start_node.name]

        curation_tracer.log_event(
            "flowchart_init",
            step="startup_timestamp",
            evidence="",
            result="",
            reasoning="",
            loaded_sections=[],
            timestamp=time(),
        )
        ## These still need to be reset within a run, so keep this
        self.node_idx = 0
        self.visited_nodes = []
        self.visit_results = []
        self.visit_evidences = []
        self.visit_reasonings = []
        self.error_count = 0

        self.run_filters(llm, article, prompts, rna_id)

        annotation, aes = self.terminal_node_check(
            llm, article, prompts, rna_id, paper_id
        )
        if annotation is None and self.current_node.node_type != "terminal":
            ## means the filtering steps did not end on a terminal node, so continue curation
            llm += self.run_nodes(article, prompts, rna_id)
            ## Once this is done, we should have hit a terminal node, so we can update the annotation and aes
            annotation, aes = self.terminal_node_check(
                llm, article, prompts, rna_id, paper_id
            )

        curation_tracer.log_event(
            "flowchart_end",
            setp="finish_timestamp",
            evidence="",
            result="",
            reasoning="",
            loaded_sections=[],
            timestamp=time(),
        )
        all_nodes = list(self._nodes.keys())
        result = {n: None for n in all_nodes}
        result.update({f"{n}_result": None for n in all_nodes})
        for visited, visit_result, visit_evidence, visit_reasoning in zip(
            self.visited_nodes,
            self.visit_results,
            self.visit_evidences,
            self.visit_reasonings,
        ):
            result[visited] = True
            result[f"{visited}_result"] = visit_result
            result[f"{visited}_evidence"] = visit_evidence
            result[f"{visited}_reasoning"] = visit_reasoning
        result.update({"annotation": annotation, "aes": aes})
        trace = str(llm)
        self.loaded_sections = []
        return trace, result
