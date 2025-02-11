import typing as ty
from dataclasses import dataclass
from guidance.models._model import Model
from guidance import user, assistant, select, gen
from epmc_xml.article import Article
from mirna_curator.utils.tracing import curation_tracer

from mirna_curator.flowchart.curation import NodeType, CurationFlowchart
from mirna_curator.flowchart.flow_prompts import CurationPrompts

from mirna_curator.llm_functions.conditions import (
    prompted_flowchart_step_bool,
    prompted_flowchart_terminal,
)
from mirna_curator.model.llm import STOP_TOKENS
from time import time


def find_section_heading(llm, target, possibles):
    """
    Finds the most likely section heading given the ones found in the paper.

    This is not a guidance function, so the state of the LLM is not modified.
    I think that means we can clear/reset the LLM with no ill effects outside this function



    """
    llm.reset()
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
            llm += "\n think about it briefly, then make a selection.\n"
            llm += (
                "In your reasoning:\n"
                "- Skip obvious steps\n"
                "- Structure as 'If A then B because C'\n"
                "- Maximum 10 words per step\n"
                "- Use symbols (→, =, ≠, etc.) instead of words if appropriate\n"
                "- Abbreviate common terms (prob/probability, calc/calculate)\n"
                "Your response should be clear but minimal. Show key logical steps only.\n"
            )
        with assistant():
            llm += (
                f"The section heading {target} implies "
                + gen("reasoning", max_tokens=512, stop=STOP_TOKENS)
                + " therefore the most likely section heading is: "
            )
            llm += select(possibles, name="target_section_name")
        target_section_name = llm["target_section_name"]
        # curation_tracer.log_event(
        #     "flowchart_section_choice",
        #     step="choose_section",
        #     evidence="",
        #     result=target_section_name,
        #     reasoning=llm["reasoning"],
        #     loaded_sections=[],
        #     timestamp=time(),
        # )
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
    node_type: ty.Literal["internal", "terminal"]
    name: str


class ComputationGraph:
    def __init__(self, flowchart: CurationFlowchart):
        self.construct_nodes(flowchart)
        self.loaded_sections = []

    def construct_nodes(self, flowchart: CurationFlowchart) -> None:
        """
        Constructs the nodes we will use, but does not link anything together yet
        """
        self._nodes = {}
        # first pass, construct the nodes without transitions
        for flow_node_name, flow_node_props in flowchart.nodes.items():
            if flow_node_props.type == NodeType("decision"):
                function = prompted_flowchart_step_bool
                prompt = flow_node_props.data.condition
                node_type = "internal"
            elif flow_node_props.type == NodeType("terminal"):
                function = prompted_flowchart_terminal
                prompt = flow_node_props.data.terminal
                node_type = "terminal"
            ## Initialise node with empty transitions dict
            this_node = ComputationNode(
                function=function,
                name=flow_node_name,
                node_type=node_type,
                transitions={},
                prompt_name=prompt,
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

    def execute_graph(
        self,
        paper_id: str,
        llm: Model,
        article: Article,
        rna_id: str,
        prompts: CurationPrompts,
    ):
        curation_tracer.set_paper_id(paper_id)
        graph_node = self._nodes[self.start_node.name]

        curation_tracer.log_event(
            "flowchart_init",
            step="starup_timestamp",
            evidence="",
            result="",
            reasoning="",
            loaded_sections=[],
            timestamp=time(),
        )
        node_idx = 0
        visited_nodes = []
        visit_results = []
        while graph_node.node_type == "internal":
            prompt = list(
                filter(lambda p: p.name == graph_node.prompt_name, prompts.prompts)
            )[0]
            visited_nodes.append(graph_node.name)

            ## see if we already have the target section loaded - this should speed things up provided we can reuse the context
            if not prompt.target_section in self.loaded_sections:
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
            try:
                ## Now we load a section to the context only once, we have to get the node result here.
                if target_section_name in self.loaded_sections:
                    llm += graph_node.function(
                        article.sections[target_section_name],
                        False,
                        prompt.prompt,
                        rna_id,
                    )
                else:
                    llm += graph_node.function(
                        article.sections[target_section_name],
                        True,
                        prompt.prompt,
                        rna_id,
                    )
                    self.loaded_sections.append(target_section_name)
            except Exception as e:
                print(e)
                print(llm)
                exit()

            node_result = llm["answer"].lower().replace("*", "") == "yes"
            node_evidence = llm["evidence"]
            node_reasoning = llm["reasoning"]

            curation_tracer.log_event(
                "flowchart_internal",
                step=graph_node.name,
                evidence=node_evidence,
                result=llm["answer"].lower().replace("*", ""),
                reasoning=node_reasoning,
                loaded_sections=self.loaded_sections,
                timestamp=time(),
            )

            visit_results.append(node_result)

            ## Move to the next node...
            if graph_node.transitions.get(node_result, None) is not None:
                graph_node = graph_node.transitions[node_result]
            else:
                annotation = None
                break
            if graph_node.node_type == "terminal":
                aes = {}
                visited_nodes.append(graph_node.name)
                prompt = list(
                    filter(lambda p: p.name == graph_node.prompt_name, prompts.prompts)
                )[0]
                if prompt.name == "no_annotation":
                    annotation = None
                    break
                else:
                    annotation = prompt.annotation

                detector = list(
                    filter(lambda d: d.name == prompt.detector, prompts.detectors)
                )[0]
                ## Now we load a section to the context only once, we have to get the node result here.
                if target_section_name in self.loaded_sections:
                    llm += graph_node.function(
                        article.sections[target_section_name],
                        False,
                        detector.prompt,
                        rna_id,
                        paper_id,
                    )
                else:
                    llm += graph_node.function(
                        article.sections[target_section_name],
                        True,
                        detector.prompt,
                        rna_id,
                        paper_id,
                    )
                    self.loaded_sections.append(target_section_name)
                aes[detector.name] = llm["protein_name"].strip()
                target_name = llm["protein_name"].strip()
                node_reasoning = llm["detector_reasoning"]
                node_evidence = llm["evidence"]
                curation_tracer.log_event(
                    "flowchart_terminal",
                    step=graph_node.name,
                    evidence=node_evidence,
                    result=target_name,
                    reasoning=node_reasoning,
                    loaded_sections=self.loaded_sections,
                    timestamp=time(),
                )

            node_idx += 1

        all_nodes = list(self._nodes.keys())
        result = {n: None for n in all_nodes}
        result.update({f"{n}_result": None for n in all_nodes})
        for visited, visit_result in zip(visited_nodes, visit_results):
            result[visited] = True
            result[f"{visited}_result"] = visit_result
        result.update({"annotation": annotation, "aes": aes})
        trace = str(llm)
        self.loaded_sections = []
        return trace, result
