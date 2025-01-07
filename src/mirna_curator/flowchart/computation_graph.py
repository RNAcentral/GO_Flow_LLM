import typing as ty
from dataclasses import dataclass
from guidance.models._model import Model
from guidance import user, assistant, select
from epmc_xml.article import Article

# from pydantic import BaseModel
from mirna_curator.flowchart.curation import NodeType, CurationFlowchart
from mirna_curator.flowchart.flow_prompts import CurationPrompts

from mirna_curator.llm_functions.conditions import (
    prompted_flowchart_step_bool,
    prompted_flowchart_terminal,
)

def find_section_heading(llm, target, possibles):
    with user():
        llm += (
            f"We are looking for the closest section heading to '{target}' from "
            f"the following possbilities: {','.join(possibles)}. "
            "Which of the available headings most likely to contain the information "
            "we would expect from a section titled '{target}'?"
        )
    with assistant():
        llm += select(
            possibles, name="target_section_name"
        )
    target_section_name = llm["target_section_name"]
    print(llm)
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

    def execute_graph(self, llm: Model, article: Article, rna_id: str, prompts: CurationPrompts):
        graph_node = self._nodes[self.start_node.name]

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
                        target_section_name = find_section_heading(llm, prompt.target_section, list(article.sections.keys()))
                    else:
                        target_section_name = list(article.sections.keys())[
                            check_subtitles.index(True)
                        ]
                else:
                    target_section_name = prompt.target_section

            ## Now we load a section to the context only once, we have to get the node result here.
            if target_section_name in self.loaded_sections:
                llm += graph_node.function(
                    "" , prompt.prompt, rna_id
                )    
            else:
                llm += graph_node.function(
                    article.sections[target_section_name], prompt.prompt, rna_id
                )
                self.loaded_sections.append(target_section_name)
            node_result = llm['answer'].lower() == "yes"
            visit_results.append(node_result)

            
            ## Move to the next node...
            graph_node = graph_node.transitions[node_result]
            if graph_node.node_type == "terminal":
                aes = {}
                visited_nodes.append(graph_node.name)
                prompt = list(
                filter(lambda p: p.name == graph_node.prompt_name, prompts.prompts))[0]
                if prompt.name == "no_annotation":
                    annotation = None
                    break
                else:
                    annotation = prompt.annotation
                

                detector = list(filter(lambda d: d.name == prompt.detector, prompts.detectors))[0]
                ## Now we load a section to the context only once, we have to get the node result here.
                if target_section_name in self.loaded_sections:
                    llm += graph_node.function(
                        "" , detector.prompt, rna_id
                    )    
                else:
                    llm += graph_node.function(
                        article.sections[target_section_name], detector.prompt, rna_id
                    )
                    self.loaded_sections.append(target_section_name)
                aes[detector.name] = llm["protein_name"].strip()
                


            node_idx += 1

        result = {
            "visited_nodes": visited_nodes,
            "visit_results": visit_results,
            "annotation" : annotation,
            "aes": aes

        }
        trace = str(llm)
        return trace, result