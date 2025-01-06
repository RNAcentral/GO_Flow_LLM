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
                prompt = None
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
                        with user():
                            llm += (
                                f"We are looking for the closest section heading to {prompt.target_section} from "
                                f"the following possbilities: {article.sections.keys()}. Which one is closest?"
                            )
                        with assistant():
                            llm += select(
                                article.sections.keys(), name="target_section_name"
                            )
                        target_section_name = llm["target_section_name"]
                    else:
                        target_section_name = list(article.sections.keys())[
                            check_subtitles.index(True)
                        ]

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
            node_result = llm['answer'] == "yes"
            visit_results.append(node_result)

            
            ## Move to the next node...
            graph_node = graph_node.transitions[node_result]
            if graph_node.node_type == "terminal":
                visited_nodes.append(graph_node.name)
            node_idx += 1

        print(visited_nodes)
        print(visit_results)
        pass
