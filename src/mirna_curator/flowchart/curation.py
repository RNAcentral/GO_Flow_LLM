schema = """
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Flowchart Schema",
  "type": "object",
  "required": ["nodes", "startNode"],
  "properties": {
    "nodes": {
      "type": "object",
      "patternProperties": {
        "^[a-zA-Z0-9_]+$": {
          "type": "object",
          "required": ["type", "data"],
          "properties": {
            "type": {
              "type": "string",
              "enum": ["decision", "terminal"]
            },
            "data": {
              "type": "object",
              "properties": {
                "desc": {
                  "type": "string"
                },
                "condition": {
                  "type": "string"
                }
              },
              "required": ["condition"]
            },
            "transitions": {
              "type": "object",
              "properties": {
                "true": {
                  "type": "string",
                  "pattern": "^[a-zA-Z0-9_]+$"
                },
                "false": {
                  "type": "string",
                  "pattern": "^[a-zA-Z0-9_]+$"
                },
                "next": {
                  "type": "string",
                  "pattern": "^[a-zA-Z0-9_]+$"
                }
              }
            }
          }
        }
      }
    },
    "startNode": {
      "type": "string",
      "pattern": "^[a-zA-Z0-9_]+$"
    }
  }
}
"""
from enum import Enum
from typing import Dict, Optional, List
from pydantic import BaseModel, Field, constr


class NodeType(str, Enum):
    decision = "conditional_prompt_boolean"  ## These are where we make decision
    decision_tool = "conditional_tool_use"  ## Nodes that use a tool
    terminal = "terminal_full"  ## These are where we make classifications
    terminal_short_circuit = "terminal_short_circuit"  ## No annotation basically
    terminal_conditional = "terminal_conditional"  ## These make classifications, but with some extra conditions


class NodeData(BaseModel):
    desc: Optional[str] = None  ## Text description of the node
    prompt_name: Optional[str] = None  ## Prompt to lookup for conditions
    terminal_name: Optional[str] = None  ## Terminal to look up for final classification
    tools: Optional[List[str]] = None  ## List of tool names to be available in the node


class NodeTransitions(BaseModel):
    true: Optional[constr(pattern=r"^[a-zA-Z0-9_]+$")] = None
    false: Optional[constr(pattern=r"^[a-zA-Z0-9_]+$")] = None
    next: Optional[constr(pattern=r"^[a-zA-Z0-9_]+$")] = None


class Node(BaseModel):
    type: NodeType
    data: NodeData
    transitions: Optional[NodeTransitions] = None


class CurationFlowchart(BaseModel):
    nodes: Dict[str, Node]
    startNode: constr(pattern=r"^[a-zA-Z0-9_]+$")
