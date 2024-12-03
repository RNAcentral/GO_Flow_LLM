def json_to_mermaid(flow_data):
    """Convert flowchart JSON to Mermaid diagram syntax."""
    mermaid_lines = ["stateDiagram-v2"]

    # Add initial transition to start node
    mermaid_lines.append(f"    [*] --> {flow_data['startNode']}")

    # Process each node
    for node_id, node in flow_data["nodes"].items():
        # Handle transitions based on node type
        if node["type"] == "decision":
            if "transitions" in node:
                if "true" in node["transitions"]:
                    condition = node["data"].get("condition", "true")
                    mermaid_lines.append(
                        f"    {node_id} --> {node['transitions']['true']}: {node['data']['text']} (true)"
                    )
                if "false" in node["transitions"]:
                    mermaid_lines.append(
                        f"    {node_id} --> {node['transitions']['false']}: {node['data']['text']} (false)"
                    )

        elif node["type"] == "action":
            if "transitions" in node and "next" in node["transitions"]:
                mermaid_lines.append(
                    f"    {node_id} --> {node['transitions']['next']}: next"
                )

        elif node["type"] == "terminal":
            mermaid_lines.append(f"    {node_id} --> [*]")

    return "\n".join(mermaid_lines)


# Example usage
if __name__ == "__main__":
    example_flow = {
        "nodes": {
            "start": {
                "type": "decision",
                "data": {"text": "Is user logged in?", "condition": "user.isLoggedIn"},
                "transitions": {"true": "dashboard", "false": "login"},
            },
            "login": {
                "type": "action",
                "data": {"text": "Show login page"},
                "transitions": {"next": "start"},
            },
            "dashboard": {"type": "terminal", "data": {"text": "Display dashboard"}},
        },
        "startNode": "start",
    }

    mermaid_output = json_to_mermaid(example_flow)
    print(mermaid_output)
