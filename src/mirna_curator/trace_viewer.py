from flask import Flask, render_template_string, request
import json
import click

app = Flask(__name__)

# Store filename as a global variable that can be set via CLI
TRACE_FILENAME = "traces.ndjson"


def load_traces():
    traces = []
    with open(TRACE_FILENAME, "r") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))
    return traces


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Trace Viewer</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            max-width: 1200px; 
            margin: 0 auto;
            padding: 20px;
        }
        .navigation { 
            margin: 20px 0; 
        }
        .trace-content {
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 5px;
            white-space: pre-wrap;
        }
        .trace-content > div {
            margin: 0;
            padding: 1px 0;
            line-height: 1;
        }
        .key { 
            color: #666;
            font-weight: bold;
        }
        .value { 
            margin-left: 10px;
        }
        .nav-button {
            padding: 5px 15px;
            margin: 0 5px;
            background-color: #4CAF50;
            color: white;
            text-decoration: none;
            border-radius: 3px;
        }
        .nav-button:hover {
            background-color: #45a049;
        }
        .counter {
            margin: 0 15px;
        }
        .filter-section {
            margin-bottom: 20px;
            display: flex;
            gap: 10px;
            align-items: center;
        }
        select {
            padding: 5px;
            border-radius: 3px;
            border: 1px solid #ddd;
        }
        .filter-label {
            font-weight: bold;
            color: #666;
            margin-right: 5px;
        }
    </style>
</head>
<body>
    <h1>Trace Viewer</h1>
    
    <div class="filter-section">
        <form method="get" style="display: flex; gap: 15px; align-items: center;">
            <div>
                <span class="filter-label">Run ID:</span>
                <select name="run_id" onchange="this.form.submit()">
                    <option value="">All Run IDs</option>
                    {% for rid in run_ids %}
                    <option value="{{ rid }}" {% if rid == selected_run_id %}selected{% endif %}>
                        {{ rid }}
                    </option>
                    {% endfor %}
                </select>
            </div>

            <div>
                <span class="filter-label">Paper ID:</span>
                <select name="paper_id" onchange="this.form.submit()">
                    <option value="">All Paper IDs</option>
                    {% for pid in paper_ids %}
                    <option value="{{ pid }}" {% if pid == selected_paper_id %}selected{% endif %}>
                        {{ pid }}
                    </option>
                    {% endfor %}
                </select>
            </div>

            <div>
                <span class="filter-label">Step:</span>
                <select name="step" onchange="this.form.submit()">
                    <option value="">All Steps</option>
                    {% for s in steps %}
                    <option value="{{ s }}" {% if s == selected_step %}selected{% endif %}>
                        {{ s }}
                    </option>
                    {% endfor %}
                </select>
            </div>

            <input type="hidden" name="index" value="0">
        </form>
    </div>

    <div class="navigation">
        {% set prev_index = index - 1 if index > 0 else 0 %}
        {% set next_index = index + 1 if index < total_traces - 1 else total_traces - 1 %}
        <a href="/?index={{ prev_index }}{% if selected_run_id %}&run_id={{ selected_run_id }}{% endif %}{% if selected_paper_id %}&paper_id={{ selected_paper_id }}{% endif %}{% if selected_step %}&step={{ selected_step }}{% endif %}" class="nav-button">Previous</a>
        <span class="counter">Trace {{ index + 1 }} of {{ total_traces }}</span>
        <a href="/?index={{ next_index }}{% if selected_run_id %}&run_id={{ selected_run_id }}{% endif %}{% if selected_paper_id %}&paper_id={{ selected_paper_id }}{% endif %}{% if selected_step %}&step={{ selected_step }}{% endif %}" class="nav-button">Next</a>
    </div>
    <div class="trace-content">
        {% for key, value in trace.items() %}
        <div>
            <span class="key">{{ key }}:</span>
            <span class="value">{{ value }}</span>
        </div>
        {% endfor %}
    </div>
</body>
</html>
"""


@app.route("/")
def show_trace():
    all_traces = load_traces()

    # Get filter values from request
    selected_run_id = request.args.get("run_id", "")
    selected_paper_id = request.args.get("paper_id", "")
    selected_step = request.args.get("step", "")

    # Filter traces based on all selected filters
    traces = all_traces
    if selected_run_id:
        traces = [t for t in traces if t.get("run_id") == selected_run_id]
    if selected_paper_id:
        traces = [t for t in traces if t.get("paper_id") == selected_paper_id]
    if selected_step:
        traces = [t for t in traces if t.get("step") == selected_step]

    # Get all unique values for dropdowns
    run_ids = sorted(set(t.get("run_id") for t in all_traces if t.get("run_id")))
    paper_ids = sorted(set(t.get("paper_id") for t in all_traces if t.get("paper_id")))
    steps = sorted(set(t.get("step") for t in all_traces if t.get("step")))

    index = int(request.args.get("index", 0))

    # Ensure index is within bounds
    if index < 0:
        index = 0
    if index >= len(traces):
        index = len(traces) - 1

    return render_template_string(
        HTML_TEMPLATE,
        trace=traces[index] if traces else {},
        index=index,
        total_traces=len(traces),
        run_ids=run_ids,
        paper_ids=paper_ids,
        steps=steps,
        selected_run_id=selected_run_id,
        selected_paper_id=selected_paper_id,
        selected_step=selected_step,
    )


@click.command()
@click.option(
    "--filename", "-f", default="traces.ndjson", help="Path to the NDJSON trace file"
)
@click.option("--port", "-p", default=5000, help="Port to run the Flask server on")
@click.option(
    "--host", "-h", default="127.0.0.1", help="Host to run the Flask server on"
)
def run_app(filename, port, host):
    """Run the Trace Viewer application with the specified trace file."""
    global TRACE_FILENAME
    TRACE_FILENAME = filename
    click.echo(f"Starting Trace Viewer with file: {filename}")
    app.run(debug=True, host=host, port=port)


if __name__ == "__main__":
    run_app()
