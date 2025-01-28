from mirna_curator.flowchart import flow_prompts 
from flask import Flask, render_template_string, request, jsonify
from pydantic import ValidationError
import click


prompt_dict = {}
prompt_lookup = {}

from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Text Editor</title>
    <style>
        textarea { width: 80%; height: 300px; margin: 20px; }
        button, select { margin: 20px; }
    </style>
</head>
<body>
    <form method="GET">
        <select name="option" onchange="this.form.submit()">
            {% for option in options %}
                <option value="{{ option }}" {% if option == selected %}selected{% endif %}>
                    {{ option }}
                </option>
            {% endfor %}
        </select>
    </form>
    <br>
    <textarea id="editor">{{ content }}</textarea>
    <br>
    <button onclick="saveContent()">Save</button>

    <script>
        function saveContent() {
            const content = document.getElementById('editor').value;
            fetch('/save', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    option: document.querySelector('select').value,
                    content: content
                })
            })
            .then(response => response.json())
            .then(data => alert('Saved successfully!'));
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    selected = request.args.get('option', list(prompt_dict.keys())[0])
    return render_template_string(HTML_TEMPLATE, 
        content=prompt_dict[selected],
        options=prompt_dict.keys(),
        selected=selected)

@app.route('/save', methods=['POST'])
def save():
    global prompt_dict
    global prompt_data
    global prompt_filename

    data = request.json
    prompt_dict[data['option']] = data['content']

    
    try:
        prompt_to_update = prompt_data.prompts.index(prompt_lookup[data['option']])
        prompt_data.prompts[prompt_to_update].prompt = data['content']
    except ValueError:
        prompt_to_update = prompt_data.detectors.index(prompt_lookup[data['option']])
        prompt_data.detectors[prompt_to_update].prompt = data['content']

    with open(prompt_filename, 'w') as out:
        out.write(prompt_data.model_dump_json(indent=2))
    return jsonify({"status": "success"})

def set_content_dict(new_dict):
    global prompt_dict
    prompt_dict = new_dict


@click.command()
@click.option('--prompts', type=click.Path(exists=True), help='JSON file with content dictionary')
def run(prompts):
    global content_dict
    global prompt_dict
    global prompt_lookup
    global prompt_filename
    global prompt_data
    prompt_filename = prompts

    try:
        prompt_string = open(prompts, "r").read()
        prompt_data = flow_prompts.CurationPrompts.model_validate_json(prompt_string)
    except ValidationError as e:
        print("Error loading prompts, aborting")
        exit()

    prompt_dict = {p.name:p.prompt for p in prompt_data.prompts}
    prompt_dict.update({d.name:d.prompt for d in prompt_data.detectors})

    prompt_lookup = {p.name:p for p in prompt_data.prompts}
    prompt_lookup.update({d.name:d for d in prompt_data.detectors})
    
    app.run(debug=True)

if __name__ == '__main__':
    run()