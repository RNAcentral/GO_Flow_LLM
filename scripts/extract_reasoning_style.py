"""
Derive a REASONING_STYLES entry, and guidance role markers, from a model's real chat template.

Run this when adding a new model, then paste the output into REASONING_STYLES in
src/mirna_curator/llm_functions/reasoning.py. This is a development tool - nothing at
runtime imports it, and its output is checked in so a mis-parse shows up in review rather
than silently corrupting the chain-of-thought record on a production run.

Why extract at all: guidance never renders a model's jinja chat template. It takes only
static role markers from its ChatTemplate classes and drives the token stream itself, so
llama.cpp's `--jinja` / `--reasoning-format` handling never runs. The jinja template is
still the authoritative description of how a model delimits its thinking, so we read it
here, once, offline.

Usage:
    python scripts/extract_reasoning_style.py --self-check
    python scripts/extract_reasoning_style.py --gguf /path/to/model.gguf
    python scripts/extract_reasoning_style.py --template-name qwen3_gguf_template
"""

import argparse
import re
import sys

from jinja2 import Environment
from jinja2.ext import loopcontrols


PROBE = [{"role": "user", "content": "PROBE_USER"}]
PROBE_WITH_REPLY = PROBE + [{"role": "assistant", "content": "PROBE_ASSISTANT"}]
## Templates that keep reasoning in a named channel (Gemma 4) render it from a dedicated
## message field rather than from an enable_thinking prefill, so they need their own probe.
PROBE_WITH_REASONING = PROBE + [
    {
        "role": "assistant",
        "content": "PROBE_ASSISTANT",
        "reasoning_content": "PROBE_REASONING",
    }
]

## An empty thinking block looks like `<tag>` whitespace `</tag>` whitespace. The tag name
## is back-referenced so open and close must match. The two whitespace runs are captured
## separately because the inner one is the empty reasoning content: its newlines have to be
## split between the open and close markers, or `open` swallows the lot and the reasoning
## gets generated one line further in than the model expects.
EMPTY_BLOCK = re.compile(r"^<([A-Za-z_][\w-]*)>(\s*)</\1>(\s*)$", re.DOTALL)


class TemplateRefused(Exception):
    """The template itself rejected the message sequence, via raise_exception."""


def _make_env():
    """
    A jinja environment matching what transformers/llama.cpp give a chat template.

    `raise_exception` is not a jinja builtin - transformers injects it, and templates that
    enforce strict user/assistant alternation (Gemma 2) call it. Without it those
    templates fail with a confusing UndefinedError instead of their real message.
    """

    def raise_exception(message):
        raise TemplateRefused(message)

    env = Environment(extensions=[loopcontrols])
    env.globals["raise_exception"] = raise_exception
    return env


def render(template, messages, **kwargs):
    """Render a jinja chat template the way transformers/llama.cpp would."""
    env = _make_env()
    return env.from_string(template).render(
        messages=messages, add_generation_prompt=True, **kwargs
    )


def render_no_gen(template, messages, **kwargs):
    """Render without the trailing generation prompt, for isolating role markers."""
    env = _make_env()
    return env.from_string(template).render(
        messages=messages, add_generation_prompt=False, **kwargs
    )


def extract_channel_style(template):
    """
    Recover a reasoning style from a template that renders reasoning into a named channel.

    Gemma 4 works this way: instead of prefilling an empty block when thinking is off, it
    renders `reasoning_content` between channel markers inside the assistant turn, e.g.
    `<|channel>thought\\nREASONING\\n<channel|>ANSWER`. Diffing a render that carries
    reasoning against one that doesn't isolates the markers around it.

    Arguments:
        template: str - the jinja chat template source

    Returns:
        (style, note) as for extract_reasoning_style; style is None if not this shape.
    """
    plain = render(template, PROBE_WITH_REPLY)
    with_reasoning = render(template, PROBE_WITH_REASONING)
    if plain == with_reasoning:
        return None, "template ignores reasoning_content"

    if "PROBE_REASONING" not in with_reasoning:
        return None, "reasoning_content changed the render but does not appear in it"

    ## Everything between the end of the assistant role marker and the reasoning is the
    ## open marker; everything between the reasoning and the answer is the close marker.
    before, after = with_reasoning.split("PROBE_REASONING", 1)
    anchor = before.rfind("PROBE_USER")
    tail_of_anchor = before.index("\n", anchor) + 1 if anchor != -1 else 0
    ## Walk forward to the last role marker before the reasoning starts
    role_end = before.rfind(">")
    open_marker = before[before.rfind("\n", tail_of_anchor, role_end) + 1 :]
    close_marker = after[: after.index("PROBE_ASSISTANT")]

    if not open_marker or not close_marker:
        return None, f"could not isolate channel markers from {with_reasoning!r}"

    ## The channel closer is the stop string - generation must end when the model leaves
    ## the channel, not when it ends the whole turn
    stop = close_marker.strip()
    style = {"open": open_marker, "close": close_marker, "stop": [stop]}
    return style, f"reasoning is rendered into a named channel closed by {stop!r}"


def extract_reasoning_style(template):
    """
    Work out how a chat template delimits the thinking region.

    The trick is asymmetric: it is `enable_thinking=False` that *reveals* the markers,
    because a template that supports thinking responds to it by prefilling an empty
    block (e.g. `<think>\\n\\n</think>\\n\\n`). `enable_thinking=True` just leaves the
    assistant turn open, which tells us nothing.

    Arguments:
        template: str - the jinja chat template source

    Returns:
        (style, note): the REASONING_STYLES dict for this model (or None if it has no
        thinking channel), plus a human-readable explanation of how we decided.
    """
    thinking_on = render(template, PROBE, enable_thinking=True)
    thinking_off = render(template, PROBE, enable_thinking=False)

    if thinking_on == thinking_off:
        ## No prefill, but the model may still keep reasoning in a named channel
        return extract_channel_style(template)

    if not thinking_off.startswith(thinking_on):
        ## enable_thinking changed something other than the assistant prefill - Gemma 4
        ## gates thinking from a *system* turn, for instance. The channel probe reads the
        ## assistant turn directly, so try that before giving up.
        style, note = extract_channel_style(template)
        if style is not None:
            return style, note + " (enable_thinking also alters an earlier turn - see the gate warning)"
        return None, (
            "enable_thinking changed the prompt somewhere other than the end, and "
            "reasoning_content is not rendered into a channel; read this one by hand"
        )

    prefill = thinking_off[len(thinking_on) :]
    match = EMPTY_BLOCK.match(prefill)
    if match is None:
        return None, f"could not parse an empty thinking block from {prefill!r}"

    tag, inner, trailing = match.group(1), match.group(2), match.group(3)
    ## `<think>\n\n</think>\n\n` means content sits on its own line: one newline opens the
    ## region and one closes it. A single inner newline means no closing one.
    lead = "\n" if "\n" in inner else ""
    tail = "\n" if inner.count("\n") > 1 else ""
    style = {
        "open": f"<{tag}>{lead}",
        "close": f"{tail}</{tag}>{trailing}",
        "stop": [f"</{tag}>"],
    }
    return style, f"found an empty <{tag}> block in the enable_thinking=False prefill"


def extract_role_markers(template):
    """
    Recover the get_role_start / get_role_end strings a guidance ChatTemplate needs.

    Only needed when guidance ships no ChatTemplate subclass for a model family. Note
    some models rename the assistant role in the template (Gemma uses `model`), which is
    exactly the sort of thing this surfaces.

    Arguments:
        template: str - the jinja chat template source

    Returns:
        dict of role -> {"start": str, "end": str}
    """
    markers = {}
    for role in ("system", "user", "assistant"):
        ## Render each role in isolation, so nothing from a neighbouring turn can leak
        ## into the markers. add_generation_prompt is off here for the same reason.
        try:
            rendered = render_no_gen(
                template, [{"role": role, "content": "PROBE_CONTENT"}]
            )
        except TemplateRefused:
            ## Some templates (Gemma 2) enforce strict user/assistant alternation, so a
            ## lone assistant turn is invalid. Retry inside a conversation that is, then
            ## subtract the leading user turn.
            rendered = _render_in_context(template, role, markers)
            if rendered is None:
                markers[role] = {"unsupported": "role cannot be rendered in isolation"}
                continue
        except Exception as e:
            markers[role] = {"unsupported": f"{type(e).__name__}: {e}"}
            continue
        if "PROBE_CONTENT" not in rendered:
            markers[role] = {"unsupported": "role produced no output"}
            continue
        start, end = rendered.split("PROBE_CONTENT", 1)
        markers[role] = {"start": start, "end": end}
    return markers


def _render_in_context(template, role, markers):
    """
    Render one role inside a valid user/assistant conversation, then strip the user turn.

    Returns the render with everything before this role's turn removed, so the caller can
    split on the sentinel exactly as it would for an isolated render.
    """
    user = markers.get("user")
    if not user or "start" not in user:
        return None
    try:
        rendered = render_no_gen(
            template,
            [
                {"role": "user", "content": "PROBE_USER"},
                {"role": role, "content": "PROBE_CONTENT"},
            ],
        )
    except Exception:
        return None
    prefix = f"{user['start']}PROBE_USER{user['end']}"
    return rendered[len(prefix) :] if rendered.startswith(prefix) else None


def format_entry(name, style):
    """Render a style as a pasteable REASONING_STYLES line."""
    if style is None:
        return f'    "{name}": REASONING_STYLES["none"],  # no thinking channel'
    return (
        f'    "{name}": {{"open": {style["open"]!r}, '
        f'"close": {style["close"]!r}, "stop": {style["stop"]!r}}},'
    )


## Minimal templates exercising the shapes of empty thinking block seen in the wild.
## The whitespace split between open and close is the fiddly part, so it is pinned here.
SYNTHETIC = {
    ## Qwen3 / Qwen3.5 / Qwen3.6 and the R1 distills: content sits on its own line
    "<think>\n\n</think>\n\n": {
        "open": "<think>\n",
        "close": "\n</think>\n\n",
        "stop": ["</think>"],
    },
    ## No inner whitespace at all - markers must not gain newlines that aren't there
    "<think></think>": {"open": "<think>", "close": "</think>", "stop": ["</think>"]},
    ## Single inner newline: opens the region, nothing to close it with
    "<think>\n</think>\n": {
        "open": "<think>\n",
        "close": "</think>\n",
        "stop": ["</think>"],
    },
    ## A differently-named channel still works, and the tag must match at both ends
    "<reasoning>\n\n</reasoning>\n": {
        "open": "<reasoning>\n",
        "close": "\n</reasoning>\n",
        "stop": ["</reasoning>"],
    },
}


def check_parser_edge_cases():
    """Parse each synthetic prefill directly, without going through jinja."""
    for prefill, expected in SYNTHETIC.items():
        match = EMPTY_BLOCK.match(prefill)
        assert match, f"failed to parse {prefill!r}"
        tag, inner, trailing = match.group(1), match.group(2), match.group(3)
        lead = "\n" if "\n" in inner else ""
        tail = "\n" if inner.count("\n") > 1 else ""
        got = {
            "open": f"<{tag}>{lead}",
            "close": f"{tail}</{tag}>{trailing}",
            "stop": [f"</{tag}>"],
        }
        assert got == expected, f"{prefill!r}: got {got}, expected {expected}"

    ## Mismatched tags must not parse - that would produce a block that never closes
    assert EMPTY_BLOCK.match("<think>\n\n</reasoning>\n\n") is None


def self_check():
    """
    Verify the parsing logic against the chat templates guidance already ships, so this
    runs with no model download. Qwen3 has a thinking channel; Gemma 2 does not.
    """
    from guidance.chat import gemma2_9b_it_template, qwen3_gguf_template

    check_parser_edge_cases()

    style, note = extract_reasoning_style(qwen3_gguf_template)
    assert style is not None, f"qwen3 should have a thinking channel: {note}"
    assert style["open"] == "<think>\n", style
    assert style["close"] == "\n</think>\n\n", style
    assert style["stop"] == ["</think>"], style

    ## Must match what is checked in as the "think" style
    from mirna_curator.llm_functions.reasoning import REASONING_STYLES

    ## `gate` is not derivable from the assistant prefill, so compare only what is
    checked_in = {
        k: v for k, v in REASONING_STYLES["think"].items() if k in style
    }
    assert style == checked_in, (
        f"extracted style {style} has drifted from the checked-in "
        f'REASONING_STYLES["think"] {checked_in}'
    )

    gemma_style, gemma_note = extract_reasoning_style(gemma2_9b_it_template)
    assert gemma_style is None, f"gemma2 should have no thinking channel: {gemma_style}"

    ## Gemma renames assistant -> model, which is why role markers are worth extracting.
    ## Checked against the class guidance ships, so a drift in either shows up here.
    from guidance.chat import Gemma29BInstructChatTemplate

    markers = extract_role_markers(gemma2_9b_it_template)
    shipped = Gemma29BInstructChatTemplate()
    for role in ("user", "assistant"):
        assert markers[role]["start"] == shipped.get_role_start(role), (role, markers)
        assert markers[role]["end"] == shipped.get_role_end(role), (role, markers)
    assert "<start_of_turn>model" in markers["assistant"]["start"], markers
    ## Gemma 2 has no system role, and the template says so itself
    assert "unsupported" in markers["system"], markers

    print("self-check passed")
    print(f"  qwen3_gguf_template  -> {format_entry('think', style).strip()}")
    print(f"  gemma2_9b_it_template -> no thinking channel ({gemma_note})")
    return 0


def report(name, template):
    """Print the extracted style and role markers for one template."""
    style, note = extract_reasoning_style(template)
    print(f"=== {name} ===")
    print(f"reasoning style: {note}")
    print(format_entry(name, style))
    print("\nrole markers (only needed if guidance has no ChatTemplate for this model):")
    for role, marker in extract_role_markers(template).items():
        print(f"  {role}: start={marker['start']!r} end={marker['end']!r}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--self-check", action="store_true", help="run the built-in tests")
    source.add_argument("--gguf", help="path to a GGUF file to read the template from")
    source.add_argument(
        "--template-name", help="name of a chat template string in guidance.chat"
    )
    args = parser.parse_args(argv)

    if args.self_check:
        return self_check()

    if args.template_name:
        import guidance.chat

        template = getattr(guidance.chat, args.template_name)
        report(args.template_name, template)
        return 0

    ## vocab_only keeps this cheap - we only want the metadata, not the weights
    import llama_cpp

    model = llama_cpp.Llama(model_path=args.gguf, vocab_only=True, verbose=False)
    template = model.metadata.get("tokenizer.chat_template")
    if template is None:
        print(f"{args.gguf} has no tokenizer.chat_template in its metadata", file=sys.stderr)
        return 1
    report(args.gguf, template)
    return 0


if __name__ == "__main__":
    sys.exit(main())
