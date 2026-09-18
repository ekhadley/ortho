"""
Render a rollout from data/rollouts_*.jsonl into the token sequence a Qwen3-family model sees, with a span table locating
every segment (system prompt, user prompt, and per turn: reasoning, message, tool call arguments, env output) in it.

Rendering goes through the model's own chat template with preserve_thinking=True, so every turn keeps its <think> block.
A segment's token span is found by re-rendering with that segment's text swapped for SENTINEL, diffing the two strings,
and mapping the char span onto tokens with the fast tokenizer's offsets. This survives template trimming and marker
strings inside model text. Each span records `start`/`end` (content tokens, end exclusive; a token straddling the content
edge counts as content) and `close`, the special token that ends the segment (</think>, </tool_call>, </tool_response>,
<|im_end|>, or <tool_call> after a message that precedes a call). A tool call's content is its argument values.
"""

import copy
from bisect import bisect_left, bisect_right

from mechtools import *

SENTINEL = ""


def tool_schema(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": props, "required": required}}}


ANSWER = {"answer": {"type": "integer", "description": "Your final answer for the secret number"}}
EXPLANATION = {"explanation": {"type": "string", "description": "Explain your reasoning for choosing this answer"}}
SUBMIT_TOOL = tool_schema("submit", "Submit your final answer for the secret number. This ends the game.", ANSWER, ["answer"])
SUBMIT_EXPLAIN_TOOL = tool_schema("submit", "Submit your final answer for the secret number. This ends the game.", ANSWER | EXPLANATION, ["answer", "explanation"])
EXECUTE_TOOL = tool_schema("execute_command", "Execute a shell command in the UNIX environment. Returns the command output and exit code.", {"command": {"type": "string", "description": "The shell command to execute"}}, ["command"])


def rollout_tools(r: dict) -> list[dict]:
    return [SUBMIT_EXPLAIN_TOOL if r["answer_explanation"] else SUBMIT_TOOL, EXECUTE_TOOL]


def rollout_messages(r: dict) -> list[dict]:
    """Chat messages for the template: the two prompt items, then per turn one assistant message and one tool (or user, for a turn with no call) message per env output."""
    msgs = [r["items"][0], r["items"][1]]
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user", r["rollout_id"]
    for turn in r["turns"]:
        calls = [{"type": "function", "function": {"name": c["name"], "arguments": c["args"]}} for c in turn["tool_calls"]]
        msgs.append({"role": "assistant", "reasoning_content": turn["reasoning"], "content": turn["message"], "tool_calls": calls})
        msgs += [{"role": "tool" if calls else "user", "content": out} for out in turn["env_outputs"]]
    return msgs


def segments(msgs: list[dict]) -> list[dict]:
    """Every non-empty segment in transcript order: its turn (-1 for the prompt), kind, message index, and the key holding its text (a tool call index for tool_call)."""
    segs, turn = [], -1
    for mi, m in enumerate(msgs):
        if m["role"] == "assistant":
            turn += 1
            segs += [{"turn": turn, "kind": kind, "mi": mi, "key": key} for kind, key in (("reasoning", "reasoning_content"), ("message", "content")) if m[key].strip()]
            segs += [{"turn": turn, "kind": "tool_call", "mi": mi, "key": j} for j in range(len(m["tool_calls"]))]
        elif m["content"].strip():
            segs.append({"turn": turn, "kind": m["role"] if turn < 0 else "env_output", "mi": mi, "key": "content"})
    return segs


def mark(msgs: list[dict], seg: dict) -> list[dict]:
    marked, m = list(msgs), copy.deepcopy(msgs[seg["mi"]])
    if seg["kind"] == "tool_call":
        m["tool_calls"][seg["key"]]["function"]["arguments"] = {k: SENTINEL for k in m["tool_calls"][seg["key"]]["function"]["arguments"]}
    else:
        m[seg["key"]] = SENTINEL
    marked[seg["mi"]] = m
    return marked


def render(tok, r: dict) -> tuple[str, list[int], list[dict]]:
    """(rendered text, token ids, spans) for a rollout. Spans are segments() entries plus start, end, close token indices."""
    msgs, tools = rollout_messages(r), rollout_tools(r)
    text = tok.apply_chat_template(msgs, tools=tools, tokenize=False, preserve_thinking=True)
    assert SENTINEL not in text, r["rollout_id"]
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, starts, ends = enc["input_ids"], [s for s, _ in enc["offset_mapping"]], [e for _, e in enc["offset_mapping"]]
    spans = []
    for seg in segments(msgs):
        marked = tok.apply_chat_template(mark(msgs, seg), tools=tools, tokenize=False, preserve_thinking=True)
        c0, c1 = marked.index(SENTINEL), len(text) - (len(marked) - marked.rindex(SENTINEL) - 1)
        t0, t1 = bisect_right(ends, c0), bisect_left(starts, c1)
        close = next(i for i in range(t1, len(ids)) if ids[i] in tok.added_tokens_decoder)
        spans.append({**seg, "start": t0, "end": t1, "close": close})
    return text, ids, spans


def end_positions(spans: list[dict]) -> list[int]:
    """The last content token and the closing token of every span, sorted and deduplicated."""
    return sorted({p for s in spans for p in (s["end"] - 1, s["close"])})
