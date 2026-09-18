"""
Checks rollout_tokens.render against the qwen rollouts that have a transcript. Tokenizer only, runs on CPU.

    uv run python test_tokens.py                      # every rollout (~30 s)
    uv run python test_tokens.py --n 50               # a seeded sample
    uv run python test_tokens.py --show <rollout_id>  # print one rollout's span table

Asserts, per rollout: the re-tokenized rendering equals the template's own tokenization; one <think> block per turn;
every span decodes to its source text (up to the template's trimming) and its close token is the expected marker; spans
are in order and disjoint. Then prints token-length and span-count stats.
"""

import argparse
import collections
import random
import statistics
import time

from transformers import AutoTokenizer

from mechtools import *

from import_rollouts import load_rollouts
from rollout_tokens import render, rollout_messages, rollout_tools, end_positions

CLOSERS = {"system": {"<|im_end|>"}, "user": {"<|im_end|>"}, "reasoning": {"</think>"}, "message": {"<tool_call>", "<|im_end|>"}, "tool_call": {"</tool_call>"}, "env_output": {"</tool_response>", "<|im_end|>"}}


def source_text(r: dict, span: dict) -> str | list[str]:
    """What the span should decode to: a string, or for a tool call the argument values it must contain."""
    turn = r["turns"][span["turn"]] if span["turn"] >= 0 else None
    match span["kind"]:
        case "system" | "user":
            return r["items"][0 if span["kind"] == "system" else 1]["content"]
        case "reasoning" | "message":
            return turn[span["kind"]]
        case "env_output":
            return turn["env_outputs"][0]
        case "tool_call":
            return [str(v) for v in turn["tool_calls"][span["key"]]["args"].values()]


def check(tok, r: dict) -> tuple[int, list[dict]]:
    text, ids, spans = render(tok, r)
    assert ids == tok.apply_chat_template(rollout_messages(r), tools=rollout_tools(r), tokenize=True, return_dict=False, preserve_thinking=True), r["rollout_id"]
    assert text.count("<|im_start|>assistant\n<think>\n") == r["n_turns"], r["rollout_id"]
    assert all(len(t["env_outputs"]) == 1 for t in r["turns"]), r["rollout_id"]
    prev_close = -1
    for span in spans:
        assert prev_close < span["start"] < span["end"] <= span["close"], (r["rollout_id"], span)
        prev_close = span["close"]
        assert tok.decode(ids[span["close"]]) in CLOSERS[span["kind"]], (r["rollout_id"], span, tok.decode(ids[span["close"]]))
        decoded, source = tok.decode(ids[span["start"]:span["end"]]), source_text(r, span)
        if isinstance(source, list):
            assert all(v in decoded for v in source) and (len(source) > 1 or decoded.strip() == source[0].strip()), (r["rollout_id"], span, decoded, source)
        else:
            assert decoded.strip() == source.strip(), (r["rollout_id"], span, decoded[:200], source[:200])
    return len(ids), spans


def show(tok, r: dict) -> None:
    text, ids, spans = render(tok, r)
    print(f"{bold}{r['rollout_id']}{endc}  {len(ids)} tokens  {len(spans)} spans  {len(end_positions(spans))} end positions")
    for s in spans:
        print(f"  turn {s['turn']:>2}  {s['kind']:<10} [{s['start']:>6}, {s['end']:>6})  close {s['close']:>6} {tok.decode(ids[s['close']])!r:<18} {tok.decode(ids[s['start']:s['end']])[:70]!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", default="data/rollouts_qwen.jsonl")
    p.add_argument("--model", default="Qwen/Qwen3.6-27B")
    p.add_argument("--n", type=int, default=0, help="check a seeded random sample instead of every rollout")
    p.add_argument("--show", default=None, help="rollout_id whose span table to print")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    rollouts = [r for r in load_rollouts(args.rollouts) if r["items"]]
    if args.show:
        return show(tok, next(r for r in rollouts if r["rollout_id"] == args.show))
    if args.n:
        rollouts = random.Random(0).sample(rollouts, args.n)

    t0, lengths, kinds, n_pos = time.time(), {}, collections.Counter(), 0
    for r in pbar(rollouts, desc="checking"):
        n_tokens, spans = check(tok, r)
        lengths[r["rollout_id"]] = n_tokens
        kinds.update(s["kind"] for s in spans)
        n_pos += len(end_positions(spans))
    print(f"{green}{len(rollouts)} rollouts ok in {time.time() - t0:.0f}s{endc}")
    print(f"  tokens: median {statistics.median(lengths.values()):.0f}  max {max(lengths.values())} ({max(lengths, key=lengths.get)})  total {sum(lengths.values())}  over 32k: {sum(v > 32_000 for v in lengths.values())}")
    print(f"  spans: {dict(kinds)}  end positions total {n_pos}")


if __name__ == "__main__":
    main()
