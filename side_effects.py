#!./.venv/bin/python
"""What a vector changes on grader prompts it should leave alone: the shift tables of a <name>_side_effects.jsonl from train_vector.py, intervened against base samples from the same run.

    ./side_effects.py grader_proj
"""
import json
import sys

from mechtools import *

from common import SHIFT_HEADERS, VECTORS, shift_rows

name = sys.argv[1]
samples = [json.loads(line) for line in (VECTORS / "Qwen3.8-27B" / f"{name}_side_effects.jsonl").open()]
for kind in ["hack", "none", "clean"]:
    pick = lambda intervened: [s for s in samples if s["prompts"] == kind and s["intervened"] == intervened]
    show_table(SHIFT_HEADERS, shift_rows(f"{name} {kind}", pick(True), pick(False)))
