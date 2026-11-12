#!/bin/bash
# From the repo root: uploads data/, logs/ and acts/ to the HF dataset eekay/ortho under the same paths. Resumable: files already there are skipped, nothing is deleted.
# The dataset is public, so everything under those three directories is published. Left out: acts/weights (a cache of Qwen's weights) and the two downloaded copies of Will's dataset.
# The Hub rejects a commit holding a file that starts as text and turns binary later (a terminal dump): the fix is a line for its extension in the dataset's .gitattributes.
# A fresh clone gets it all back with: hf download eekay/ortho --repo-type dataset --local-dir .
.venv/bin/hf upload eekay/ortho . . --repo-type dataset --include "data/*" --include "logs/*" --include "acts/*" --exclude "acts/weights/*" --exclude "data/will/*" --exclude "acts/kto_sweep/cheatbench/will/*" --exclude "*.gitattributes" --exclude "*__pycache__*"
