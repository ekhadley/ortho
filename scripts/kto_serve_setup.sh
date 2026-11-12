#!/bin/bash
# On the GPU box, from the repo copy (~/ortho): a serving venv with vLLM 0.32.0 and scripts/vllm_proj.py installed as a vLLM plugin, for scripts/kto_serve_row.sh.
# The weights download on the first serve. No vllm-lens, so the server is not eager.
set -e
mkdir -p ~/proj_plugin ~/hooks
cp scripts/vllm_proj.py ~/proj_plugin/vllm_proj.py
cp scripts/kto_serve_row.sh ~/serve_row.sh
cat > ~/proj_plugin/pyproject.toml <<'TOML'
[project]
name = "vllm-proj-plugin"
version = "0.1.0"
dependencies = []

[project.entry-points."vllm.general_plugins"]
proj = "vllm_proj:register"

[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
py-modules = ["vllm_proj"]
TOML
uv venv ~/serve --python 3.12
uv pip install --python ~/serve/bin/python vllm==0.32.0
uv pip install --python ~/serve/bin/python -e ~/proj_plugin
~/serve/bin/python -c "import vllm, torch; print(vllm.__version__, torch.__version__)"
~/serve/bin/python -c "from importlib.metadata import entry_points; print([e for e in entry_points(group=\"vllm.general_plugins\")])"
echo SERVE_SETUP_DONE
