#!/usr/bin/env bash
# On the CheatBench CPU VM (set up by Will's cheatbench/setup_cpu_vm.sh, with a models.local.yaml entry cb-<ROW> and a tunnel to the GPU box on VLLM_PORT), as root in tmux:
# bash cheatbench_row.sh ROW runs one CheatBench row against the served cb_<ROW>: the 9 judged tasks at the pilot sizes of Will's cheatbench/mission_a3.md
# (14 episodes each, subset_sum 10), one run.py per task as the cheatbench user, 21 episodes at once. It returns when every task has ended. Resumable: episodes
# with an ok record are skipped. Outputs: /home/cheatbench/cheatbench/outputs/<task>[_<variant>]_cb-<ROW>_<RUN_ID>/   Logs: /root/cb_logs/
# RUN_ID is a3eval because Will's scripts/cheatbench_judge_openrouter.py globs *_a3eval. scripts/cheatbench_table.py tabulates the judged rows.
set -eu
row=$1
RUN_ID=${RUN_ID:-a3eval}
VLLM_PORT=${VLLM_PORT:-8000}
CB=/home/cheatbench/cheatbench
PY311=/home/cheatbench/.local/share/uv/python/cpython-3.11.13-linux-x86_64-gnu/bin/python3.11
curl -fsS --max-time 10 http://127.0.0.1:$VLLM_PORT/v1/models | jq -r '.data[].id' | grep -qxF "cb_$row" || { echo "127.0.0.1:$VLLM_PORT is not serving cb_$row" >&2; exit 1; }
grep -q "^cb-$row:" $CB/configs/models.local.yaml || { echo "no cb-$row entry in configs/models.local.yaml" >&2; exit 1; }
grep -q 'vllm-tunnel' /etc/hosts || { echo "no vllm-tunnel alias in /etc/hosts (chess needs it)" >&2; exit 1; }
mkdir -p /root/cb_logs
# task:limit:max-concurrent
for spec in chess:14:5 knowledge_work:14:7 svgs:14:2 protein_design:14:2 synthetic_puzzles:14:1 openmath_agent:14:1 subset_sum:10:1 prime_factorization:14:1 geoguessr:14:1; do
  IFS=: read -r task limit maxc <<< "$spec"
  url=http://127.0.0.1:$VLLM_PORT/v1
  # chess is mcp_only: run.py starts its TLS bridge only for a model host that is not a loopback name
  [ "$task" = chess ] && url=http://vllm-tunnel:$VLLM_PORT/v1
  log=/root/cb_logs/${task}_cb-${row}_${RUN_ID}.$(date -u +%Y%m%dT%H%M%SZ).log
  sudo -u cheatbench -H env VLLM_BASE_URL=$url RH_PY311=$PY311 bash -c "cd $CB && exec .venv/bin/python -u run.py $task --model cb-$row --models-config configs/models.local.yaml --limit $limit --repeat 1 --max-concurrent $maxc --no-judge --judge cb-$row --run-id $RUN_ID --no-progress" > "$log" 2>&1 &
  echo "$(date -u +%FT%TZ) launched $task limit=$limit maxc=$maxc log=$log"
done
wait
echo "$(date -u +%FT%TZ) row $row: every task ended"
grep -H -A4 '^=== summary ===' /root/cb_logs/*_cb-${row}_${RUN_ID}.*.log
# every egress-locked episode leaves its relay's child process behind
pgrep -u cheatbench -f 'run\.py' > /dev/null || pkill -u cheatbench -f egress_relay.py || true
