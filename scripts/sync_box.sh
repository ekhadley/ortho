#!/bin/bash
# Run from the laptop, in the repo root: setsid nohup bash scripts/sync_box.sh [ssh host, default vast] [minutes, default 30] > logs/sync_box.log 2>&1 < /dev/null &
# A backup of a run launched on the box (run.py under nohup there), so a dead box loses at most one interval. Every interval it copies into the
# same places here: the Inspect logs (logs/<env>/*.eval, which hold every finished sample even mid-run, so records can be rebuilt from them with
# rhenvs.convert), the converted records (data/inspect/), and the launcher's output (~/bulk.log as logs/parity_bulk.txt). rsync sends only new
# and changed files (and only the changed parts of a file), never deletes anything here or replaces a file that is newer here, and keeps the previous copy of each replaced .eval in
# ~/.cache/ortho_box_sync_prev, in case a pass read a log while Inspect was rewriting it. A failed pass (box down, network) is logged and the
# next one tries again. pkill -f sync_box.sh stops it.
HOST=${1:-vast}
MINUTES=${2:-30}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=4"
while true; do
    echo "=== $(date -u +%FT%TZ) syncing from $HOST"
    timeout 1500 rsync -au --info=name --include='*/' --include='*.eval' --exclude='*' --backup --backup-dir="$HOME/.cache/ortho_box_sync_prev" -e "$SSH" "$HOST":ortho/logs/ logs/
    echo "logs: exit $?"
    timeout 1500 rsync -au --info=name -e "$SSH" "$HOST":ortho/data/inspect/ data/inspect/
    echo "records: exit $?"
    timeout 300 rsync -a -e "$SSH" "$HOST":bulk.log logs/parity_bulk.txt
    echo "bulk.log: exit $?"
    sleep $((MINUTES * 60))
done
