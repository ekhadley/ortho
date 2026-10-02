#!/bin/bash
# Run from the laptop, in the repo root: bash scripts/init_vast.sh [ssh host, default vast]. Once per new container: shell setup, the box's
# github key (copied from ~/.ssh/vast_box_ed25519 here, which stays out of the repo), HF_TOKEN (from .env here, also out of the repo) and the
# claude code cli. Turns off vast's tmux-on-login.
set -e
HOST=${1:-vast}
source .env
ssh "$HOST" 'mkdir -p ~/.ssh'
scp ~/.ssh/vast_box_ed25519 "$HOST":.ssh/id_ed25519
ssh "$HOST" "HF_TOKEN=${HF_TOKEN:?} bash -s" <<'EOF'
set -e
chmod 600 ~/.ssh/id_ed25519
touch ~/.no_auto_tmux
apt-get update
apt-get install -y eza neovim btop tmux
git config --global user.email "ekhadley@gmail.com"
git config --global user.name "Ethan Hadley"
curl -fsSL https://claude.ai/install.sh | bash

cat >> ~/.bashrc <<'RC'

# init_vast.sh
[[ -n "$VIRTUAL_ENV" ]] && deactivate
export PATH=$HOME/bin:$HOME/.local/bin:/usr/local/bin:$PATH
export EDITOR="nvim"
alias bashrc="nvim ~/.bashrc"
alias v="vim"
alias wdis="wandb disabled"
alias wen="wandb enabled"
alias uvv="source ./.venv/bin/activate"
alias gs="git status"
alias gco="git checkout"
alias gcam="git commit -am"
alias gf="git fetch"
alias gu="git push"
alias gd="git pull"
alias hfcd="hf cache delete"
alias l="eza"
alias ll="eza -la --git"
alias L="eza -TL"
alias lg="eza -la --git  | grep -i"
alias big="eza -ll --total-size -s=size"
alias nsmi="nvidia-smi"
export PS1='\[\e[1;32m\]\u@\h\[\e[0m\]:\[\e[1;34m\]\w\[\e[0m\]\$ '
[[ -d /workspace/ortho ]] && cd /workspace/ortho
RC
echo "export HF_TOKEN=\"$HF_TOKEN\"" >> ~/.bashrc
EOF
