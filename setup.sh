#!/usr/bin/env bash
# Sets up the retrieval lab on a GPU machine. Runs as a cloud startup script (root, at first boot) or by hand.
# Safe to re-run. Env: LAB_REF = branch or commit (default main); LAB_SETUP_STEPS = all (default) or base (system
# packages, clone and venv only). With your lab token as its argument (curl … | bash -s TOKEN) it also connects to the
# lab (LAB_URL, default https://lab.tryrealm.dev) and shows lab status. Writes ~/.lab/setup.done, or ~/.lab/setup.failed
# naming the failed step.
# As root it also adds a login message to ~/.bashrc: the setup state.
set -Eeuo pipefail

# Startup scripts may run without a usable HOME.
if [ -z "${HOME:-}" ] || [ "$HOME" = / ]; then
  HOME=$(getent passwd "$(id -un)" | cut -d: -f6)
  export HOME
fi
LAB_DIR=$HOME/.lab
REPO=$HOME/retrieval-lab
REF=${LAB_REF:-main}
STEPS=${LAB_SETUP_STEPS:-all}

mkdir -p "$LAB_DIR"
rm -f "$LAB_DIR/setup.done" "$LAB_DIR/setup.failed"
# Output goes through tee, so tell the lab CLI the terminal's width (it would assume 80 columns).
COLUMNS=$( { [ -t 1 ] && tput cols; } 2> /dev/null || echo 120)
export COLUMNS
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PIP_NO_CACHE_DIR=1  # a GPU box's disk holds the models; a second copy of every wheel does not fit
exec > >(tee -a "$LAB_DIR/setup.log") 2>&1

step=start
# Subshells inherit the trap (set -E), so only the first failure reports.
trap '[ -e "$LAB_DIR/setup.failed" ] || { echo "$step (exit $?)" > "$LAB_DIR/setup.failed"; echo "failed: $step"; }' ERR
begin() {
  step=$1
  echo "== $(date -Is) $step"
}

begin "login message"
# On a box (root), every interactive login says how setup went and enters the lab's venv.
if [ "$(id -u)" = 0 ]; then
  touch "$HOME/.bashrc"
  sed -i '/^# >>> retrieval-lab >>>$/,/^# <<< retrieval-lab <<<$/d' "$HOME/.bashrc"
  cat >> "$HOME/.bashrc" << 'EOF'
# >>> retrieval-lab >>>
case $- in *i*)
  if [ -f ~/.lab/setup.done ]; then
    cd ~/retrieval-lab && source .venv/bin/activate
    echo "ready · notebook: http://127.0.0.1:8888/lab/tree/lab.ipynb"
  elif [ -f ~/.lab/setup.failed ]; then
    tail -n 15 ~/.lab/setup.log
    echo "setup failed: $(cat ~/.lab/setup.failed) · run it again, or show this to an organizer"
  else
    echo "setup running: $(grep '^== ' ~/.lab/setup.log | tail -n 1 | cut -d' ' -f3-)"
    echo 'watch: tail -f ~/.lab/setup.log'
  fi
;; esac
# <<< retrieval-lab <<<
EOF
fi

begin "system packages"
echo "user $(id -un), HOME $HOME, ref $REF, steps $STEPS"
[[ $STEPS = all || $STEPS = base ]] || { echo "LAB_SETUP_STEPS: all or base"; false; }
missing=()
command -v git > /dev/null || missing+=(git)
python3 -c 'import ensurepip' 2> /dev/null || missing+=(python3-venv)
# vLLM compiles kernels when the model server starts: Triton needs Python.h, FlashInfer needs ninja.
[ -f "$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')/Python.h" ] || missing+=(python3-dev)
command -v ninja > /dev/null || missing+=(ninja-build)
# The Vespa track runs Vespa in Docker.
command -v docker > /dev/null || missing+=(docker.io)
if [ ${#missing[@]} -gt 0 ]; then
  sudo=
  [ "$(id -u)" = 0 ] || sudo=sudo
  # At first boot unattended-upgrades may hold the dpkg lock for a while.
  $sudo apt-get -o DPkg::Lock::Timeout=600 update -q
  DEBIAN_FRONTEND=noninteractive $sudo apt-get -o DPkg::Lock::Timeout=600 install -yq "${missing[@]}"
fi

begin gpu
# The agent model, the reranker, vLLM and Vespa's data take about 90 GB beside the OS; Vespa stops feeding at 95% full.
if [ "$STEPS" = all ] && command -v nvidia-smi > /dev/null && nvidia-smi -L > /dev/null 2>&1; then
  rm -rf "$HOME/.cache/pip"
  disk_gb=$(df -BG --output=size / | tail -n 1 | tr -dc 0-9)
  [ "$disk_gb" -ge 190 ] || { echo "disk ${disk_gb} GB, a GPU box needs 200 · delete this box, create one with a 200 GB disk"; false; }
fi
# Confidential-computing GPUs (Verda's .CC types) refuse CUDA until their ready state is set, again after every boot.
if [ "$(id -u)" = 0 ] && command -v nvidia-smi > /dev/null && nvidia-smi conf-compute -f 2> /dev/null | grep -q 'CC status: ON'; then
  cat > /etc/systemd/system/lab-cc-ready.service << 'EOF'
[Unit]
Description=Set the confidential-computing GPUs ready, so CUDA can use them
After=nvidia-persistenced.service

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'nvidia-smi conf-compute -grs | grep -q ": ready" || nvidia-smi conf-compute -srs 1'

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  systemctl enable --now lab-cc-ready.service
fi

begin "clone $REF"
if [ -d "$REPO/.git" ]; then
  git -C "$REPO" fetch -q origin
else
  git clone -q https://github.com/oskrim/retrieval-lab "$REPO"
fi
# A notebook with your own runs in it is kept beside the new one.
if [ -n "$(git -C "$REPO" status --porcelain -- lab.ipynb)" ]; then
  kept="lab-$(date +%Y%m%d-%H%M%S).ipynb"
  cp "$REPO/lab.ipynb" "$REPO/$kept"
  git -C "$REPO" checkout -- lab.ipynb
  echo "your notebook is kept as $kept"
fi
# An untracked search/ or agent.py (from an older lab's lab new) would be overwritten by the update or block it; it
# waits in ~/.lab/untracked, then goes back over the lab's as your edits of it.
for f in search agent.py; do
  if [ -e "$REPO/$f" ] && [ ! -e "$LAB_DIR/untracked/$f" ] && [ -z "$(git -C "$REPO" ls-files -- "$f")" ]; then
    mkdir -p "$LAB_DIR/untracked"
    mv "$REPO/$f" "$LAB_DIR/untracked/$f"
  fi
done
if git -C "$REPO" rev-parse -q --verify "origin/$REF" > /dev/null; then
  git -C "$REPO" checkout -q "$REF"
  # Your edits go aside, the lab moves to the newest commit (branch lab-before-update keeps the old one), and your
  # edits come back on top. A file the lab changed where you did stays yours; the lab's version waits beside it.
  stashes=$(git -C "$REPO" stash list | wc -l)
  git -C "$REPO" stash push -q
  git -C "$REPO" branch -f lab-before-update HEAD
  git -C "$REPO" reset -q --hard "origin/$REF"
  if [ "$(git -C "$REPO" stash list | wc -l)" -gt "$stashes" ]; then
    if ! git -C "$REPO" stash apply -q > /dev/null 2>&1; then
      for f in $(git -C "$REPO" diff --name-only --diff-filter=U); do
        mkdir -p "$(dirname "$LAB_DIR/lab-version/$f")"
        git -C "$REPO" show "origin/$REF:$f" > "$LAB_DIR/lab-version/$f"
        git -C "$REPO" checkout -q --theirs -- "$f"
        echo "your $f is kept; the lab's new version of it is ~/.lab/lab-version/$f"
      done
      git -C "$REPO" reset -q
    fi
    git -C "$REPO" stash drop -q
  fi
else
  git -C "$REPO" checkout -q "$REF"
fi
for f in search agent.py; do
  if [ -e "$LAB_DIR/untracked/$f" ]; then
    cp -aT "$LAB_DIR/untracked/$f" "$REPO/$f"
    rm -rf "${LAB_DIR:?}/untracked/$f"
    echo "your $f is kept: git diff -- $f shows your edits"
  fi
done
git -C "$REPO" log -1 --format='at %h %s'

begin venv
[ -x "$REPO/.venv/bin/python" ] || python3 -m venv "$REPO/.venv"
# lab also works where .bashrc does not run (ssh box 'lab status', tmux scripts).
[ "$(id -u)" = 0 ] && ln -sf "$REPO/.venv/bin/lab" /usr/local/bin/lab

if [ "$STEPS" = all ]; then
  begin "pip install"
  "$REPO/.venv/bin/pip" install -q -e "$REPO"
  begin "gpu check"
  # A GPU that CUDA cannot use would silently turn the box into no-GPU mode.
  if command -v nvidia-smi > /dev/null && nvidia-smi -L > /dev/null 2>&1; then
    "$REPO/.venv/bin/python" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' \
      || { echo "the GPU is there but CUDA cannot use it"; false; }
  fi
  begin "lab setup"
  (cd "$REPO" && .venv/bin/lab setup)
  begin "vespa up"
  (cd "$REPO" && .venv/bin/lab vespa up)
fi

if [ "$(id -u)" = 0 ] && [ "$STEPS" = all ]; then
  begin notebook
  # Listens on 127.0.0.1 only, reached through the SSH tunnel (ssh -L 8888:127.0.0.1:8888), so it needs no token.
  cat > /etc/systemd/system/lab-jupyter.service << EOF
[Unit]
Description=Jupyter for the retrieval lab
After=network.target

[Service]
WorkingDirectory=$REPO
Environment=HOME=$HOME
ExecStart=$REPO/.venv/bin/jupyter lab --no-browser --allow-root --ip 127.0.0.1 --port 8888 --IdentityProvider.token= --ServerApp.password=
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  systemctl enable lab-jupyter.service
  systemctl restart lab-jupyter.service
fi

if [ -n "${1:-}" ]; then
  begin connect
  "$REPO/.venv/bin/lab" connect "${LAB_URL:-https://lab.tryrealm.dev}" "$1" \
    || echo "connect failed · run the setup line from your lab page again"
  "$REPO/.venv/bin/lab" status || true
fi

begin done
touch "$LAB_DIR/setup.done"
if [ -z "${1:-}" ] && [ ! -f "$REPO/.lab.json" ]; then
  echo "not connected: run the setup line from your lab page, with your token"
fi
