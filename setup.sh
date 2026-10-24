#!/usr/bin/env bash
# Sets up the retrieval lab on a GPU machine. Runs as a cloud startup script (root, at first boot) or by hand.
# Safe to re-run. Env: LAB_REF = branch or commit (default main); LAB_SETUP_STEPS = all (default) or base
# (system packages, clone and venv only). Writes ~/.lab/setup.done, or ~/.lab/setup.failed naming the failed step.
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
exec > >(tee -a "$LAB_DIR/setup.log") 2>&1

step=start
trap 'echo "$step (exit $?)" > "$LAB_DIR/setup.failed"; echo "setup failed at step: $step"' ERR
begin() {
  step=$1
  echo "== $(date -Is) $step"
}

begin "system packages"
echo "user $(id -un), HOME $HOME, ref $REF, steps $STEPS"
[[ $STEPS = all || $STEPS = base ]] || { echo "LAB_SETUP_STEPS must be all or base"; false; }
missing=()
command -v git > /dev/null || missing+=(git)
python3 -c 'import ensurepip' 2> /dev/null || missing+=(python3-venv)
# vLLM's Triton compiles a C helper against Python.h when the model server starts.
[ -f "$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')/Python.h" ] || missing+=(python3-dev)
if [ ${#missing[@]} -gt 0 ]; then
  sudo=
  [ "$(id -u)" = 0 ] || sudo=sudo
  # At first boot unattended-upgrades may hold the dpkg lock for a while.
  $sudo apt-get -o DPkg::Lock::Timeout=600 update -q
  DEBIAN_FRONTEND=noninteractive $sudo apt-get -o DPkg::Lock::Timeout=600 install -yq "${missing[@]}"
fi

begin "clone $REF"
if [ -d "$REPO/.git" ]; then
  git -C "$REPO" fetch -q origin
else
  git clone -q https://github.com/oskrim/retrieval-lab "$REPO"
fi
if git -C "$REPO" rev-parse -q --verify "origin/$REF" > /dev/null; then
  git -C "$REPO" checkout -q "$REF"
  git -C "$REPO" merge -q --ff-only "origin/$REF"
else
  git -C "$REPO" checkout -q "$REF"
fi
git -C "$REPO" log -1 --format='at %h %s'

begin venv
[ -x "$REPO/.venv/bin/python" ] || python3 -m venv "$REPO/.venv"

if [ "$STEPS" = all ]; then
  begin "pip install"
  "$REPO/.venv/bin/pip" install -q -e "$REPO"
  begin "lab setup"
  (cd "$REPO" && .venv/bin/lab setup)
fi

begin done
touch "$LAB_DIR/setup.done"
