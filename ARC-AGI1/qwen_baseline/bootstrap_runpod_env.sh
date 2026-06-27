#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ACTION="${1:-bootstrap}"
VENV_DIR="${VENV_DIR:-/root/arc311}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
UNSLOTH_TREE="${UNSLOTH_TREE:-/workspace/kaggle_artifacts/pip-install-unsloth-flash-patch_output}"
SGLANG_WHEEL_DIR="${SGLANG_WHEEL_DIR:-/workspace/kaggle_artifacts/notebookc4ca2ea220_output/offline_pkgs}"
TRITON_PTXAS_PATH_DEFAULT="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"

die() {
  echo "[bootstrap] $*" >&2
  exit 1
}

need_path() {
  local path="$1"
  [ -e "$path" ] || die "missing required path: $path"
}

activate_env() {
  [ -d "$VENV_DIR" ] || die "venv not found at $VENV_DIR"
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  export PATH="$VENV_DIR/bin:$PATH"
  export TRITON_PTXAS_PATH="$TRITON_PTXAS_PATH_DEFAULT"
}

site_packages_dir() {
  python - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
}

sync_unsloth_tree() {
  local target_dir="$1"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "$UNSLOTH_TREE"/ "$target_dir"/
  else
    cp -a "$UNSLOTH_TREE"/. "$target_dir"/
  fi
}

install_sglang_wheels() {
  python -m pip install --no-index --find-links="$SGLANG_WHEEL_DIR" \
    torch==2.8.0 \
    ninja==1.13.0 \
    pybase64==1.4.2 \
    pydantic==2.11.7 \
    sglang==0.5.1.post3
}

verify_env() {
  activate_env
  python - <<'PY'
import torch
import unsloth
import sglang

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("sglang", sglang.__version__)
print("unsloth_ok", hasattr(unsloth, "__file__"))
PY
}

write_activate_helper() {
  local helper="$ROOT_DIR/activate_runpod_env.sh"
  cat > "$helper" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "$VENV_DIR/bin/activate"
export PATH="$VENV_DIR/bin:\$PATH"
export TRITON_PTXAS_PATH="$TRITON_PTXAS_PATH_DEFAULT"
cd "$ROOT_DIR"
EOF
  chmod +x "$helper"
}

bootstrap() {
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "python interpreter not found: $PYTHON_BIN"
  need_path "$UNSLOTH_TREE"
  need_path "$SGLANG_WHEEL_DIR"

  if [ ! -d "$VENV_DIR" ]; then
    echo "[bootstrap] creating venv at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi

  activate_env
  echo "[bootstrap] upgrading base packaging tools"
  python -m pip install --upgrade pip setuptools wheel

  echo "[bootstrap] installing SGLang wheelhouse into $VENV_DIR"
  install_sglang_wheels

  local site_packages
  site_packages="$(site_packages_dir)"
  echo "[bootstrap] syncing Unsloth package tree into $site_packages"
  sync_unsloth_tree "$site_packages"

  write_activate_helper
  echo "[bootstrap] verifying env"
  verify_env

  cat <<EOF
[bootstrap] ready
repo: $ROOT_DIR
venv: $VENV_DIR
activate: source $ROOT_DIR/activate_runpod_env.sh
example: python starter.py --use-sglang --sglang-tp-size 1 --nprocs 1 ...
EOF
}

case "$ACTION" in
  bootstrap)
    bootstrap
    ;;
  verify)
    verify_env
    ;;
  activate-helper)
    write_activate_helper
    echo "$ROOT_DIR/activate_runpod_env.sh"
    ;;
  *)
    die "unknown action: $ACTION (expected: bootstrap, verify, activate-helper)"
    ;;
esac
