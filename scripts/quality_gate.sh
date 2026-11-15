#!/usr/bin/env bash
# One entry point for every quality check (issue #8). See docs/lint_rules.md.
#
#   scripts/quality_gate.sh                  # check the committed HEAD
#   scripts/quality_gate.sh --ref main       # check another ref
#   scripts/quality_gate.sh --worktree       # tracked files, including edits
#   scripts/quality_gate.sh --require-offline   # fail if network can't be cut
#   scripts/quality_gate.sh --skip-notebooks
#
# Requires: git, uv, tar. Network is used only to provision the locked
# environment; every check after that runs with network removed when the
# platform allows (Linux user namespaces).
set -euo pipefail

REF=HEAD
FROM_WORKTREE=0
REQUIRE_OFFLINE=0
SKIP_NOTEBOOKS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --ref) REF="$2"; shift 2 ;;
    --worktree) FROM_WORKTREE=1; shift ;;
    --require-offline) REQUIRE_OFFLINE=1; shift ;;
    --skip-notebooks) SKIP_NOTEBOOKS=1; shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

ROOT=$(git rev-parse --show-toplevel)
PYVER=$(cat "$ROOT/.python-version")
WORK=$(mktemp -d "${TMPDIR:-/tmp}/sim-sdk-gate.XXXXXX")
OUT="$ROOT/outputs/quality_gate"
mkdir -p "$OUT"
trap 'rm -rf "$WORK"' EXIT

say() { printf '\n== %s\n' "$*"; }

# ---- 1. Explicit inputs: a clean snapshot ---------------------------------
COMMIT=$(git -C "$ROOT" rev-parse "$REF")
export SOURCE_DATE_EPOCH=$(git -C "$ROOT" log -1 --format=%ct "$COMMIT")
snapshot() {
  mkdir -p "$1"
  if [ "$FROM_WORKTREE" = 1 ]; then
    (cd "$ROOT" && git ls-files -z | xargs -0 tar cf -) | tar xf - -C "$1"
  else
    git -C "$ROOT" archive "$COMMIT" | tar xf - -C "$1"
  fi
}
say "snapshot: $( [ $FROM_WORKTREE = 1 ] && echo "working tree (tracked files) at" ) $COMMIT"
for d in a b build_a build_b; do snapshot "$WORK/$d"; done
SRC="$WORK/a"
LOCK="$SRC/requirements/lock.txt"
RUNTIME_LOCK="$SRC/requirements/lock-runtime.txt"

# ---- 2. Locked environment (network allowed) ------------------------------
say "provision Python $PYVER and requirements/lock.txt (hash-verified)"
uv python install "$PYVER" >/dev/null 2>&1 || true
uv venv -q -p "$PYVER" "$WORK/venv"
uv pip sync -q --python "$WORK/venv/bin/python" --require-hashes \
  --build-constraints "$LOCK" "$LOCK"
uv pip install -q --python "$WORK/venv/bin/python" --no-deps \
  --no-build-isolation -e "$SRC"
# Smoke-test environment: runtime dependencies only, cached now for offline use.
uv venv -q -p "$PYVER" "$WORK/smoke"
uv pip sync -q --python "$WORK/smoke/bin/python" --require-hashes \
  --build-constraints "$LOCK" "$RUNTIME_LOCK"

# ---- 3. Network off for everything below ---------------------------------
ISOLATION="NOT ENFORCED"
RUNNER=(bash)
if command -v unshare >/dev/null 2>&1 && unshare -rn true 2>/dev/null; then
  if unshare -rn bash -c 'ip link set lo up' 2>/dev/null; then
    ISOLATION="enforced (network namespace, loopback only)"
    RUNNER=(unshare -rn bash -c 'ip link set lo up && exec bash "$0"')
  fi
fi
say "network isolation: $ISOLATION"
if [ "$ISOLATION" = "NOT ENFORCED" ] && [ "$REQUIRE_OFFLINE" = 1 ]; then
  echo "--require-offline: network isolation is unavailable on this host" >&2
  exit 1
fi

cat > "$WORK/offline.sh" <<EOF
set -uo pipefail
export UV_OFFLINE=1 PIP_NO_INDEX=1 SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH
export PYTHONDONTWRITEBYTECODE=1
PY="$WORK/venv/bin/python"
BIN="$WORK/venv/bin"
FAIL=0
status() { printf '%-28s %s\n' "\$1" "\$2" >> "$WORK/summary.txt"; [ "\$2" = PASS ] || FAIL=1; }
step() { name=\$1; shift; printf '\n== %s\n' "\$name"; if "\$@"; then status "\$name" PASS; else status "\$name" FAIL; fi; }
lint_all() {
  (cd "\$1" && {
    "\$PY" scripts/check_invariants.py
    find . -name '*.py' -not -path './.venv/*' -not -path './tests/lint_fixtures/*' \
      | LC_ALL=C sort | xargs "\$PY" scripts/check_no_bare_random.py
    "\$BIN/flake8" healthcare_sim_sdk tests scripts
    "\$BIN/ruff" check --no-cache --output-format concise .
    "\$PY" scripts/sdk_lint.py
  } 2>&1)
}
cd "$SRC"
step invariants "\$PY" scripts/check_invariants.py
step rng_guard bash -c 'find . -name "*.py" -not -path "./tests/lint_fixtures/*" | LC_ALL=C sort | xargs "$WORK/venv/bin/python" scripts/check_no_bare_random.py'
step flake8 "\$BIN/flake8" healthcare_sim_sdk tests scripts
step ruff "\$BIN/ruff" check --no-cache .
step sdk_lint "\$PY" scripts/sdk_lint.py
lint_all "$WORK/a" > "$WORK/lint_a.txt"; lint_all "$WORK/b" > "$WORK/lint_b.txt"
step lint_output_identical cmp "$WORK/lint_a.txt" "$WORK/lint_b.txt"
step tests_public "\$PY" -m pytest -q -p no:cacheprovider -m "not private" tests/
if [ "$SKIP_NOTEBOOKS" = 0 ]; then
  step notebooks "\$PY" scripts/execute_notebooks.py
else
  status notebooks SKIPPED
fi
for x in a b; do
  (cd "$WORK/build_\$x" && "\$PY" -m build --wheel --no-isolation -o "$WORK/dist_\$x" . >/dev/null 2>&1)
done
step wheel_reproducible "\$PY" "$SRC/scripts/gate_checks.py" same-wheel "$WORK/dist_a" "$WORK/dist_b" "$WORK/wheel_sha256"
step wheel_contents "\$PY" "$SRC/scripts/gate_checks.py" wheel-contents "$WORK/dist_a" "$SRC"
mkdir -p "$WORK/outside"
step wheel_smoke bash -c '
  uv pip install -q --offline --no-deps --python "$WORK/smoke/bin/python" "$WORK"/dist_a/*.whl &&
  cd "$WORK/outside" && "$WORK/smoke/bin/python" -I "$SRC/scripts/gate_checks.py" smoke'
exit \$FAIL
EOF

: > "$WORK/summary.txt"
set +e
"${RUNNER[@]}" "$WORK/offline.sh"
RESULT=$?
set -e

{
  echo "commit           $COMMIT$( [ $FROM_WORKTREE = 1 ] && echo ' + working-tree edits' )"
  echo "python           $("$WORK/venv/bin/python" -c 'import sys;print(sys.version.split()[0])')"
  echo "lock sha256      $("$WORK/venv/bin/python" -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$LOCK")"
  echo "network          $ISOLATION"
  echo "wheel sha256     $(cat "$WORK/wheel_sha256" 2>/dev/null || echo none)"
  echo
  cat "$WORK/summary.txt"
} | tee "$OUT/summary.txt"
cp "$WORK/lint_a.txt" "$OUT/lint_output.txt" 2>/dev/null || true
[ $RESULT = 0 ] && echo "quality gate: PASS" || echo "quality gate: FAIL"
exit $RESULT
