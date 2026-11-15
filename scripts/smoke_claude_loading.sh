#!/usr/bin/env bash
# Smoke check: which instruction files does Claude Code load from this repo?
#
# Needs a signed-in `claude` CLI, so it is run by hand, not in the gate.
# It asks Claude Code to read one scenario file and records every instruction
# file loaded, through an InstructionsLoaded hook. Expected:
#   - CLAUDE.md (the symlink to AGENTS.md) at session start
#   - .claude/rules/nested-agents.md when the scenario file is read
#
#   scripts/smoke_claude_loading.sh            # prints the files and PASS/FAIL
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
command -v claude >/dev/null || { echo "claude CLI not found"; exit 2; }
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
cat > "$WORK/settings.json" <<JSON
{"hooks":{"InstructionsLoaded":[{"hooks":[{"type":"command",
 "command":"cat >> '$WORK/loaded.jsonl'; echo >> '$WORK/loaded.jsonl'"}]}]}}
JSON
echo "claude $(claude --version)"
cd "$ROOT"
if ! claude -p "Use the Read tool on healthcare_sim_sdk/scenarios/stroke_prevention/scenario.py with limit 5, then reply with the one word: done" \
    --settings "$WORK/settings.json" --allowedTools Read < /dev/null > "$WORK/reply.txt" 2>&1; then
  echo "claude did not run (is it signed in?):"; sed 's/^/  /' "$WORK/reply.txt"; exit 2
fi
touch "$WORK/loaded.jsonl"
echo "instruction files loaded:"
python3 - "$WORK/loaded.jsonl" "$ROOT" <<'PY'
import json, os, sys
log, root = sys.argv[1], os.path.realpath(sys.argv[2])
seen = []
for line in open(log):
    line = line.strip()
    if not line:
        continue
    event = json.loads(line)
    path = os.path.relpath(os.path.realpath(event.get("file_path", "")), root)
    seen.append(path)
    print(f"  {event.get('load_reason', '?'):<18} {path}")
# CLAUDE.md is a symlink, so its real path is AGENTS.md.
need = {"AGENTS.md": "root guidance (through CLAUDE.md)",
        ".claude/rules/nested-agents.md": "path-scoped rule for local guidance"}
missing = [f"{p}: {why}" for p, why in need.items() if p not in seen]
for item in missing:
    print(f"NOT LOADED  {item}")
print("PASS" if not missing else "FAIL")
sys.exit(1 if missing else 0)
PY
