#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GITHUB_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE="$GITHUB_ROOT/_apcalc_teacher_tools"
DEST="$GITHUB_ROOT/apcalc/_teacher_tools_mirror"

if [ ! -d "$SOURCE" ]; then
  echo "ERROR: AP Calculus tools folder not found: $SOURCE"
  exit 1
fi

if [ ! -d "$GITHUB_ROOT/apcalc/.git" ]; then
  echo "ERROR: apcalc Git repository not found: $GITHUB_ROOT/apcalc"
  exit 1
fi

mkdir -p "$DEST"

rsync -a --delete \
  --exclude='.DS_Store' \
  --exclude='__MACOSX/' \
  --exclude='runtime/' \
  --exclude='requests/' \
  --exclude='warmup_builder/backups/' \
  --exclude='warmup_builder/pending/' \
  --exclude='*.log' \
  --exclude='*.pid' \
  --exclude='*.tmp' \
  --exclude='*.zip' \
  --exclude='*.bak' \
  --exclude='*.pyc' \
  --exclude='__pycache__/' \
  "$SOURCE/" "$DEST/"

cat > "$DEST/README.md" <<'EOF_README'
# AP Calculus Teacher Tools Mirror

This folder is a one-way diagnostic mirror of the local `_apcalc_teacher_tools` source.

- Source of truth for the running teacher app remains the local `_apcalc_teacher_tools` folder.
- Do not edit files in this mirror.
- Refresh the mirror by running `_apcalc_teacher_tools/Refresh AP Calc GitHub Mirror.command`.
- Runtime logs, request packages, temporary files, backups, caches, and ZIP files are intentionally excluded.
- The leading underscore keeps this diagnostic folder out of the normal GitHub Pages/Jekyll site build.

The generated `MIRROR_MANIFEST.json` records a SHA-256 hash for every mirrored file so the exact mirrored state can be verified from GitHub.
EOF_README

python3 - "$DEST" <<'PY'
import hashlib
import json
import os
import sys

root = os.path.abspath(sys.argv[1])
manifest_path = os.path.join(root, "MIRROR_MANIFEST.json")
items = []
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
    for name in sorted(filenames):
        path = os.path.join(dirpath, name)
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        if rel == "MIRROR_MANIFEST.json":
            continue
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        items.append({"path": rel, "sha256": h.hexdigest(), "bytes": os.path.getsize(path)})

payload = {
    "schema_version": 1,
    "source": "_apcalc_teacher_tools",
    "destination": "apcalc/_teacher_tools_mirror",
    "file_count": len(items),
    "files": items,
}
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")
PY

echo
echo "AP Calculus teacher-tools mirror refreshed."
echo "Source:      $SOURCE"
echo "Git mirror:  $DEST"
echo
echo "Next: run GitHub Sync -> Commit + Push."
echo "After that, ChatGPT can inspect the exact mirrored tool state from the apcalc repository."
echo
read -r -p "Press Return to close..." _
