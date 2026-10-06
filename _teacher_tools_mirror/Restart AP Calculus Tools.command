#!/bin/bash
set -euo pipefail

if [ -d "$HOME/GitHub" ]; then
  ROOT="$HOME/GitHub"
elif [ -d "$HOME/Documents/GitHub" ]; then
  ROOT="$HOME/Documents/GitHub"
else
  echo "Could not find the GitHub workspace."
  read -r
  exit 1
fi

TOOLS="$ROOT/_apcalc_teacher_tools"
SERVER="$TOOLS/app/server.py"
LOG="$TOOLS/runtime/apcalc_tools_restart.log"
mkdir -p "$TOOLS/runtime"

if [ ! -f "$SERVER" ]; then
  echo "AP Calculus Tools runtime is missing: $SERVER"
  read -r
  exit 1
fi

OLD_PIDS="$(/usr/sbin/lsof -tiTCP:8769 -sTCP:LISTEN 2>/dev/null || true)"
if [ -n "$OLD_PIDS" ]; then
  /bin/kill $OLD_PIDS >/dev/null 2>&1 || true
  for i in 1 2 3 4 5 6 7 8; do
    if ! /usr/sbin/lsof -tiTCP:8769 -sTCP:LISTEN >/dev/null 2>&1; then
      break
    fi
    /bin/sleep 0.25
  done
  LEFT="$(/usr/sbin/lsof -tiTCP:8769 -sTCP:LISTEN 2>/dev/null || true)"
  [ -z "$LEFT" ] || /bin/kill -9 $LEFT >/dev/null 2>&1 || true
fi

: > "$LOG"
PLANNER_NO_BROWSER=1 PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 "$SERVER" >> "$LOG" 2>&1 &

for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  VERSION="$(/usr/bin/curl -fsS --max-time 1 http://127.0.0.1:8769/api/version 2>/dev/null || true)"
  if echo "$VERSION" | /usr/bin/grep -q '2.0-clean'; then
    STAMP="$(/bin/date +%s)"
    /usr/bin/open "http://127.0.0.1:8769/?fresh=$STAMP"
    exit 0
  fi
  /bin/sleep 0.5
done

echo "AP Calculus Tools did not start the clean runtime."
echo "Check: $LOG"
read -r
exit 1
