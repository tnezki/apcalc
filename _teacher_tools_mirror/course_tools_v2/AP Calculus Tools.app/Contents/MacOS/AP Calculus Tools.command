#!/bin/bash
set -e
TOOLS="$(cd "$(dirname "$0")/../../../.." && pwd)"
exec /bin/bash "$TOOLS/Restart AP Calculus Tools.command"
