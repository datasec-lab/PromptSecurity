#!/usr/bin/env bash
# Record the PromptSecurity CLI walkthrough as an animated terminal (docs/demo_cli.gif).
# This is an OPTIONAL secondary demo; the main README demo is docs/demo.gif
# (the attack/defense pipeline, rendered by docs/make_demo_gif.py).
#
# The command sequence below is deliberately API-key-free: it discovers components,
# inspects a method, generates a placeholder (no model call), lists placeholders, and
# opens the dashboard. Nothing here contacts an LLM, so the recording is real and cheap.
#
# Requirements:
#   - asciinema   : https://asciinema.org/docs/installation
#   - agg         : https://github.com/asciinema/agg  (cast -> gif)
#   Run from the repo root with the venv activated:  bash docs/record_demo.sh
set -euo pipefail
cd "$(dirname "$0")/.."

command -v asciinema >/dev/null || { echo "Please install asciinema first."; exit 1; }
command -v agg       >/dev/null || { echo "Please install agg (asciinema/agg) first."; exit 1; }

# Note: the main README demo (docs/demo.gif) is the attack/defense pipeline rendered by
# docs/make_demo_gif.py. This script produces a *separate*, optional terminal-walkthrough GIF.
CAST=docs/demo_cli.cast
GIF=docs/demo_cli.gif

# A tiny driver that "types" each command with a pause so the GIF is readable.
run() { printf '\033[1;32m$\033[0m %s\n' "$*"; sleep 1.2; eval "$*"; sleep 1.6; }

export PROMPTSECURITY_DEMO=1
asciinema rec --overwrite --cols 96 --rows 30 --idle-time-limit 1.5 "$CAST" --command "$(cat <<'SCRIPT'
bash -c '
run(){ printf "\033[1;32m$\033[0m %s\n" "$*"; sleep 1.2; eval "$*"; sleep 1.6; }
clear
echo "# PromptSecurity — a plug-in benchmark for LLM prompt security"; sleep 1.5
run "python -m experiments --list all | head -40"
run "python -m experiments --info ArtPrompt attack"
run "python -m experiments --model gpt-4o --attack ArtPrompt --defense no_defense --dataset harmbench --sample-limit 3 --generate-only"
run "python -m experiments --list-placeholders | head -15"
run "python -m experiments --dashboard --dashboard-limit 8"
echo; echo "# Swap any of the five slots — attack, defense, model, judger, dataset."; sleep 2
'
SCRIPT
)"

agg --theme monokai --font-size 15 "$CAST" "$GIF"
echo "Wrote $GIF"
