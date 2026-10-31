#!/bin/sh
set -eu
# runpodctl reads its key from ~/.runpod/config.toml.
if [ -n "${RUNPOD_API_KEY:-}" ] && [ ! -f "$HOME/.runpod/config.toml" ]; then
  umask 077
  mkdir -p "$HOME/.runpod"
  printf 'apikey = "%s"\napiurl = "https://api.runpod.io/graphql"\n' "$RUNPOD_API_KEY" > "$HOME/.runpod/config.toml"
fi
exec intro-detector "$@"
