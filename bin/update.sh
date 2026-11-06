#!/usr/bin/env bash
set -euo pipefail
umask 077

usage='Usage: sudo bash bin/update.sh --version vX.Y.Z --azuracast-dir DIR (optional setup input on stdin)'
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
image=azuracast-intro-detector:local
version=""
azuracast_dir=""
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
while (($#)); do
    case "$1" in
        --version) [[ $# -ge 2 ]] || fail "$usage"; version=$2; shift 2 ;;
        --azuracast-dir) [[ $# -ge 2 ]] || fail "$usage"; azuracast_dir=$2; shift 2 ;;
        *) fail "$usage" ;;
    esac
done
[[ "$version" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-rc\.[0-9]+)?$ && -n "$azuracast_dir" ]] || fail "$usage"
[[ "$EUID" == 0 ]] || fail 'Run this script with sudo, or from a root shell.'
cd "$root"
[[ -z $(git status --porcelain --untracked-files=no) ]] || \
    fail 'The detector checkout has local changes; update it manually.'
previous=$(git rev-parse HEAD)
git fetch --quiet --tags origin
target=$(git rev-parse --verify --quiet "refs/tags/$version^{commit}") || fail "Release $version was not found."
running=$(docker compose ps --status running -q detector 2>/dev/null || true)
if [[ "$target" == "$previous" && -n "$running" && -f "$azuracast_dir/plugins/azuracast-sweepers/.detector.json" ]]; then
    printf 'Intro Detector %s is already installed.\n' "$version"
    exit 0
fi
had_image=false
if docker image inspect "$image" >/dev/null 2>&1; then
    docker image tag "$image" azuracast-intro-detector:previous
    had_image=true
fi
git -c advice.detachedHead=false checkout --quiet --detach "$target"
printf 'Updating the Intro Detector to %s...\n' "$version"
if ! bash bin/setup.sh --azuracast-dir "$azuracast_dir" --unattended; then
    git -c advice.detachedHead=false checkout --quiet --detach "$previous"
    if $had_image; then
        docker image tag azuracast-intro-detector:previous "$image"
        docker compose up -d --no-build --no-deps --force-recreate detector >/dev/null 2>&1 || true
    fi
    fail "The update to $version failed; the previous version was restored."
fi
