#!/usr/bin/env bash
set -euo pipefail
umask 077

[[ $# == 2 && ( "$1" == service || "$1" == worker ) ]] || {
    printf 'Usage: bash bin/image-context.sh service|worker output.tar\n' >&2
    exit 1
}
kind=$1
output=$2
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
list=$(mktemp)
selected=$(mktemp)
trap 'rm -f -- "$list" "$selected"' EXIT
git -C "$root" ls-files -z > "$list"
while IFS= read -r -d '' path; do
    include=false
    case "$path" in
        pod/tools/apply_listening_labels.py|pod/experiments/vocal-references.json)
            include=false ;;
        LICENSE|THIRD_PARTY_NOTICES.md|docs/LICENSES.md|docs/MODELS.md|licenses/*.txt|licenses/model-cards/*.md|licenses/manifest.json|gpu/image_inventory.py)
            include=true ;;
        pod/tools/melband_backend.py|pod/experiments/requirements-*.txt|pod/experiments/runpod-start.sh)
            include=true ;;
        pod/tools/*.py|pod/tools/intro_review.html|pod/experiments/*.py)
            [[ "$kind" != service ]] || include=true ;;
        gpu/Dockerfile|gpu/prefetch.py|gpu/requirements-*.lock)
            [[ "$kind" != worker ]] || include=true ;;
        Dockerfile|pyproject.toml|MANIFEST.in|docker-entrypoint.sh|src/*.py|src/intro_detector/data/selector-training-v1.json.gz|gpu/run-lanes.sh|gpu/lanes_status.py)
            [[ "$kind" != service ]] || include=true ;;
    esac
    if $include; then
        [[ -f "$root/$path" && ! -L "$root/$path" && "$(realpath -e -- "$root/$path")" == "$root/$path" ]] || {
            printf 'Refusing missing or symlinked build input: %s\n' "$path" >&2
            exit 1
        }
        printf '%s\0' "$path" >> "$selected"
    fi
done < "$list"
required=(LICENSE THIRD_PARTY_NOTICES.md docs/LICENSES.md docs/MODELS.md licenses/manifest.json gpu/image_inventory.py pod/experiments/runpod-start.sh)
if [[ "$kind" == service ]]; then
    required+=(Dockerfile pyproject.toml MANIFEST.in docker-entrypoint.sh src/intro_detector/__init__.py)
else
    required+=(gpu/Dockerfile gpu/prefetch.py pod/tools/melband_backend.py pod/experiments/requirements-stem-comparison.txt pod/experiments/requirements-qwen.txt)
    required+=(gpu/requirements-build.lock gpu/requirements-base.lock gpu/requirements-stem.lock gpu/requirements-qwen.lock gpu/requirements-qwen-asr.lock)
    required+=(gpu/requirements-qwen-build.lock gpu/requirements-soxr-build.lock gpu/requirements-soxr.lock)
fi
for path in "${required[@]}"; do
    grep -Fxzq -- "$path" "$selected" || {
        printf 'Missing versioned build input: %s\n' "$path" >&2
        exit 1
    }
done
tar -C "$root" --format=gnu --sort=name --mtime="@${SOURCE_DATE_EPOCH:-0}" \
    --owner=0 --group=0 --numeric-owner --mode=u=rw,go=r \
    --null --verbatim-files-from --no-recursion -cf "$output" -T "$selected"