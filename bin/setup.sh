#!/usr/bin/env bash
set +x
set -euo pipefail
umask 077

usage='Usage: sudo bash bin/setup.sh [--azuracast-dir DIR] [--unattended]'
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
image=azuracast-intro-detector:local
log=$(mktemp "${TMPDIR:-/tmp}/intro-detector-setup.XXXXXX.log")
backup=""
registry_config=""
build_context=""
azuracast_dir=""
unattended=false
exec 3>&1 4>&2
exec >> "$log" 2>&1
status() { printf "$@" >&3; }
on_error() {
    printf 'Setup could not be completed. Technical details: %s\n' "$log" >&4
    if [[ -n "$backup" ]]; then
        printf 'Original settings are backed up in %s. The data volume was not removed.\n' "$backup" >&4
    fi
}
fail() { printf 'ERROR: %s\n' "$*" >&4; on_error; exit 1; }
cleanup() {
    result=$?
    if [[ -n "$registry_config" ]]; then rm -rf -- "$registry_config"; fi
    if [[ -n "$build_context" ]]; then rm -f -- "$build_context"; fi
    if [[ "$result" == 0 ]]; then rm -f "$log"; fi
}
trap on_error ERR
trap cleanup EXIT

while (($#)); do
    case "$1" in
        --azuracast-dir) [[ $# -ge 2 ]] || fail "$usage"; azuracast_dir=$2; shift 2 ;;
        --unattended) unattended=true; shift ;;
        *) fail "$usage" ;;
    esac
done
[[ "$EUID" == 0 ]] || fail 'Run this script with sudo, or from a root shell.'
[[ -z "${COMPOSE_FILE:-}" ]] || fail 'Custom COMPOSE_FILE requires manual installation.'
for path in .env config.toml config.example.toml compose.override.yaml secrets secrets/runpod-ssh-key \
    secrets/runpod-ssh-key.pub .setup-backups; do
    [[ ! -L "$root/$path" ]] || fail "Refusing a symlinked setup path: $path"
done
if [[ -f "$root/.env" ]] && grep -Eq '^[[:space:]]*(export[[:space:]]+)?COMPOSE_FILE[[:space:]]*=' "$root/.env"; then
    fail 'COMPOSE_FILE in .env requires manual installation.'
fi
plugin_dir=""
if [[ -n "$azuracast_dir" ]]; then
    azuracast_dir=$(cd "$azuracast_dir" && pwd -P) || fail "AzuraCast directory not found: $azuracast_dir"
    plugin_dir="$azuracast_dir/plugins/azuracast-sweepers"
    [[ -d "$plugin_dir" && ! -L "$plugin_dir/.detector.json" ]] || fail 'Install the Sweepers plugin first.'
fi
command -v docker >/dev/null || fail 'Install Docker Engine and Docker Compose first.'
docker info >/dev/null
docker compose version >/dev/null || fail 'Docker Compose v2 is required.'
cd "$root"
available=$(df -Pk "$root" | awk 'NR == 2 {print $4}')
((available >= 1048576)) || fail 'At least 1 GB of free disk space is required.'
if ((available < 5242880)); then status 'WARNING: Less than 5 GB free; make space before scanning a library.\n'; fi

status 'Building the detector service. This does not start a GPU worker...\n'
build_context=$(mktemp "${TMPDIR:-/tmp}/intro-detector-context.XXXXXX.tar")
bash bin/image-context.sh service "$build_context"
old_image=$(docker image inspect --format '{{.Id}}' "$image" 2>/dev/null || true)
docker build --tag "$image" - < "$build_context"
rm -f -- "$build_context"
build_context=""
worker_metadata=$(docker run --rm --user 0:0 --network none --entrypoint python \
    -v "$root:/setup" "$image" -m intro_detector.setup_config worker-image)
mapfile -t worker_settings <<< "$worker_metadata"
[[ ${#worker_settings[@]} == 2 ]] || fail 'Could not read the bundled worker image reference.'
if [[ "${worker_settings[1]}" == 0 ]]; then
    status 'Checking the repository-provided GPU worker image...\n'
    registry_config=$(mktemp -d "${TMPDIR:-/tmp}/intro-detector-registry.XXXXXX")
    docker --config "$registry_config" manifest inspect "${worker_settings[0]}" >/dev/null || \
        fail 'The worker image could not be reached anonymously. Check network access to GHCR and retry. If it is missing or private, contact the repository maintainers. Existing settings and credentials were not changed; no GPU worker was started.'
else
    status 'Keeping the custom private worker image from existing configuration.\n'
fi
metadata=$(docker run --rm --user 0:0 --network none --entrypoint python \
    -v "$root:/setup" "$image" -m intro_detector.setup_config inspect)
mapfile -t settings <<< "$metadata"
[[ ${#settings[@]} == 3 ]] || fail 'Could not read the existing settings.'
url="" azura_key=""
if $unattended; then
    # Plugin setup passes the AzuraCast address and API key, one per line; empty lines keep existing values.
    IFS= read -r url || true
    IFS= read -r azura_key || true
    configure=false
    if [[ ! -f config.toml || "${settings[1]}" != 1 || -n "$url" || -n "$azura_key" ]]; then configure=true; fi
    url="${url:-${settings[0]}}"
else
    configure=true
    if [[ -f config.toml && "${settings[1]}" == 1 ]]; then
        status 'Keep existing AzuraCast address, API key and all configuration? [Y/n] '
        read -r answer || fail 'Interactive confirmation is required.'
        if [[ "$answer" != n && "$answer" != N && "$answer" != no ]]; then configure=false; fi
    fi
    if $configure; then
        url="${settings[0]}"
        status 'AzuraCast HTTPS address%s: ' "${url:+ [$url]}"
        read -r answer || fail 'Enter the AzuraCast address.'
        url="${answer:-$url}"
        status 'AzuraCast API key%s: ' "$([[ "${settings[1]}" == 1 ]] && printf ' [Enter keeps existing]')"
        read -r -s azura_key || fail 'Enter the AzuraCast API key.'
        status '\n'
    fi
fi
if $configure; then
    [[ -n "$url" ]] || fail 'Enter the complete AzuraCast HTTPS address.'
    status 'Preparing settings...\n'
    mkdir -p .setup-backups
    backup=$(mktemp -d "$root/.setup-backups/setup.XXXXXX")
    for path in config.toml .env; do
        if [[ -f "$path" ]]; then cp -p "$path" "$backup/$path"; fi
    done
    printf '%s\0' "$url" "$azura_key" "" | \
        docker run --rm --interactive --user 0:0 --network none --entrypoint python \
        -v "$root:/setup" "$image" -m intro_detector.setup_config prepare
    unset azura_key
else
    docker run --rm --user 0:0 --network none --entrypoint python \
        -v "$root:/setup" "$image" -m intro_detector.setup_config token
fi
chmod 600 .env
chown root:10001 config.toml
chmod 640 config.toml
mkdir -p secrets .setup-backups
chmod 700 secrets .setup-backups
if [[ ! -f secrets/runpod-ssh-key ]]; then
    status 'Generating the worker SSH key...\n'
    docker run --rm --user 0:0 --network none --entrypoint ssh-keygen \
        -v "$root/secrets:/secrets" "$image" -t ed25519 -N '' \
        -f /secrets/runpod-ssh-key -C azuracast-intro-detector
fi
chown 10001:10001 secrets/runpod-ssh-key
chmod 600 secrets/runpod-ssh-key

web=""
if [[ -n "$azuracast_dir" ]]; then
    web=$(docker compose --project-directory "$azuracast_dir" ps -q web)
    [[ -n "$web" ]] || fail 'AzuraCast is not running; start it before installing the Intro Detector.'
    network=$(docker inspect --format '{{range $name, $_ := .NetworkSettings.Networks}}{{println $name}}{{end}}' "$web" | head -n 1)
    [[ "$network" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail 'Could not determine the AzuraCast Docker network.'
    # The plugin reaches the management API by this alias; no port is published on the host.
    printf 'services:\n  detector:\n    networks:\n      default: {}\n      azuracast:\n        aliases: [intro-detector]\nnetworks:\n  azuracast:\n    external: true\n    name: %s\n' \
        "$network" > compose.override.yaml
    chmod 644 compose.override.yaml
fi

status 'Checking configuration and API permissions...\n'
docker compose config --quiet
docker compose run --rm --no-deps detector check
status 'Starting/updating the detector...\n'
docker compose up -d --no-build --no-deps --force-recreate --wait --wait-timeout 90 detector
docker compose exec -T detector intro-detector status
if [[ -n "$plugin_dir" ]]; then
    web_group=$(docker exec "$web" id -g azuracast 2>/dev/null || true)
    [[ "$web_group" =~ ^[0-9]+$ ]] || web_group=1000
    docker run --rm --user 0:0 --network none --entrypoint python -v "$root:/setup" -v "$plugin_dir:/plugin" \
        "$image" -m intro_detector.setup_config link
    chown "0:$web_group" "$plugin_dir/.detector.json"
    chmod 640 "$plugin_dir/.detector.json"
    status 'Connected to the Sweepers plugin: manage it in Sweepers > Intro Detector.\n'
fi
# Housekeeping: the replaced service image, older settings backups and build cache beyond 1 GB.
new_image=$(docker image inspect --format '{{.Id}}' "$image")
if [[ -n "$old_image" && "$old_image" != "$new_image" ]]; then
    docker image rm "$old_image" >/dev/null 2>&1 || true
fi
find .setup-backups -mindepth 1 -maxdepth 1 -name 'setup.*' -printf '%T@ %p\0' 2>/dev/null | sort -z -rn | tail -z -n +6 | \
    cut -z -d' ' -f2- | xargs -0 -r rm -rf --
# Docker limits the build cache only for the whole host; AzuraCast pulls its images and builds nothing.
docker builder prune --force --max-used-space 1gb >/dev/null 2>&1 || \
    docker builder prune --force --keep-storage 1gb >/dev/null 2>&1 || true
status '\nDetector installed/updated. Existing settings and stored analyses are kept.\n'
status 'Stations and Runpod keys are managed in Sweepers > Intro Detector.\n'
