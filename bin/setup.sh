#!/usr/bin/env bash
set +x
set -euo pipefail
umask 077

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
image=azuracast-intro-detector:local
log=$(mktemp "${TMPDIR:-/tmp}/intro-detector-setup.XXXXXX.log")
backup=""
registry_config=""
build_context=""
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

[[ $# == 0 ]] || fail 'Usage: sudo bash bin/setup.sh'
[[ "$EUID" == 0 ]] || fail 'Run this script with sudo, or from a root shell.'
[[ -z "${COMPOSE_FILE:-}" ]] || fail 'Custom COMPOSE_FILE requires manual installation.'
for path in .env config.toml config.example.toml secrets secrets/runpod-ssh-key secrets/runpod-ssh-key.pub .setup-backups; do
    [[ ! -L "$root/$path" ]] || fail "Refusing a symlinked setup path: $path"
done
if [[ -f "$root/.env" ]] && grep -Eq '^[[:space:]]*(export[[:space:]]+)?COMPOSE_FILE[[:space:]]*=' "$root/.env"; then
    fail 'COMPOSE_FILE in .env requires manual installation.'
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
configure=true
if [[ -f config.toml && "${settings[1]}" == 1 && "${settings[2]}" == 1 ]]; then
    status 'Keep existing host, API keys and all configuration? [Y/n] '
    read -r answer || fail 'Interactive confirmation is required.'
    if [[ "$answer" != n && "$answer" != N && "$answer" != no ]]; then configure=false; fi
fi
if $configure; then
    url="${settings[0]}"
    status 'AzuraCast HTTPS address%s: ' "${url:+ [$url]}"
    read -r answer || fail 'Enter the AzuraCast address.'
    url="${answer:-$url}"
    [[ -n "$url" ]] || fail 'Enter the complete AzuraCast HTTPS address.'
    status 'AzuraCast API key%s: ' "$([[ "${settings[1]}" == 1 ]] && printf ' [Enter keeps existing]')"
    read -r -s azura_key || fail 'Enter the AzuraCast API key.'
    status '\nRunpod API key%s: ' "$([[ "${settings[2]}" == 1 ]] && printf ' [Enter keeps existing]')"
    read -r -s runpod_key || fail 'Enter the Runpod API key.'
    status '\nPreparing settings...\n'
    mkdir -p .setup-backups
    backup=$(mktemp -d "$root/.setup-backups/setup.XXXXXX")
    for path in config.toml .env; do
        if [[ -f "$path" ]]; then cp -p "$path" "$backup/$path"; fi
    done
    printf '%s\0' "$url" "$azura_key" "$runpod_key" | \
        docker run --rm --interactive --user 0:0 --network none --entrypoint python \
        -v "$root:/setup" "$image" -m intro_detector.setup_config prepare
    unset azura_key runpod_key
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

status 'Checking configuration and API permissions...\n'
docker compose config --quiet
docker compose run --rm --no-deps detector check
status 'Starting/updating the detector. AzuraCast broadcasting is not restarted...\n'
docker compose up -d --no-build --no-deps --force-recreate --wait --wait-timeout 90 detector
docker compose exec -T detector intro-detector status
status '\nDetector installed/updated. Existing settings and stored analyses are kept.\n'
status 'First installation uses shadow mode and manual processing.\n'
status 'Follow First station setup in README.md for a controlled sample scan.\n'