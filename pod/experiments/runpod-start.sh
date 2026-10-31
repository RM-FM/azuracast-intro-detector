#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$HOME" /root/.ssh /run/sshd
# Runpod PUBLIC_KEY contains the selected SSH public key, never a private key.
if [[ -n "${PUBLIC_KEY:-}" ]]; then
  printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi
ssh-keygen -A
exec /usr/sbin/sshd -D -e -o PasswordAuthentication=no -o PermitRootLogin=prohibit-password -o AuthorizedKeysFile=/root/.ssh/authorized_keys
