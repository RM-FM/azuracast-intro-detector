FROM python:3.12-slim-bookworm

ARG CORRESPONDING_SOURCE_URL
ARG RUNPODCTL_VERSION=3.0.1
ARG RUNPODCTL_SHA256=b66d11eca8cacbf18a2de6943c6e18a79ffbdc8a3c30aa13a49dc6cfc15d4239
RUN apt-get update \
 && apt-get install -y --no-install-recommends openssh-client rsync ca-certificates curl \
 && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL -o /usr/local/bin/runpodctl \
    "https://github.com/runpod/runpodctl/releases/download/v${RUNPODCTL_VERSION}/runpodctl-linux-amd64" \
 && echo "${RUNPODCTL_SHA256}  /usr/local/bin/runpodctl" | sha256sum -c - \
 && chmod 755 /usr/local/bin/runpodctl

WORKDIR /app
COPY pyproject.toml ./
COPY LICENSE THIRD_PARTY_NOTICES.md ./
COPY MANIFEST.in ./
COPY licenses/ licenses/
COPY docs/LICENSES.md docs/MODELS.md docs/
COPY src/ src/
RUN pip install --no-cache-dir .
COPY pod/ pod/
COPY gpu/run-lanes.sh gpu/lanes_status.py gpu/
COPY gpu/image_inventory.py gpu/
COPY LICENSE THIRD_PARTY_NOTICES.md docs/LICENSES.md docs/MODELS.md /usr/share/doc/intro-detector/
COPY licenses/ /usr/share/doc/intro-detector/licenses/
RUN printf '%s\n' "${CORRESPONDING_SOURCE_URL:-Image redistribution has not been reviewed; see LICENSES.md.}" \
    > /usr/share/doc/intro-detector/CORRESPONDING_SOURCE
RUN python gpu/image_inventory.py --debian --notices /usr/share/doc/intro-detector/licenses \
    --output /usr/share/doc/intro-detector/inventory/service
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN chmod 755 /usr/local/bin/docker-entrypoint.sh \
 && useradd --uid 10001 --create-home detector && mkdir -p /data && chown detector /data
USER detector
ENV INTRO_DETECTOR_POD_SOURCE=/app/pod \
    INTRO_DETECTOR_CONFIG=/config/config.toml \
    PYTHONUNBUFFERED=1
VOLUME /data
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["run"]
