# GPU worker image: both Python environments and all pinned models baked in, so a new Pod
# starts without downloads.
ARG PYTHON_IMAGE=python:3.11-slim-bookworm@sha256:94a8fee35252d06ab75856f15f12c7bae67b10e8a6da93c7fded82b2610926cd
FROM ${PYTHON_IMAGE} AS build-base
ARG DEBIAN_SNAPSHOT=20261030T000000Z
ARG SOURCE_DATE_EPOCH=0
ENV SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH} PIP_DISABLE_PIP_VERSION_CHECK=1
RUN printf 'Types: deb deb-src\nURIs: https://snapshot.debian.org/archive/debian/%s/\nSuites: bookworm bookworm-updates\nComponents: main\nSigned-By: /usr/share/keyrings/debian-archive-keyring.gpg\nCheck-Valid-Until: no\n\nTypes: deb deb-src\nURIs: https://snapshot.debian.org/archive/debian-security/%s/\nSuites: bookworm-security\nComponents: main\nSigned-By: /usr/share/keyrings/debian-archive-keyring.gpg\nCheck-Valid-Until: no\n' \
    "$DEBIAN_SNAPSHOT" "$DEBIAN_SNAPSHOT" > /etc/apt/sources.list.d/debian.sources \
 && apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 libsoxr-dev wget openssh-server rsync build-essential pkg-config libgomp1 libgfortran5 libquadmath0 \
 && rm -rf /var/lib/apt/lists/*

FROM build-base AS ffmpeg-libraries
ARG FFMPEG_VERSION=8.1.2
ARG FFMPEG_SHA256=464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c
RUN wget -q -O /tmp/ffmpeg.tar.xz "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" \
 && printf '%s  /tmp/ffmpeg.tar.xz\n' "$FFMPEG_SHA256" | sha256sum -c - \
 && mkdir /tmp/ffmpeg-source \
 && tar -xf /tmp/ffmpeg.tar.xz --strip-components=1 -C /tmp/ffmpeg-source \
 && cd /tmp/ffmpeg-source \
 && ./configure --prefix=/opt/ffmpeg-lgpl --disable-static --enable-shared \
     --disable-gpl --disable-nonfree --disable-autodetect --disable-x86asm \
     --disable-doc --disable-programs --disable-debug --enable-pic \
 && make -j2 && make install \
 && mkdir -p /opt/ffmpeg-lgpl/legal \
 && cp COPYING.LGPLv2.1 LICENSE.md /opt/ffmpeg-lgpl/legal/ \
 && rm -rf /tmp/ffmpeg-source /tmp/ffmpeg.tar.xz

FROM build-base AS dependencies
COPY --from=ffmpeg-libraries /opt/ffmpeg-lgpl /opt/ffmpeg-lgpl
ENV PKG_CONFIG_PATH=/opt/ffmpeg-lgpl/lib/pkgconfig LD_LIBRARY_PATH=/opt/ffmpeg-lgpl/lib
COPY gpu/requirements-*.lock /app/requirements/
RUN pip install --no-cache-dir --require-hashes -r /app/requirements/requirements-build.lock \
 && pip install --no-cache-dir --no-build-isolation --require-hashes -r /app/requirements/requirements-base.lock \
 && pip install --no-cache-dir --no-build-isolation --require-hashes \
     -r /app/requirements/requirements-soxr-build.lock \
 && CMAKE_ARGS=-DUSE_SYSTEM_LIBSOXR=ON pip install --no-cache-dir --no-build-isolation \
     --require-hashes --no-binary=soxr --force-reinstall --no-deps \
     -r /app/requirements/requirements-soxr.lock
ENV HOME=/workspace/cache TORCH_HOME=/workspace/cache/torch POC_MODEL_CACHE=/workspace/cache PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MALLOC_ARENA_MAX=2
WORKDIR /app
COPY pod/tools/ /app/tools/
COPY pod/experiments/ /app/experiments/
RUN mkdir -p /run/sshd /root/.ssh && chmod 700 /root/.ssh

# Separate environments for the stem/Whisper stages and for Qwen3-ASR.
ENV HF_HOME=/workspace/cache/huggingface
RUN python -m venv --without-pip --system-site-packages /workspace/stem-comparison-venv \
 && /workspace/stem-comparison-venv/bin/python -m pip install --no-cache-dir \
     --no-build-isolation --require-hashes -r /app/requirements/requirements-stem.lock
RUN python -m venv --without-pip --system-site-packages /workspace/qwen-asr-venv \
 && /workspace/qwen-asr-venv/bin/python -m pip install --no-cache-dir --require-hashes \
    -r /app/requirements/requirements-qwen-build.lock \
 && /workspace/qwen-asr-venv/bin/python -m pip install --no-cache-dir --no-build-isolation --no-binary=av --require-hashes \
     -r /app/requirements/requirements-qwen.lock \
 && /workspace/qwen-asr-venv/bin/python -m pip install --no-cache-dir --no-deps --require-hashes \
     -r /app/requirements/requirements-qwen-asr.lock

FROM dependencies AS worker
COPY gpu/prefetch.py gpu/image_inventory.py /app/
RUN mkdir -p /usr/share/doc/intro-detector/inventory/runtime-replacements \
 && python /app/image_inventory.py --prune-unlisted-nvidia-libraries \
     --output /usr/share/doc/intro-detector/inventory/nvidia-pruned/base.json \
 && python /app/image_inventory.py --normalize-native-runtimes \
     --output /usr/share/doc/intro-detector/inventory/runtime-replacements/base.json \
 && /workspace/stem-comparison-venv/bin/python /app/image_inventory.py --normalize-native-runtimes \
     --output /usr/share/doc/intro-detector/inventory/runtime-replacements/stem.json \
 && /workspace/qwen-asr-venv/bin/python /app/image_inventory.py --normalize-native-runtimes \
     --output /usr/share/doc/intro-detector/inventory/runtime-replacements/qwen.json
RUN /workspace/stem-comparison-venv/bin/python /app/prefetch.py stem
RUN /workspace/qwen-asr-venv/bin/python /app/prefetch.py qwen
RUN python /app/image_inventory.py --output /usr/share/doc/intro-detector/inventory/models.json \
    --model-root whisper=/workspace/cache/whisper \
    --model-root demucs=/workspace/cache/torch/hub/checkpoints \
    --model-root melband=/workspace/cache/melband \
    --model-root panns=/workspace/cache/panns \
    --model-root panns-labels=/workspace/cache/panns_data \
    --model-root huggingface=/workspace/cache/huggingface/hub \
    --model-root smad=/workspace/experiments/whisper-evidence/model-cache
COPY LICENSE THIRD_PARTY_NOTICES.md docs/LICENSES.md docs/MODELS.md /usr/share/doc/intro-detector/
COPY licenses/ /usr/share/doc/intro-detector/licenses/
ARG CUSPARSELT_LICENSE_WHEEL_URL=https://files.pythonhosted.org/packages/3b/9a/72ef35b399b0e183bc2e8f6f558036922d453c4d8237dab26c666a04244b/nvidia_cusparselt_cu12-0.6.3-py3-none-manylinux2014_x86_64.whl
ARG CUSPARSELT_LICENSE_WHEEL_SHA256=e5c8a26c36445dd2e6812f1177978a24e2d37cacce7e090f297a688d1ec44f46
RUN wget -q -O /tmp/cusparselt-license.whl "$CUSPARSELT_LICENSE_WHEEL_URL" \
 && printf '%s  /tmp/cusparselt-license.whl\n' "$CUSPARSELT_LICENSE_WHEEL_SHA256" | sha256sum -c - \
 && python -c "import pathlib,zipfile; z=zipfile.ZipFile('/tmp/cusparselt-license.whl'); d=b'LICENSE AGREEMENT FOR NVIDIA SOFTWARE DEVELOPMENT KITS'; p=pathlib.Path('/usr/share/doc/intro-detector/licenses/nvidia-cusparselt-cu12/License.txt'); p.parent.mkdir(parents=True,exist_ok=True); t=z.read('cusparselt/LICENSE.txt'); assert t.startswith(d); p.write_bytes(t)" \
 && rm /tmp/cusparselt-license.whl
ARG CORRESPONDING_SOURCE_URL
RUN printf '%s\n' "${CORRESPONDING_SOURCE_URL:-Image redistribution has not been reviewed; see LICENSES.md.}" \
    > /usr/share/doc/intro-detector/CORRESPONDING_SOURCE
RUN python /app/image_inventory.py --debian --notices /usr/share/doc/intro-detector/licenses \
    --output /usr/share/doc/intro-detector/inventory/base \
 && /workspace/stem-comparison-venv/bin/python /app/image_inventory.py --output /usr/share/doc/intro-detector/inventory/stem \
 && /workspace/qwen-asr-venv/bin/python /app/image_inventory.py --output /usr/share/doc/intro-detector/inventory/qwen
RUN python /app/image_inventory.py --output /usr/share/doc/intro-detector/inventory/native \
    --native-prefix /opt/ffmpeg-lgpl --native-name ffmpeg --native-version 8.1.2 \
    --native-source-url https://ffmpeg.org/releases/ffmpeg-8.1.2.tar.xz \
    --native-source-sha256 464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c
# Everything is local now; fail instead of silently downloading another revision.
ENV HF_HUB_OFFLINE=1

ARG IMAGE_VERSION=unreleased
ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.source="https://github.com/RM-FM/azuracast-intro-detector" \
    org.opencontainers.image.revision="${SOURCE_REVISION}" \
    org.opencontainers.image.version="${IMAGE_VERSION}" \
    org.opencontainers.image.documentation="https://github.com/RM-FM/azuracast-intro-detector/blob/main/docs/LICENSES.md"

EXPOSE 22
ENTRYPOINT ["bash", "/app/experiments/runpod-start.sh"]
