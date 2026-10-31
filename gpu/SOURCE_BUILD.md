# Worker Corresponding Source

The source bundle is tied to the image's package inventory, source revision and
Docker image ID. Its `source-manifest.json` records the exact source files and
SHA256 values. Do not substitute a newer source release for an inventoried binary.

## Rebuild the Worker

Use Linux x86_64 with Docker BuildKit and sufficient free disk space. No GPU is
needed for package installation, model download or offline inventory checks.

```bash
bash bin/image-context.sh worker worker-context.tar
docker build -f gpu/Dockerfile -t intro-detector-worker:local - < worker-context.tar
```

The Dockerfile fixes the Python base digest and Debian snapshot. Python dependencies
are installed from the hash-locked requirements in `gpu/`. Model coordinates are
fixed in `gpu/prefetch.py`; the notice manifest records their license provenance.
The default entrypoint starts the worker environment. Do not start it to validate
licenses or use a paid GPU merely to rebuild packages.

## Debian Components

For each source package, the bundle contains its exact `.dsc`, upstream archive and
Debian patch/build archive. Preserve the complete set referenced by the `.dsc`.
Each referenced archive is checked against the exact `.dsc` SHA256 and size, and
the source package/version must match the installed image inventory. Sources are
retrieved from the pinned Debian snapshot configured in `gpu/Dockerfile`.
Use `dpkg-source -x PACKAGE.dsc` to unpack it; build with Debian's documented
`dpkg-buildpackage` workflow and the package's declared build dependencies.

`gpu/fetch_debian_sources.py` downloads the exact source/version pairs from the
image inventory. It uses the Dockerfile's pinned snapshot and falls back to the
matching snapshot pool descriptor when a base-image source is absent from the APT
source index. `gpu/fetch_native_sources.py` downloads the hash-locked PyPI and
upstream archives and recreates the pinned Git archives.

```bash
python3 gpu/fetch_native_sources.py --output source-material/native
python3 gpu/fetch_debian_sources.py \
	--inventory image-legal/inventory/base/debian.tsv \
	--output source-material/debian --snapshot 20261030T000000Z
python3 gpu/build_source_bundle.py \
	--inventory image-legal --source-material source-material \
	--repository . --worker-context worker-context.tar \
	--image-id sha256:... --revision COMMIT --output corresponding-source.tar.gz
```

GNU and audio-library runtime replacements use the installed Debian binaries in
all three Python environments. Exact source package/version and before/after hashes
are recorded per environment under `inventory/runtime-replacements/`. The matching
Debian source and build material are provided; GNU runtime exceptions and warranty
notices remain applicable.

## LGPL Replacement

FFmpeg shared libraries are built from the source archive identified by the native
inventory. The configure flags and install path are in `gpu/Dockerfile`. Rebuild
modified libraries with the same public ABI and install them under
`/opt/ffmpeg-lgpl/lib` in a derived image. PyAV is rebuilt from its matching source
distribution with `PKG_CONFIG_PATH` pointing at that installation; Cython and build
tools use the supplied locks.

For LAME, use the matching `lameenc` wrapper source, its CMake/build/patch files and
the native LAME archive. Build a modified native library and rebuild the Python
extension against it. SoundFile's bundled libsndfile is replaced by the Debian
shared library; its exact Debian source package and version are in the inventory.
SoXR is rebuilt from its hash-locked source with
`CMAKE_ARGS=-DUSE_SYSTEM_LIBSOXR=ON`, linking dynamically to Debian's libsoxr.
Include the matching binding source, the exact libsoxr submodule commit, and Debian
build/source material. Pillow's FriBidi source and corresponding build material are
also included where that system library is used. Retain upstream licenses and
notices. No detector terms restrict modification, replacement or debugging rights
granted by the applicable LGPL.

## Separate NVIDIA Terms

NVIDIA libraries retain their own shipped licenses. They are not relicensed under
the detector's MIT license or GNU licenses. The source bundle does not represent
unavailable proprietary NVIDIA source as open source. Preserve the applicable
redistribution conditions and third-party notices when distributing a derived image.

See [docs/LICENSES.md](../docs/LICENSES.md) for the version-specific inventory and
redistribution obligations. These build directions are not a legal certification.