# Third-party licenses

License inventory of every package, upstream and ML model used by the detector, against
the pinned versions in `Dockerfile`, `gpu/Dockerfile`, `gpu/prefetch.py`,
`pod/experiments/requirements-*.txt` and the backend download URLs in `pod/tools/`.
Sources: GitHub, Hugging Face, PyPI and Zenodo. This is an engineering inventory,
not legal advice.

Use case assessed: running the detector for a radio station, commercially (no resale)
or non-commercially. Both uses are permitted subject to each component's terms,
so the commercial and non-commercial columns are identical.

Legend: ✅ allowed · ⚠️ allowed with obligations or residual risk

## ML models (production Pod pipeline)

| Component | Code license | Weights license | Commercial | Non-commercial | Notes |
| --- | --- | --- | --- | --- | --- |
| Demucs `htdemucs` (`demucs==4.0.1`) | MIT | MIT | ✅ (⚠️) | ✅ | Repository archived. Residual risk: trained partly on MUSDB18-HQ, a research-only dataset |
| Mel-Band RoFormer code, `ZFTurbo/Music-Source-Separation-Training@9430afc` | MIT | – | ✅ | ✅ | Pinned MIT source with its license file, verified at download |
| Mel-Band RoFormer weights, `KimberleyJSN/melbandroformer@ac9b061` | – | MIT (HF card) | ✅ | ✅ | Training data undocumented |
| Whisper turbo (`openai-whisper==20250625`) | MIT | MIT | ✅ | ✅ | Trained on web data |
| Qwen3-ASR-1.7B `@7278e1e` (`qwen-asr==0.0.6`) | Apache-2.0 | Apache-2.0 | ✅ | ✅ | Ship license and NOTICE when redistributing |
| Qwen3-ForcedAligner-0.6B `@c7cbfc2` | Apache-2.0 | Apache-2.0 | ✅ | ✅ | As above |
| PANNs `Cnn14_DecisionLevelMax` (Zenodo record 3987831) | MIT (`audioset_tagging_cnn`, `panns-inference`) | **CC-BY-4.0** | ⚠️ | ⚠️ | Attribution required when the weights are redistributed, see below |
| SMAD CRNN `duclvQ/smad@157a631` | MIT (published on HF only; no GitHub repo) | MIT | ✅ | ✅ | Synthetic research corpus (en/vi); loaded without Hub code |
| Spleeter 2-stem, sherpa-onnx FP16 export | MIT (Deezer) / Apache-2.0 (sherpa-onnx) | MIT | ✅ | ✅ | Installed in the image, not called by the production stages |
| ExtraTrees selector and `src/intro_detector/data/selector-training-v1.json.gz` | own | own | ✅ | ✅ | Only SHA256 values, numeric features and labels: no audio, transcripts or lyrics |
| Musical intro timing `percussion-energy-30s-v1` | own | – | ✅ | ✅ | No neural model |

## Python packages

| Packages | License | Commercial / non-commercial |
| --- | --- | --- |
| numpy, scikit-learn 1.8.0, scipy, joblib, threadpoolctl, torch 2.5.1, torchaudio 2.5.1, soundfile, numba, flask, sox (pysox), python-dotenv | BSD | ✅ / ✅ |
| demucs, dora-search, julius, openunmix, einops, beartype, rotary-embedding-torch, PyYAML, panns-inference, torchlibrosa, audioread, tiktoken, triton, nagisa, pytz, onnxruntime, tomlkit | MIT | ✅ / ✅ |
| librosa | ISC | ✅ / ✅ |
| qwen-asr, transformers, accelerate, huggingface-hub, safetensors, sherpa-onnx, qwen-omni-utils, gradio | Apache-2.0 | ✅ / ✅ |
| lameenc (via demucs, statically linked to LAME) | LGPL-3.0-or-later | ✅ / ✅; ⚠️ on redistribution and replacement material |
| soxr (via librosa; extension dynamically links to Debian libsoxr0) | LGPL-2.1-or-later | ✅ / ✅; ⚠️ on redistribution and replacement material |
| libsndfile (SoundFile's bundled copy is replaced by Debian libsndfile1) | LGPL-2.1-or-later | ✅ / ✅; ⚠️ on redistribution and replacement material |
| **soynlp** 0.0.493 (upstream qwen-asr dependency) | **GPL-3.0** | Not installed: qwen-asr is installed with `--no-deps`; Korean alignment uses whitespace word splitting instead |
| NVIDIA CUDA libraries (pulled in by the `cu121` torch wheels) | NVIDIA EULA | ✅ / ✅; redistribution only as the EULA permits |
| diffq | CC-BY-**NC**-4.0 | **Not installed:** demucs 4.0.1 installs only `requirements_minimal.txt`. Do not add it |

## Binaries, base images and services

| Component | License | Commercial / non-commercial |
| --- | --- | --- |
| Python 3.11 / 3.12 | PSF | ✅ / ✅ |
| Debian bookworm base images | mixed DFSG-free | ✅ / ✅ |
| ffmpeg (Debian build with `--enable-gpl`) | GPL-2.0-or-later | ✅ / ✅; run as a separate program. ⚠️ GPL obligations in a public image |
| openssh, rsync, curl | BSD / GPL-3.0 / curl | ✅ / ✅; separate programs |
| `runpodctl` 3.0.1 | **GPL-3.0** | ✅ / ✅; separate program. ⚠️ GPL obligations in a public image |
| Runpod | Terms of service, not a software license | ✅ |
| AzuraCast (AGPL-3.0) | Used only through its HTTP API | No AzuraCast source is included or linked into the detector |

## Included legal documents

[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) identifies the model publishers
and required attributions. [licenses/manifest.json](../licenses/manifest.json)
records the origin and checksum of each verbatim upstream notice and model-card
declaration. Do not edit or reformat those files without updating their provenance.

Both Docker images include the notice bundle and this guide under
`/usr/share/doc/intro-detector/`. Python source and wheel distributions also carry
the legal documents. Package-specific copyright, NOTICE, EULA and author files
are copied from the installed distributions; Debian copyright files and common
licenses are retained separately.

The Mel-Band loader uses pinned MIT MSST code, based on Phil Wang's implementation.
Both upstream MIT notices are included. Its separate model weights and SMAD carry
MIT declarations in their pinned model cards. Those cards supply no separate
copyright notice; the detector does not invent one. Qwen's pinned cards declare
Apache-2.0. An absent model-repository LICENSE file is not replaced by a claim that
the inference-code license covers unrelated weights.

## Obligations when publishing

Publishing source only does not itself redistribute remotely downloaded weights.
It still requires permission for all third-party code actually included in the
repository and must not be presented as clearance for downloading unlicensed
weights. Notices alone do not cure missing permission.

Publishing an image with the models baked in additionally requires:

- **PANNs attribution (CC-BY-4.0):** Kong, Cao, Iqbal, Wang, Wang, Plumbley,
  "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern
  Recognition (Pretrained Models)", doi:10.5281/zenodo.3987831, licensed
  CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). State that the
  weights are unmodified.
- **MIT, BSD, ISC and Apache-2.0 components:** include their license texts and
  copyright notices. For Apache-2.0 components, also include their NOTICE files
  where these exist.
- **LGPL components** (including lameenc, libsoxr and libsndfile): preserve their
  licenses, provide corresponding source as required by the applicable LGPL
  version, and preserve the recipient's modification and replacement rights.
  A Python package source archive may omit the native library bundled in its wheel.
- **GPL components** (including Debian ffmpeg/rsync and the service's runpodctl):
  provide complete corresponding source for the actual distributed builds, not
  merely a link to a repository or a version number.
- **NVIDIA CUDA components:** retain their actual shipped license texts and comply
  with each component's redistribution conditions. A PyPI package's availability
  does not establish that every included file is redistributable.

## Corresponding-source bundle

### Worker native libraries

The worker builds FFmpeg `8.1.2` shared libraries from the official release archive
with `--disable-gpl`, `--disable-nonfree` and `--disable-autodetect`. PyAV is built
from its hash-locked source distribution against these libraries, not installed
from a wheel containing additional codecs. The separate Debian FFmpeg executable
retains its Debian license and corresponding-source obligations.

The library archive SHA256 is
`464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c`.
Ship that archive, the configure/build instructions in `gpu/Dockerfile`, PyAV's
matching source archive and the upstream licenses with the corresponding-source
bundle. The native inventory records both actual library hashes and this source
archive hash.

GNU `libgomp`, `libgfortran` and `libquadmath` copies bundled inside Python wheels
are replaced by the installed Debian versions in each Python environment.
Per-environment replacement records list modified package paths, original/replacement
hashes and exact Debian binary/source versions. Preserve the GNU licenses and runtime
exceptions; include the matching GCC sources. This does not change the NVIDIA libraries or
relicense them under the GPL. Modified wheel records are updated to match the
distributed bytes. Libraries remain replaceable in a derived image.

LAME bindings include LAME `4.0`; their source delivery includes the `lameenc 1.9.0`
build/patch files and native source archive matching the upstream build hash
`3df5124d5ad3a98312ffd7ba6a9b36230e4f8a3e66d3ce0f425e336c32d216eb`.
SoundFile's bundled libsndfile is replaced by Debian `libsndfile1` version
`1.2.0-1+deb12u1` from source package `libsndfile`. SoXR is built with
`USE_SYSTEM_LIBSOXR=ON`, so its extension links to Debian `libsoxr0` version
`0.1.3-4` from source package `libsoxr`; its `soxr 1.1.0` Python source is also
included. Pillow uses the Debian FriBidi `1.0.8-2.1` runtime from source package
`fribidi`. The GNU runtime replacements use GCC source version
`12.2.0-14+deb12u1`. The inventory and Debian source verifier tie each binary to
its exact source version.

Triton `3.1.0` declares MIT in its pinned upstream
[license](https://github.com/triton-lang/triton/blob/v3.1.0/LICENSE); Tokenizers
`0.22.2` declares Apache-2.0 in its pinned upstream
[license](https://github.com/huggingface/tokenizers/blob/v0.22.2/LICENSE). The
corresponding canonical license texts are included in `licenses/`.

### NVIDIA redistribution scope

The pinned PyTorch stack includes CUDA `12.1` components, cuDNN `9.1.0.70` and
nvJitLink `12.9.86`. The candidate inventory also lists the precise files in the
CUPTI and NCCL Python distributions. Apply each shipped component's terms rather
than assuming a single toolkit version covers every library. The relevant archived agreements are
[CUDA 12.1](https://docs.nvidia.com/cuda/archive/12.1.0/eula/index.html),
[cuDNN 9.1](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.1.0/reference/eula.html)
and [CUDA 12.9](https://docs.nvidia.com/cuda/archive/12.9.1/eula/index.html).

These agreements provide redistribution rights subject to their conditions; no
blanket individual approval is implied. Check the actual installed files against
the redistributable portions and applicable component notices. Ship unchanged
NVIDIA object code as part of this application's materially additional functionality,
not as a standalone SDK. Retain licenses, copyright and third-party notices, make
the applicable terms available to recipients, and do not claim NVIDIA endorsement.
Do not include driver or developer-tool binaries without checking their specific
conditions. Export restrictions and any applicable codec patent obligations remain
separate; this engineering inventory is not legal advice.

The versioned [NVIDIA redistribution record](NVIDIA_REDISTRIBUTION.md) records the
review of the CUPTI helpers, NCCL notice, retained CUDA libraries and cuSPARSELt
supplement. Its current decision is `APPROVED`; every release still exports the
exact candidate inventory and requires the maintainer's explicit review marker.
Do not treat this technical record as legal advice or as approval of a different
image, package set or source revision.

Before publishing either image, assemble and retain a source bundle tied to its
exported package inventory and image digest:

1. Include the exact Debian source packages, with `.dsc`, upstream archives and
   Debian patches/build files, for the applicable copyleft binaries and libraries.
   Use the inventory's source package and source version, not just its binary name.
2. For runpodctl, include the source matching the released executable and the
   necessary Go dependency sources and build scripts. A tag archive without its
   required dependencies is not automatically complete corresponding source.
3. Include the exact native-library sources and needed build/relinking material
   for LGPL libraries shipped inside Python wheels. Check wheel contents rather
   than assuming the matching Python sdist contains these libraries.
4. Include any modifications and the scripts needed to generate and install the
   covered works. If distribution in a User Product triggers installation-information
   requirements, supply that information too.
5. Make the bundle freely accessible at a durable public location, with clear
   directions beside the image download. Keep it available for the period required
   by the chosen license-compliance method; an expiring CI artifact is not sufficient.

No additional detector terms restrict the LGPL modification, replacement or
debugging rights. The libraries must remain replaceable in a derived image; do not
add technical or contractual restrictions that defeat those rights. Preserve all
applicable warranty disclaimers and do not imply endorsement by upstream authors.

Pull requests to `develop` and `main` run the Linux unit tests. Successful branch
pushes build a commit-specific Candidate in the private
`azuracast-intro-detector-gpu-candidate` package; they never publish the worker
image.

Publishing is triggered by publishing a GitHub Release whose tag is `vMAJOR.MINOR.PATCH`,
matches `pyproject.toml`, and points to a commit on `main`. The release body must
contain the exact marker `REDISTRIBUTION REVIEWED: APPROVED`. This marker records a
maintainer's manual review of exact-source completeness, LGPL replacement rights,
NVIDIA terms and the model notices; it is not an automatic legal certification.
The release workflow pins the Debian snapshot, downloads hash-locked Native sources,
retrieves every Debian source/version from the image inventory, and verifies all
archives. It then builds the corresponding-source bundle from the exact image
inventory, image ID and source revision, verifies its manifest, and attaches it as
`corresponding-source.tar.gz` to the release. Only after that succeeds does it push
the immutable versioned GHCR image, switch that package to public, and use a fresh
runner with no GHCR credentials to pull the image by tag and digest and verify the
source asset checksum. The release also promotes the checked image to `v1`.

If any test, source, review, image or anonymous-pull check fails, the public image
push/promotion does not complete. The GitHub Release itself is the explicit human
release action; branch pushes alone never make an image public.

For a manual Docker build, set `--build-arg CORRESPONDING_SOURCE_URL=https://...` after
completing that review. The URL is recorded inside the image. An empty URL records
that redistribution has not been reviewed.

### Source manifest format

The source archive has a root `source-manifest.json` with `schema: 1`, `image_id`
(the reviewed Docker image ID), `revision` (the full source commit),
`build_instructions` and a `components` array. A file record contains `path` relative
to the archive root and its `sha256`. Referenced files must be non-empty regular
files; the verifier does not extract the archive.

Each component records `id`, `version`, `license`, an HTTPS `origin`,
`source_required`, `files` (source file records) and `native_libraries` copied from
the exact inventory, including their installed paths and SHA256 values. Use IDs
`python/ENVIRONMENT/NAME@VERSION` and `debian/BINARY@VERSION`; Debian entries also
record the inventory's `source` and `source_version`. Every inventoried component
must be covered exactly once. A component not requiring source records its
`source_not_required_reason`. GPL/LGPL components require corresponding-source
file records; LGPL entries additionally reference `replacement_instructions`.
NVIDIA entries reference `redistribution_basis`: a file documenting the applicable
version-specific terms, permitted redistribution and obligations actually fulfilled.
`build_instructions` is also a checksummed file record.

The automated check verifies coverage, identity and material checksums, not whether
an archive is legally complete or rebuilds correctly. Review every native library's
own license and actual version, including dependencies bundled inside wheels;
the parent Python distribution's license is not a substitute. Models and their
attributions need their separate pinned provenance and notice review. A new binary,
model, dependency or source revision invalidates the previous image review.

## Remaining Checks

- `gpu/image_inventory.py` records exact Python distributions, requirements,
  shipped license/NOTICE/EULA files, native-library hashes, and optionally Debian binary/source versions
  and copyright files. GPU builds keep inventories for the base and both
  environments under `/usr/share/doc/intro-detector/inventory/`. Exact GPL/LGPL
  corresponding-source delivery must be verified before an image is published.
- Source-bundle completeness, native-library replacement material and the NVIDIA
  redistribution review are publication gates for each actual image build.
- Training-data provenance is a separate residual risk for Demucs (MUSDB18-HQ), Whisper
  (web data), Mel-Band RoFormer (undocumented) and SMAD (synthetic research
  corpus). A model license does not by itself resolve training-data claims.
- Spleeter/sherpa-onnx are installed in the GPU image but not used by the pipeline.
