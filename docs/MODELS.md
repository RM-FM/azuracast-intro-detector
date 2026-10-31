# Models and licenses

The GPU worker runs a first-30-second pipeline (`pod/`) with a per-recording ASR language.
All packages, binaries and models, their licenses and publication requirements are
listed in [LICENSES.md](LICENSES.md).

| Component | Pinned in the pipeline | Code license | Weights license | Notes |
| --- | --- | --- | --- | --- |
| Demucs `htdemucs` | demucs 4.0.1, profile `htdemucs-4.0.1-fp32-seg7-overlap025-shifts0-v1` | MIT | MIT (repository) | Repository archived |
| Mel-Band RoFormer | code `ZFTurbo/Music-Source-Separation-Training@9430afc`, weights `KimberleyJSN/melbandroformer@ac9b061` (SHA256 `87201f4d…559e`) | MIT | MIT (HF tag) | |
| Whisper turbo | `openai-whisper==20250625`; also detects each recording's language on its Mel-Band stem | MIT | MIT | |
| Qwen3-ASR-1.7B | revision `7278e1e70fe206f11671096ffdd38061171dd6e5` | Apache-2.0 | Apache-2.0 | `qwen-asr==0.0.6`; revision seeded by the service |
| Qwen3-ForcedAligner-0.6B | revision `c7cbfc2048c462b0d63a45797104fc9db3ad62b7` | Apache-2.0 | Apache-2.0 | |
| PANNs `Cnn14_DecisionLevelMax` | Zenodo record 3987831 | MIT (`audioset_tagging_cnn`), `panns-inference==0.1.1` | **CC-BY-4.0** | Attribution required; trained on AudioSet |
| SMAD singing/music CRNN | `duclvQ/smad@157a6316` | MIT | MIT (HF tag) | Trained on a synthetic research corpus (en/vi); loaded without Hub code |
| ExtraTrees selector | `scikit-learn==1.8.0` | BSD-3-Clause | own | Trained on 39 listener-labelled songs (`src/intro_detector/data/`, features and labels only) |
| Musical intro timing | `percussion-energy-30s-v1` | own | – | No neural model |
| Runtime | `torch==2.5.1` (CUDA 12.1), ffmpeg (Debian), `runpodctl` 3.0.1 | BSD / NVIDIA EULA / GPL-2.0+ (Debian build) / GPL-3.0 | – | All weights are baked into the GPU image |

PANNs weights (CC-BY-4.0) need attribution, and the GPL/LGPL components need
license texts and complete corresponding source for the actual distributed builds.
NVIDIA redistribution conditions and LGPL replacement rights must also be reviewed
before a model-baked image is published.
