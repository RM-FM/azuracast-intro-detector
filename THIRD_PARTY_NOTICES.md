# Third-party notices

The MIT license in [LICENSE](LICENSE) covers RM-FM's own code. It does not relicense
third-party software, model weights, datasets or service terms. Components retain
their own licenses, copyright notices and warranty disclaimers. Not every listed
component is included in the Python package; models are downloaded for the GPU worker.

Verbatim license texts, upstream copyright notices and pinned model-card declarations
are in [licenses/manifest.json](licenses/manifest.json). The manifest records each
document's source and SHA256. The full inventory and redistribution requirements are
in [docs/LICENSES.md](docs/LICENSES.md).

## Mel-Band RoFormer inference

Pinned upstream: ZFTurbo/Music-Source-Separation-Training,
`9430afc97432ba22cdae8171a0567759c229df82`, based on lucidrains/BS-RoFormer.
The loader downloads and verifies MSST's LICENSE alongside its source.
The detector adapter selects attention kernels at runtime.
The upstream files themselves are unmodified.

Copyright (c) 2024 Roman Solovyev (ZFTurbo). Copyright (c) 2023 Phil Wang.
Their individual, unmodified MIT notices are preserved in
[MSST's license](licenses/upstream/msst-LICENSE.txt) and
[the original implementation's license](licenses/upstream/bs-roformer-LICENSE.txt).

The weights are from KimberleyJSN/melbandroformer, revision
`ac9b0614ab3cd7f77219e18ba494dfd93956c348`. The publisher declares MIT in the
[model card](licenses/model-cards/melbandroformer.md); no separate copyright notice
is supplied there. The corresponding [MIT terms](licenses/MIT.txt) are included.

## Other pretrained models

- Demucs: Copyright (c) Meta Platforms, Inc. and affiliates. The code and weights
	use the [upstream MIT license](licenses/upstream/demucs-LICENSE.txt).
- Whisper: the [upstream MIT notice](licenses/upstream/whisper-LICENSE.txt)
	preserves OpenAI's copyright and applies to its code and weights.
- Qwen3-ASR-1.7B and Qwen3-ForcedAligner-0.6B: the publisher declares Apache-2.0
	in the pinned [ASR](licenses/model-cards/qwen-asr.md) and
	[aligner](licenses/model-cards/qwen-aligner.md) cards.
	The full [Apache-2.0 terms](licenses/Apache-2.0.txt) are included.
- SMAD: duclvQ/smad, revision `157a6316366bcf8831baf23df742edb71f07eb47`,
	declares MIT in its [model card](licenses/model-cards/smad.md).

These model weights are distributed unmodified. No licensor endorsement is implied.
An upstream request for a research citation is not presented as an additional
license restriction.

## PANNs weights

Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang and Mark D. Plumbley,
*PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern
Recognition (Pretrained Models)*, Zenodo record 3987831:
https://doi.org/10.5281/zenodo.3987831.

The `Cnn14_DecisionLevelMax` checkpoint is distributed unmodified under
Creative Commons Attribution 4.0 International:
https://creativecommons.org/licenses/by/4.0/.
The full [CC-BY-4.0 legal text](licenses/CC-BY-4.0.txt) is included.
Attribution does not imply endorsement.

## Packages and binaries

Each image includes its exact Python and Debian package inventory, copied upstream
license/NOTICE/EULA files and Debian common license texts under
`/usr/share/doc/intro-detector/inventory/`. The versioned model notices are under
`/usr/share/doc/intro-detector/licenses/`.

The service image contains the unmodified `runpodctl` 3.0.1 executable as a separate
program under [GPL-3.0](licenses/GPL-3.0.txt). This does not relicense the detector's
independent MIT code. The executable and its dependencies require their own notices
and complete corresponding source when the image is redistributed.

The worker's FFmpeg, native libraries and NVIDIA components retain their own terms.
The detector imposes no additional restriction on modification, replacement or
reverse engineering for debugging such modifications where the LGPL permits them.
SoundFile's bundled libsndfile is replaced with the Debian shared library; SoXR is
rebuilt to link to Debian's libsoxr shared library. GNU runtimes bundled in wheels
are replaced with their exact Debian runtime packages. The worker inventory records
the replacement hashes and Debian source versions.

PyTorch's CUDA wheel includes the cuSPARSELt runtime `libcusparseLt`. Its NVIDIA
license supplement and runtime redistribution terms are included in the worker
image under `/usr/share/doc/intro-detector/licenses/nvidia-cusparselt-cu12/`.

The Worker dependency set also includes Triton `3.1.0` under MIT, as declared by its
[pinned upstream license](https://github.com/triton-lang/triton/blob/v3.1.0/LICENSE),
and Tokenizers `0.22.2` under Apache-2.0, as declared by its
[pinned upstream license](https://github.com/huggingface/tokenizers/blob/v0.22.2/LICENSE).
Their canonical license texts are included above with the other MIT and Apache-2.0
notices.

## Image redistribution

The image includes separately licensed MIT, BSD, ISC, Apache-2.0, PSF, LGPL,
GPL and NVIDIA components. Each package's own copyright notices, license
texts and applicable NOTICE files must accompany a redistributed image.
The image inventory collects the installed Python license files and Debian
copyright files; it is an inventory, not a legal clearance.

GPL/LGPL source, exact-version availability, modifications, applicable
relinking/installation requirements and NVIDIA redistribution terms require
separate review before public image distribution. Merely linking to a generic
upstream repository or adding this notice is not necessarily sufficient.
The public source location for a reviewed build is recorded in
`/usr/share/doc/intro-detector/CORRESPONDING_SOURCE`. It must also be displayed beside
the image download. Notices and a URL alone are not a complete source offering.
An image without a completed exact-build review must not be described as cleared.
