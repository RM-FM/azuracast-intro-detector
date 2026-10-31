# NVIDIA CUDA Redistribution Review

REVIEW STATUS: TECHNICAL REVIEW COMPLETE; CANDIDATE VERIFIED

APPLICATION CONDITIONS: TECHNICALLY VERIFIED AGAINST WORKER DESIGN

REDISTRIBUTION DECISION: APPROVED

This inventory records the NVIDIA native files in the pinned Worker package set.
A package license label alone does not establish that every file in a wheel may be
redistributed. Do not publish a Worker image until the exact candidate inventory
confirms these findings and the owner changes the decision line to
`REDISTRIBUTION DECISION: APPROVED`.

## Observed Package Versions

- CUDA runtime packages: 12.1.105
- cuBLAS: 12.1.3.1
- cuDNN: 9.1.0.70
- cuFFT: 11.0.2.54
- cuRAND: 10.3.2.106
- cuSOLVER: 11.4.5.107
- cuSPARSE: 12.1.0.106
- cuSPARSELt embedded in Torch: runtime API version 502 (0.5.2)
- NCCL: 2.21.5
- nvJitLink: 12.9.86
- NVTX: 12.1.105

The full file list and SHA256 values are in the exact per-environment inventory
exported with the candidate image.

## Terms Reviewed

The pinned CUDA package notices refer to the NVIDIA CUDA EULA. Its distribution
grant applies only to portions identified as distributable, incorporated in an
application with material additional functionality, and subject to the stated
conditions. It does not permit distributing the SDK as a standalone product.
The CUDA 12.1 Attachment A lists the CUDA runtime, cuBLAS, cuFFT, cuSPARSE,
cuSOLVER, cuRAND, NVRTC, CUPTI and NVTX Linux libraries. The CUDA 12.9.1
Attachment A lists `libnvJitLink.so`. The CUDA supplement does not state a
general requirement for case-by-case NVIDIA approval when these listed files are
redistributed under its conditions; it directs licensees to NVIDIA if the terms
do not fit their use.

The cuDNN 9.1 agreement states that runtime `.so` and `.dll` files are
redistributable under the agreement's conditions. The image retains the package's
license notice. These grants are conditional; this document is not a blanket legal
clearance or an assertion that individual NVIDIA approval is always required.

The cuSPARSELt supplement states that runtime files ending in `.so` and `.h` may
be distributed as part of an application. Torch 2.5.1+cu121 directly loads the
embedded `libcusparseLt-f80c68d1.so.0`; its `CUSPARSELT_VERSION` value is 502.
Because the Torch wheel does not include cuSPARSELt's own notice, the worker build
extracts that notice from the official `nvidia-cusparselt-cu12==0.6.3` x86_64
wheel, pinned by SHA256 in the Dockerfile. The notice carries the NVIDIA
cuSPARSELt supplement dated October 12, 2020.

Archived references:

- [CUDA 12.1 EULA](https://docs.nvidia.com/cuda/archive/12.1.0/eula/index.html)
- [CUDA 12.9.1 EULA](https://docs.nvidia.com/cuda/archive/12.9.1/eula/index.html)
- [cuDNN 9.1 EULA](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.1.0/reference/eula.html)

## File-Level Findings

`nvidia-nccl-cu12` 2.21.5 contains `libnccl.so.2`. Its exact installed package
notice grants redistribution in source and binary form subject to the included
copyright, attribution and disclaimer conditions. The image inventory retains
that notice under `inventory/*/licenses/nvidia-nccl-cu12/`; the generic PyPI
license metadata is not used as the basis for this finding.

The CUDA 12.1 Attachment A names Linux `libcupti.so`. The installed
`nvidia-cuda-cupti-cu12` 12.1.105 wheel also contains `libcheckpoint.so`,
`libnvperf_host.so`, `libnvperf_target.so` and `libpcsamplingutil.so`, which are
not individually named there. The worker does not use profiling APIs, so the
image build removes these four helper files, updates the wheel `RECORD`, and
records each removed file's hash under `inventory/nvidia-pruned/`. It retains
`libcupti.so.12`.

The same build removes `libcusolverMg.so.11`. It is not named in CUDA 12.1
Attachment A, has no ELF dependency from the retained cuSolver/Torch libraries,
and the worker does not use its multi-GPU API. The build records its removed-file
hash and retains the listed `libcusolver.so.11`.

The base Python environment and both worker venvs share the same NVIDIA package
installation, so the base prune report covers the runtime files used by all three
environments.

The exact candidate inventory must confirm the pruned-file reports, retained
libraries and cuSPARSELt notice before distribution.

The EULA redistribution grant requires an application with material additional
functionality, access to the redistributable libraries only by that application,
and distribution terms consistent with the EULA. The worker performs the
detector's model inference and processing; Torch accesses the CUDA runtimes for
that purpose, not as a standalone SDK. The project MIT license explicitly
excludes third-party components, and the worker retains their separate notices.
These conditions are technically satisfied by the reviewed worker design; the
exact candidate inventory remains the final check before setting the decision to
`REDISTRIBUTION DECISION: APPROVED`.

Do not imply NVIDIA endorsement. Export restrictions and codec patent obligations
remain separate considerations.
