#!/usr/bin/env bash
# Runs the Ultimate pipeline (tools/run_ultimate.py) as parallel lanes on one GPU.
# Lanes pull song chunks from a shared queue and start staggered, so one lane's CPU-only work
# (decoding, musical timing, matrix, evaluate) overlaps another lane's GPU stages.
# Usage: run-lanes.sh RUN_ID LANES CHUNK_SONGS STAGGER_SECONDS (songs in /workspace/queue/songs)
set -u
run_id=$1 lanes=$2 chunk=$3 stagger=$4
[[ $run_id =~ ^[A-Za-z0-9_-]+$ && $lanes =~ ^[1-9]$ && $chunk =~ ^[1-9][0-9]*$ && $stagger =~ ^[0-9]+$ ]] || exit 2
base=/workspace/whisper-test
work=/workspace/lanes-run
stem=/workspace/stem-comparison-venv/bin/python
qwen=/workspace/qwen-asr-venv/bin/python
shared=/workspace/experiments/whisper-evidence
exit_file=$base/ultimate-$run_id.exit
mkdir -p /workspace/experiments "$base"
rm -f "$exit_file"
exec 9>/workspace/experiments/.ultimate-run.lock
flock -n 9 || { echo 75 > "$exit_file"; exit 75; }
export TORCH_HOME="${TORCH_HOME:-/workspace/cache/torch}" HF_HOME="${HF_HOME:-/workspace/cache/huggingface}" PYTHONUNBUFFERED=1
rm -rf "$work" && mkdir -p "$work/chunks" "$work/claimed" "$work/results"
# Per-song stems and review clips take ~95 MB per song; earlier batches were already copied back.
rm -rf /workspace/lanes

# Existing private v1 images still contain Qwen's unused demo/GPL dependencies.
if ! "$qwen" -m pip uninstall -y soynlp gradio flask > "$work/preflight.log" 2>&1
then echo 3 > "$exit_file"; exit 3; fi

# Same CUDA/Triton/Qwen checks as experiments/run-ultimate.sh; compiles the Whisper kernels once before lanes share the cache.
if ! "$stem" - >> "$work/preflight.log" 2>&1 <<'PY'
import torch, whisper, panns_inference, demucs, yaml, soundfile
from whisper.triton_ops import median_filter_cuda
from whisper.timing import dtw_cuda
assert torch.cuda.is_available(), 'CUDA unavailable; no CPU fallback'
median_filter_cuda(torch.randn(1, 4, 64, device='cuda'), 7)
dtw_cuda(torch.rand(12, 80, device='cuda'))
torch.cuda.synchronize()
print('CUDA/Triton imports OK:', torch.cuda.get_device_name(0), flush=True)
PY
then echo 3 > "$exit_file"; exit 3; fi
if ! "$qwen" -c 'import torch; from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner; assert torch.cuda.is_available()' >> "$work/preflight.log" 2>&1
then echo 3 > "$exit_file"; exit 3; fi

i=0
for song in /workspace/queue/songs/*; do
  [[ -f $song ]] || continue
  dir=$work/chunks/$(printf %04d $((i / chunk)))
  mkdir -p "$dir" && ln "$song" "$dir/"
  i=$((i + 1))
done
echo "$(ls "$work/chunks" | wc -l)" > "$work/total"

lane() {
  local id=$1 evidence=/workspace/lanes/$1/experiments/whisper-evidence next code
  mkdir -p "$evidence/ultimate/logs" "$evidence/qwen"
  [[ -e $evidence/model-cache ]] || ln -s "$shared/model-cache" "$evidence/model-cache"
  [[ -e $evidence/qwen/models.json ]] || cp "$shared/qwen/models.json" "$evidence/qwen/models.json"
  sleep $((id * stagger))
  while true; do
    next=$(ls "$work/chunks" 2>/dev/null | head -n 1)
    [[ -n $next ]] || return 0
    mv "$work/chunks/$next" "$work/claimed/$next" 2>/dev/null || continue
    rm -f "$evidence/ultimate/report.json" "$evidence/ultimate/music-report.json" "$evidence/ultimate/run.json"
    POC_RUN_ID=$run_id-$next "$stem" "$base/tools/run_ultimate.py" --stem-python "$stem" --qwen-python "$qwen" \
      --songs "$work/claimed/$next" --evidence "$evidence" > "$evidence/ultimate/logs/ultimate-$run_id-$next.log" 2>&1
    code=$?
    mkdir -p "$work/results/$next.tmp"
    cp "$evidence/ultimate/report.json" "$evidence/ultimate/music-report.json" "$evidence/ultimate/run.json" \
      "$work/results/$next.tmp/" 2>/dev/null
    cp "$evidence/ultimate/logs/ultimate-$run_id-$next.log" "$work/results/$next.tmp/run.log"
    echo "lane=$id exit=$code" > "$work/results/$next.tmp/exit"
    mv "$work/results/$next.tmp" "$work/results/$next"
  done
}

for ((n = 0; n < lanes; n++)); do lane "$n" & done
wait
failed=$(grep -L 'exit=0' "$work"/results/*/exit 2>/dev/null | wc -l)
echo "$failed" > "$exit_file.tmp" && mv "$exit_file.tmp" "$exit_file"
