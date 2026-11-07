"""Print lane run progress as JSON (runs on the Pod with the system Python)."""
import glob
import json
import os
import sys

WORK = '/workspace/lanes-run'
LANES = '/workspace/lanes'


def read(path):
    try:
        with open(path) as stream:
            return stream.read().strip()
    except OSError:
        return None


def summary(work, lanes, exit_file, run_id):
    """Chunks done, plus each lane's current pipeline stage and the share of stages it finished."""
    results = os.path.join(work, 'results')
    done = [d for d in os.listdir(results) if not d.endswith('.tmp')] if os.path.isdir(results) else []
    failed = [d for d in done if 'exit=0' not in (read(os.path.join(results, d, 'exit')) or '')]
    total = read(os.path.join(work, 'total'))
    progress, running = float(len(done)), []
    for path in sorted(glob.glob(os.path.join(lanes, '*', 'experiments', 'whisper-evidence', 'ultimate', 'run.json'))):
        try:
            run = json.loads(read(path) or '')
        except ValueError:
            continue
        chunk = str(run.get('launcher_run_id') or '').rpartition('-')[2]
        if run.get('launcher_run_id') != f'{run_id}-{chunk}' or chunk in done or run.get('status') != 'running':
            continue
        stages, count = run.get('stages') or [], run.get('total_stages') or 0
        finished = sum(1 for stage in stages if stage.get('status') == 'complete')
        if count:
            progress += min(finished, count) / count
        running.append(dict(chunk=chunk, finished=finished, stages=count,
                            stage=next((s['stage'] for s in stages if s.get('status') == 'running'), None)))
    return dict(total=int(total) if total else None, done=len(done), failed=len(failed), exit=read(exit_file),
                progress=round(progress, 3), lanes=running,
                preflight=(read(os.path.join(work, 'preflight.log')) or '')[-500:])


if __name__ == '__main__':
    exit_file = sys.argv[1]
    run_id = os.path.basename(exit_file).removeprefix('ultimate-').removesuffix('.exit')
    print(json.dumps(summary(WORK, LANES, exit_file, run_id)))