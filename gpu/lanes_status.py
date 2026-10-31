"""Print lane run progress as JSON (runs on the Pod with the system Python)."""
import json
import os
import sys

work, exit_file = '/workspace/lanes-run', sys.argv[1]


def read(path):
    try:
        with open(path) as stream:
            return stream.read().strip()
    except OSError:
        return None


results = os.path.join(work, 'results')
done = [d for d in os.listdir(results) if not d.endswith('.tmp')] if os.path.isdir(results) else []
failed = [d for d in done if 'exit=0' not in (read(os.path.join(results, d, 'exit')) or '')]
total = read(os.path.join(work, 'total'))
print(json.dumps(dict(total=int(total) if total else None, done=len(done), failed=len(failed),
                      exit=read(exit_file), preflight=(read(os.path.join(work, 'preflight.log')) or '')[-500:])))
