#!/usr/bin/env bash
set -eu
RUN_DIR=${1:?usage: monitor_local_reflect.sh RUN_DIR}
RUNNER_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
"$RUNNER_DIR/.venv/bin/python" - "$RUN_DIR" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
if not (run / 'status.json').exists():
    print('Reflect: preflight; status not yet available')
    raise SystemExit()
state = json.loads((run / 'status.json').read_text())
print('Reflect stage:', state['stage'])
if state.get('total_sessions'):
    print(f"Shared bank: {state.get('ingested_sessions', 0)}/{state['total_sessions']} sessions")
for name in ['redhat', 'production']:
    value = state.get('models', {}).get(name, {})
    mode = 'full' if ((run / name / 'full-questions.private.json').exists()
                      or state['stage'].startswith('full') or state['stage'].startswith('complete')) else 'smoke'
    file = run / name / f'{mode}-questions.private.json'
    if file.exists():
        trace = json.loads(file.read_text())
        result = trace.get('result')
        if result:
            print(f"{name}: {mode} {result['correct']}/{result['total']} ({result['accuracy']}%); "
                  f"reflect errors={result['reflect_errors']}, judge errors={result['judge_errors']}")
        else:
            questions = trace.get('questions', [])
            print(f"{name}: {mode} {len(questions)}/242 captured, {sum(x['correct'] for x in questions)} correct so far")
    else:
        print(f'{name}: {mode} pending')
if state.get('error'):
    print('Failure:', state['error_type'], state['error'])
PY
if [[ -f "$RUN_DIR/reflect.log" ]]; then tail -n 10 "$RUN_DIR/reflect.log"; fi
