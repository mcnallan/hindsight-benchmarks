import hashlib
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNNER))
from snapshot_quality_wrapper import snapshot_wrapper


def test_live_source_edits_do_not_change_blocked_snapshot_execution(tmp_path):
    owner = tmp_path / 'component'; owner.mkdir()
    run = tmp_path / 'run'; run.mkdir()
    source = owner / 'run_gpu1_quality.sh'
    source.write_text('''#!/usr/bin/env bash
set -eu
RUNNER_DIR=${QUALITY_RUNNER_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
printf 'ready\\n'
while [[ ! -f "$RUNNER_DIR/release" ]]; do sleep 0.01; done
printf 'original-complete:%s\\n' "$RUNNER_DIR"
''')
    snapshot, env = snapshot_wrapper(owner, run, 'full')
    frozen = Path(snapshot['path'])
    process = subprocess.Popen(['bash', str(frozen)], env=dict(os.environ, **env),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == 'ready'
        # The real owner source becomes syntactically invalid while its child is blocked.
        source.write_text('if THIS EDIT IS MALFORMED [1]')
        (owner / 'release').touch()
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 0 and stderr == ''
        assert stdout.strip() == f'original-complete:{owner}'
        assert hashlib.sha256(frozen.read_bytes()).hexdigest() == snapshot['source_sha256']
        assert stat.S_IMODE(frozen.stat().st_mode) == 0o600
    finally:
        if process.poll() is None:
            process.kill(); process.wait(timeout=5)
    with pytest.raises(ValueError, match='Preserve existing'):
        snapshot_wrapper(owner, run, 'full')
