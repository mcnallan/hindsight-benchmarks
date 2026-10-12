"""Freeze the quality shell wrapper before launch; preserve its component directory."""
import hashlib
from pathlib import Path


def snapshot_wrapper(runner_dir: Path, run_dir: Path, phase: str):
    if phase not in ('smoke', 'full'):
        raise ValueError('Snapshot phase must be smoke or full')
    runner_dir = runner_dir.resolve()
    source = runner_dir / 'run_gpu1_quality.sh'
    destination = run_dir.resolve() / f'run_gpu1_quality-{phase}.snapshot.sh'
    body = source.read_bytes()
    if destination.exists():
        raise ValueError('Preserve existing wrapper snapshot; use a fresh phase/run')
    destination.write_bytes(body)
    destination.chmod(0o600)
    sha = hashlib.sha256(body).hexdigest()
    return {'path': str(destination), 'source_path': str(source), 'source_sha256': sha,
            'snapshot_sha256': sha, 'runner_dir': str(runner_dir)}, {
                'QUALITY_RUNNER_DIR': str(runner_dir)}
