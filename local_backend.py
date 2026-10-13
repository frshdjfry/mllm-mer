"""Launcher for the music-flamingo backend.

Offers the same methods as VertexBatchClient, so submit.py and collect.py
treat both backends alike. The runner (music_flamingo.py) is started either
directly on this machine (`local`) or as a Slurm job (`slurm`). Cluster
details (account, partition, modules, Python environment) come from the
backend config file, not from code.

Job ids in the registry are `local:<output dir>` or `slurm:<job id>`.
`collect-results` for Slurm jobs must run where `sacct` can see the job.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from music_flamingo import FAILURE_FILENAME
from schemas import BatchJobSubmission, utc_now_iso

REPO_DIR = Path(__file__).resolve().parent
OUTPUT_FILENAME = "raw_output.jsonl"

# Slurm job states (sacct) -> the JOB_STATE_* names used by the registry.
_SLURM_STATES = {
    "PENDING": "JOB_STATE_PENDING",
    "CONFIGURING": "JOB_STATE_PENDING",
    "REQUEUED": "JOB_STATE_PENDING",
    "RUNNING": "JOB_STATE_RUNNING",
    "COMPLETING": "JOB_STATE_RUNNING",
    "COMPLETED": "JOB_STATE_SUCCEEDED",
    "CANCELLED": "JOB_STATE_CANCELLED",
}
_FINISHED = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}


class LocalGpuClient:
    def __init__(self, config_path: str | Path | None = None, launcher: str = "local") -> None:
        if launcher not in {"local", "slurm"}:
            raise ValueError(f"Unknown launcher: {launcher}")
        self.config_path = Path(config_path).resolve() if config_path else None
        self.launcher = launcher

    # --- same interface as VertexBatchClient -------------------------------

    def upload_file(self, local_path: Path, uri: str) -> str:
        # Requests stay on this file system; copy only if a different path was asked for.
        if Path(uri).resolve() != Path(local_path).resolve():
            shutil.copyfile(local_path, uri)
        return uri

    def submit_batch_job(
        self,
        display_name: str,
        model: str,
        input_jsonl_uri: str,
        output_uri_prefix: str,
    ) -> BatchJobSubmission:
        if self.config_path is None:
            raise ValueError("The music-flamingo backend needs --backend-config.")
        output_dir = Path(output_uri_prefix)
        output_dir.mkdir(parents=True, exist_ok=True)
        submitted_at = utc_now_iso()
        if self.launcher == "local":
            from music_flamingo import load_config

            python = (load_config(self.config_path).get("local") or {}).get("python") or sys.executable
            result = subprocess.run(self.runner_command(input_jsonl_uri, output_dir, python))
            status = "JOB_STATE_SUCCEEDED" if result.returncode == 0 else "JOB_STATE_FAILED"
            job_id = f"local:{output_dir.resolve()}"
        else:
            script_path = output_dir.parent / "slurm_job.sh"
            script_path.write_text(self.slurm_script(display_name, input_jsonl_uri, output_dir), encoding="utf-8")
            completed = subprocess.run(
                ["sbatch", "--parsable", str(script_path)], check=True, capture_output=True, text=True
            )
            job_id = f"slurm:{completed.stdout.strip().split(';')[0]}"
            status = "JOB_STATE_PENDING"
        return BatchJobSubmission(
            job_id=job_id,
            status=status,
            submitted_at=submitted_at,
            input_jsonl_uri=str(input_jsonl_uri),
            output_uri_prefix=str(output_dir),
        )

    def get_job_status(self, job_id: str, output_uri_prefix: str = "") -> str:
        kind, _, ident = job_id.partition(":")
        if kind == "local":
            return self._status_from_files(Path(ident))
        if kind == "slurm":
            state = self._sacct_state(ident)
            if state == "JOB_STATE_SUCCEEDED" and output_uri_prefix:
                # Slurm says done; trust it only if the runner wrote its output.
                return self._status_from_files(Path(output_uri_prefix))
            return state
        raise ValueError(f"Not a music-flamingo job id: {job_id}")

    def get_batch_job(self, job_id: str) -> str:
        return job_id

    def download_job_output_text(self, batch_job: Any, output_uri_prefix: str | None = None) -> str:
        if not output_uri_prefix:
            raise ValueError("output_uri_prefix is required for music-flamingo jobs.")
        return (Path(output_uri_prefix) / OUTPUT_FILENAME).read_text(encoding="utf-8")

    def job_is_finished(self, status: str) -> bool:
        return status in _FINISHED

    def job_succeeded(self, status: str) -> bool:
        return status == "JOB_STATE_SUCCEEDED"

    # --- helpers -------------------------------------------------------------

    def runner_command(self, input_jsonl: str | Path, output_dir: Path, python: str) -> list[str]:
        return [
            python, "-m", "music_flamingo", "run",
            "--requests", str(Path(input_jsonl).resolve()),
            "--output", str((output_dir / OUTPUT_FILENAME).resolve()),
            "--config", str(self.config_path),
        ]

    def slurm_script(self, job_name: str, input_jsonl: str | Path, output_dir: Path) -> str:
        from music_flamingo import load_config

        config = load_config(self.config_path)
        slurm = config.get("slurm") or {}
        for key in ("account", "partition", "time", "python"):
            if not slurm.get(key):
                raise ValueError(f"Backend config is missing slurm.{key}")
        log_path = (output_dir.parent / "slurm-%j.out").resolve()
        lines = [
            "#!/bin/bash",
            f"#SBATCH -J {job_name[:64]}",
            f"#SBATCH --account {slurm['account']}",
            f"#SBATCH -p {slurm['partition']}",
            "#SBATCH --nodes 1",
            f"#SBATCH --gres={slurm.get('gres', 'gpu:1')}",
            f"#SBATCH --time={slurm['time']}",
            f"#SBATCH --output={log_path}",
            *[f"#SBATCH {option}" for option in slurm.get("extra_sbatch", [])],
            "set -euo pipefail",
            str(slurm.get("setup", "")).rstrip(),
        ]
        hf_home = (config.get("runtime") or {}).get("hf_home")
        if hf_home:
            lines.append(f"export HF_HOME={shlex.quote(str(hf_home))}")
        command = self.runner_command(input_jsonl, output_dir, str(slurm["python"]))
        lines += [f"cd {shlex.quote(str(REPO_DIR))}", " ".join(shlex.quote(part) for part in command), ""]
        return "\n".join(lines)

    @staticmethod
    def _status_from_files(output_dir: Path) -> str:
        if (output_dir / OUTPUT_FILENAME).is_file():
            return "JOB_STATE_SUCCEEDED"
        if (output_dir / FAILURE_FILENAME).is_file():
            return "JOB_STATE_FAILED"
        return "JOB_STATE_RUNNING"

    @staticmethod
    def _sacct_state(slurm_id: str) -> str:
        completed = subprocess.run(
            ["sacct", "-j", slurm_id, "-X", "--noheader", "--format=State"],
            check=True, capture_output=True, text=True,
        )
        return parse_sacct_state(completed.stdout)


def parse_sacct_state(stdout: str) -> str:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return "JOB_STATE_PENDING"  # not yet visible to sacct
    # e.g. "CANCELLED by 12345" -> "CANCELLED"
    state = lines[0].split()[0].rstrip("+")
    return _SLURM_STATES.get(state, "JOB_STATE_FAILED")
