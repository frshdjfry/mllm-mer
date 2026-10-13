"""Offline tests for the music-flamingo backend (no GPU, no Slurm, no network).

    python -m unittest discover -s tests
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cli  # noqa: E402
import local_backend  # noqa: E402
import music_flamingo  # noqa: E402
from collect import run_collect_results  # noqa: E402
from datasets import list_local_audio_records  # noqa: E402
from registry import read_registry  # noqa: E402
from submit import run_submit_local_experiment  # noqa: E402

EXAMPLE_CONFIG = REPO / "config" / "music_flamingo.cambridge-hpc.example.yaml"
PROMPT = """prompt_id: test_simple_v1
kind: simple
prompt_text: |
  Rate the felt emotions of this song from 0 to 100.
  Return JSON with "joy" and "sadness".
response_schema:
  joy: number
  sadness: number
"""


def fake_generate(conversation, temperature, seed):
    """Stands in for the model: echoes what it was given, as JSON."""
    content = conversation[0]["content"]
    audio = [part["path"] for part in content if part["type"] == "audio"]
    return json.dumps({"joy": 70, "sadness": 5, "heard": Path(audio[0]).name, "seed_seen": seed})


class MusicFlamingoBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.audio = self.tmp / "audio"
        (self.audio / "sub").mkdir(parents=True)
        for name in ("a1.wav", "sub/b2.mp3"):
            (self.audio / name).write_bytes(b"not really audio")
        (self.audio / "notes.txt").write_text("ignored")
        self.prompt = self.tmp / "prompt.yaml"
        self.prompt.write_text(PROMPT)
        self.config = self.tmp / "mf.yaml"
        self.config.write_text(EXAMPLE_CONFIG.read_text().replace("<CRSID>", "abc123").replace("<SLURM_ACCOUNT>", "test-account"))
        self.registry = self.tmp / "outputs" / "registry" / "experiments_registry.csv"
        self.artifacts = self.tmp / "outputs" / "experiments"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def fake_local_run(self, command, *args, **kwargs):
        """Replace the GPU subprocess with an in-process run using the fake model."""
        requests = Path(command[command.index("--requests") + 1])
        output = Path(command[command.index("--output") + 1])
        music_flamingo.run(requests, output, music_flamingo.load_config(self.config), generate=fake_generate)
        return subprocess.CompletedProcess(command, 0)

    def test_local_folder_listing(self) -> None:
        records = list_local_audio_records(self.audio)
        self.assertEqual([r.file_id for r in records], ["a1", "b2"])
        self.assertTrue(all(Path(r.file_uri).is_absolute() for r in records))
        (self.audio / "sub" / "a1.wav").write_bytes(b"x")
        with self.assertRaisesRegex(ValueError, "share the name 'a1'"):
            list_local_audio_records(self.audio)

    def test_request_to_conversation_keeps_order(self) -> None:
        request = {"custom_id": "x", "request": {"contents": [{"role": "user", "parts": [
            {"file_data": {"mime_type": "audio/wav", "file_uri": "/data/a1.wav"}}, {"text": "Rate it."}]}],
            "generationConfig": {"temperature": 0.7, "seed": 3}}}
        conversation, temperature, seed = music_flamingo.request_to_conversation(request)
        self.assertEqual([p["type"] for p in conversation[0]["content"]], ["audio", "text"])  # order kept
        self.assertEqual((temperature, seed), (0.7, 3))

    def test_end_to_end_local_launcher(self) -> None:
        fake_subprocess = types.SimpleNamespace(run=self.fake_local_run, CompletedProcess=subprocess.CompletedProcess)
        with mock.patch.object(local_backend, "subprocess", fake_subprocess):
            experiment_id = run_submit_local_experiment(
                dataset_dir=str(self.audio), prompt_spec_path=str(self.prompt), trials=2,
                registry_path=self.registry, local_artifacts_root=self.artifacts,
                backend_config=str(self.config), launcher="local", temperature=1.0, base_seed=10,
            )
        record = read_registry(self.registry)[0]
        self.assertEqual(record.backend, "music-flamingo")
        self.assertEqual(record.model, "nvidia/music-flamingo-hf@e560c33247f20e2f735c87ce9adebb8e53befe69")
        self.assertEqual(record.status, "JOB_STATE_SUCCEEDED")
        experiment_dir = self.artifacts / experiment_id
        info = json.loads((experiment_dir / "results" / "run_info.json").read_text())
        self.assertEqual(info["requests"], 4)
        self.assertEqual(len(info["code_version"]["commit"]), 40)  # which code produced the run
        self.assertIn("PyYAML", (experiment_dir / "results" / "environment.txt").read_text())
        spec = json.loads((experiment_dir / "experiment_spec.json").read_text())
        self.assertEqual(spec["backend"], "music-flamingo")
        self.assertEqual(spec["code_version"]["commit"], info["code_version"]["commit"])
        manifest = json.loads((experiment_dir / "input_manifest.json").read_text())
        import hashlib
        for record in manifest["records"]:  # which exact audio was used
            expected = hashlib.sha256(Path(record["file_uri"]).read_bytes()).hexdigest()
            self.assertEqual(record["attributes"]["sha256"], expected)

        collected = run_collect_results(self.registry, self.artifacts, project=None, location="global")
        self.assertEqual(collected, [experiment_id])
        with open(self.artifacts.parent / "experiment_results" / f"{experiment_id}.csv", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)  # 2 files x 2 trials
        for row in rows:
            self.assertEqual(Path(row["file_uri"]).name, row["heard"])  # right audio went with right request
            self.assertEqual(row["seed"], row["seed_seen"])  # seed reached the model
            self.assertEqual((row["joy"], row["temperature"]), ("70", "1.0"))
        self.assertEqual(sorted(r["seed"] for r in rows), ["10", "10", "11", "11"])
        self.assertTrue(read_registry(self.registry)[0].result_collected == "true")

    def test_audio_placeholder_in_prompt_is_rejected(self) -> None:
        request = {"custom_id": "x", "request": {"contents": [{"role": "user", "parts": [
            {"file_data": {"file_uri": "/data/a1.wav"}}, {"text": "<sound>\nRate it."}]}]}}
        with self.assertRaisesRegex(ValueError, "audio placeholder"):
            music_flamingo.request_to_conversation(request)

    def test_runner_failure_is_reported_not_collected(self) -> None:
        def broken(*_):
            raise RuntimeError("CUDA out of memory")

        requests = self.tmp / "r.jsonl"
        requests.write_text(json.dumps({"custom_id": "c", "request": {"contents": [{"parts": [{"text": "t"}]}]}}) + "\n")
        output = self.tmp / "res" / "raw_output.jsonl"
        with self.assertRaises(RuntimeError):
            music_flamingo.run(requests, output, music_flamingo.load_config(self.config), generate=broken)
        self.assertFalse(output.exists())
        self.assertIn("CUDA out of memory", (output.parent / "run_failed.txt").read_text())
        status = local_backend.LocalGpuClient().get_job_status(f"local:{output.parent}")
        self.assertEqual(status, "JOB_STATE_FAILED")

    def test_slurm_launcher_script_and_states(self) -> None:
        client = local_backend.LocalGpuClient(self.config, launcher="slurm")
        out = self.tmp / "exp" / "results"
        sbatch = mock.Mock(return_value=subprocess.CompletedProcess([], 0, stdout="4242;cluster\n"))
        with mock.patch.object(local_backend, "subprocess", types.SimpleNamespace(run=sbatch)):
            job = client.submit_batch_job("exp", "m", str(self.tmp / "exp" / "requests.jsonl"), str(out))
        self.assertEqual(job.job_id, "slurm:4242")
        self.assertEqual(sbatch.call_args.args[0][:2], ["sbatch", "--parsable"])
        script = (self.tmp / "exp" / "slurm_job.sh").read_text()
        for expected in ("#SBATCH --account test-account", "#SBATCH -p ampere", "#SBATCH --gres=gpu:1",
                         "module load rhel8/default-amp", "export HF_HOME=/rds/user/abc123/",
                         "/rds/user/abc123/hpc-work/envs/music-flamingo/bin/python -m music_flamingo run"):
            self.assertIn(expected, script)
        parse = local_backend.parse_sacct_state
        self.assertEqual(parse(""), "JOB_STATE_PENDING")
        self.assertEqual(parse("RUNNING\n"), "JOB_STATE_RUNNING")
        self.assertEqual(parse("COMPLETED\n"), "JOB_STATE_SUCCEEDED")
        self.assertEqual(parse("CANCELLED by 1234\n"), "JOB_STATE_CANCELLED")
        for failed in ("FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"):
            self.assertEqual(parse(failed), "JOB_STATE_FAILED")
        # Slurm says COMPLETED but the runner wrote no output -> not a success.
        with mock.patch.object(local_backend.LocalGpuClient, "_sacct_state", return_value="JOB_STATE_SUCCEEDED"):
            self.assertEqual(client.get_job_status("slurm:4242", str(out)), "JOB_STATE_RUNNING")

    def test_cli_requires_backend_specific_flags(self) -> None:
        cases = [
            (["submit-experiment", "--backend", "music-flamingo", "--prompt-spec", "p", "--trials", "1"],
             "--dataset-dir is required"),
            (["submit-experiment", "--prompt-spec", "p", "--trials", "1", "--project", "x"],
             "--dataset-uri is required"),
        ]
        for argv, message in cases:
            stderr = io.StringIO()
            with mock.patch.object(sys, "argv", ["cli.py", *argv]), redirect_stderr(stderr):
                with self.assertRaises(SystemExit):
                    cli.main()
            self.assertIn(message, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
