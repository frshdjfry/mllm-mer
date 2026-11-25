"""Offline tests for the qwen backend (no network, no API key, no openai package needed).

    python -m unittest discover -s tests
"""

from __future__ import annotations

import base64
import csv
import io
import json
import shutil
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
import parser as result_parser  # noqa: E402
import qwen_api  # noqa: E402
from collect import run_collect_results, run_rebuild_results  # noqa: E402
from registry import read_registry  # noqa: E402
from submit import run_submit_qwen_experiment  # noqa: E402

PROMPT = """prompt_id: test_simple_v1
kind: simple
prompt_text: |
  Rate the felt emotions of this song from 0 to 100.
  Return JSON with "joy" and "sadness".
response_schema:
  joy: number
  sadness: number
"""
SERVED = "qwen3.5-omni-plus-2026-03-15"
FIXTURES = REPO / "tests" / "fixtures"
GEMS_FIELDS = ["wonder", "transcendence", "tenderness", "peacefulness", "nostalgia", "joyful_activation", "power",
               "sadness", "tension"]


class FakeQwen:
    """Stands in for Model Studio: echoes what it was sent, as JSON."""

    def __init__(self, fail_on_call: int | None = None) -> None:
        self.calls: list[dict] = []
        self.fail_on_call = fail_on_call

    def __call__(self, kwargs: dict) -> dict:
        self.calls.append(kwargs)
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("connection dropped")
        content = kwargs["messages"][0]["content"]
        audio = next(part for part in content if part["type"] == "input_audio")["input_audio"]
        heard = base64.b64decode(audio["data"].split(",", 1)[1]).decode()
        answer = {"joy": 70, "sadness": 5, "heard": heard, "seed_seen": kwargs.get("seed"), "top_p_seen": kwargs.get("top_p")}
        return {"id": f"req-{len(self.calls)}", "model": SERVED, "finish_reason": "stop", "text": json.dumps(answer),
                "reasoning": "", "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}


class QwenTestBase(unittest.TestCase):
    """Temporary audio folder, prompt and registry, plus submit/results helpers."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.audio = self.tmp / "audio"
        self.audio.mkdir()
        for name in ("a1.wav", "b2.mp3"):
            (self.audio / name).write_bytes(name.encode())  # content = its own name, to check pairing
        self.prompt = self.tmp / "prompt.yaml"
        self.prompt.write_text(PROMPT)
        self.registry = self.tmp / "outputs" / "registry" / "experiments_registry.csv"
        self.artifacts = self.tmp / "outputs" / "experiments"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def submit(self, fake: FakeQwen, **kwargs) -> str:
        settings = {"trials": 2, "temperature": 1.0, "base_seed": 10, "top_p": 0.9, **kwargs}
        return run_submit_qwen_experiment(
            dataset_dir=str(self.audio), prompt_spec_path=str(self.prompt), model="qwen3.5-omni-plus",
            registry_path=self.registry, local_artifacts_root=self.artifacts,
            client=qwen_api.QwenApiClient(call=fake), **settings,
        )

    def results(self, experiment_id: str) -> list[dict]:
        with open(self.artifacts.parent / "experiment_results" / f"{experiment_id}.csv", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))


class QwenBackendTest(QwenTestBase):
    def test_request_translation(self) -> None:
        request = {"custom_id": "x", "request": {"model": "qwen3.8-omni-flash", "contents": [{"role": "user", "parts": [
            {"text": "Rate it."}, {"file_data": {"mime_type": "audio/mpeg", "file_uri": str(self.audio / "b2.mp3")}}]}],
            "generationConfig": {"temperature": 0.7, "seed": 3, "topP": 0.9}}}
        kwargs = qwen_api.request_to_chat_kwargs(request)
        content = kwargs["messages"][0]["content"]
        self.assertEqual([part["type"] for part in content], ["text", "input_audio"])  # order kept
        self.assertEqual(content[1]["input_audio"]["format"], "mp3")
        self.assertEqual(content[1]["input_audio"]["data"], "data:;base64," + base64.b64encode(b"b2.mp3").decode())
        self.assertEqual((kwargs["model"], kwargs["temperature"], kwargs["seed"], kwargs["top_p"]),
                         ("qwen3.8-omni-flash", 0.7, 3, 0.9))
        self.assertEqual((kwargs["stream"], kwargs["modalities"]), (True, ["text"]))
        self.assertNotIn("response_format", kwargs)  # the request did not ask for JSON
        self.assertNotIn("messages", qwen_api.sent_settings(kwargs))  # audio is never copied into the logs

        request["request"]["generationConfig"]["responseMimeType"] = "application/json"
        self.assertEqual(qwen_api.request_to_chat_kwargs(request)["response_format"], {"type": "json_object"})

        del request["request"]["generationConfig"]["topP"], request["request"]["generationConfig"]["seed"]
        kwargs = qwen_api.request_to_chat_kwargs(request)
        self.assertNotIn("top_p", kwargs)  # not given: not sent
        self.assertNotIn("seed", kwargs)

    def test_remote_uris_and_oversized_audio_are_rejected(self) -> None:
        request = {"custom_id": "x", "request": {"model": "m", "contents": [{"role": "user", "parts": [
            {"file_data": {"file_uri": "gs://bucket/a1.wav"}}]}]}}
        with self.assertRaisesRegex(ValueError, "local audio files"):
            qwen_api.request_to_chat_kwargs(request)
        with mock.patch.object(qwen_api, "MAX_BASE64_BYTES", 8):
            with self.assertRaisesRegex(ValueError, "too large"):
                qwen_api.audio_part(str(self.audio / "a1.wav"))

    def test_end_to_end_submit_and_collect(self) -> None:
        fake = FakeQwen()
        experiment_id = self.submit(fake)
        self.assertEqual(len(fake.calls), 4)  # 2 files x 2 trials
        record = read_registry(self.registry)[0]
        self.assertEqual((record.backend, record.model, record.status), ("qwen", "qwen3.5-omni-plus", "JOB_STATE_SUCCEEDED"))
        experiment_dir = self.artifacts / experiment_id
        spec = json.loads((experiment_dir / "experiment_spec.json").read_text())
        self.assertIn("enable_thinking", spec["unset_generation_settings"])
        info = json.loads((experiment_dir / "results" / "run_info.json").read_text())
        self.assertEqual((info["requests"], info["models_served"]), (4, [SERVED]))
        self.assertEqual(info["usage_totals_this_run"]["total_tokens"], 480)

        self.assertEqual(run_collect_results(self.registry, self.artifacts, project=None, location="global"), [experiment_id])
        rows = self.results(experiment_id)
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertEqual(Path(row["file_uri"]).name, row["answer_heard"])  # right audio went with right request
            self.assertEqual(row["seed"], row["answer_seed_seen"])  # seed reached the API
            self.assertEqual((row["top_p"], row["answer_top_p_seen"]), ("0.9", "0.9"))
            self.assertEqual((row["backend"], row["model_served"], row["hardware"]), ("qwen", SERVED, "alibaba_hosted"))
            self.assertEqual(row["schema_keys"], "extra: heard, seed_seen, top_p_seen")
        self.assertEqual(sorted(r["seed"] for r in rows), ["10", "10", "11", "11"])

        raw = (experiment_dir / "results" / "raw_output.jsonl").read_text()
        self.assertNotIn("base64", raw)  # sent settings are logged, the audio is not
        self.assertIn('"stream": true', raw)
        self.assertIn('"response_format": {"type": "json_object"}', raw)  # prompt specs always ask for JSON
        raw_before = raw
        self.assertEqual(run_rebuild_results(self.registry, self.artifacts), [experiment_id])
        self.assertEqual((experiment_dir / "results" / "raw_output.jsonl").read_text(), raw_before)

    def test_unset_top_p_is_marked_model_default(self) -> None:
        experiment_id = self.submit(FakeQwen(), top_p=None, base_seed=None, trials=1)
        run_collect_results(self.registry, self.artifacts, project=None, location="global")
        for row in self.results(experiment_id):
            self.assertEqual((row["top_p"], row["seed"]), ("model_default", "not_set"))

    def test_stopped_run_resumes_without_paying_twice(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            experiment_id = self.submit(FakeQwen(fail_on_call=3))
        self.assertIn("Resume with", stderr.getvalue())
        record = read_registry(self.registry)[0]
        self.assertEqual(record.status, qwen_api.STOPPED)
        results = self.artifacts / experiment_id / "results"
        self.assertTrue((results / qwen_api.FAILURE_FILENAME).is_file())
        # Not finished, so collect leaves it pending instead of closing it.
        self.assertEqual(run_collect_results(self.registry, self.artifacts, project=None, location="global"), [])
        self.assertEqual(read_registry(self.registry)[0].result_collected, "false")

        resumed = FakeQwen()
        qwen_api.run(Path(record.input_jsonl_uri), results / qwen_api.OUTPUT_FILENAME, call=resumed)
        self.assertEqual(len(resumed.calls), 2)  # only the two unanswered requests
        self.assertFalse((results / qwen_api.FAILURE_FILENAME).exists())
        self.assertEqual(run_collect_results(self.registry, self.artifacts, project=None, location="global"), [experiment_id])
        self.assertEqual(len(self.results(experiment_id)), 4)

    def test_retries_only_transient_errors(self) -> None:
        class ApiError(Exception):
            def __init__(self, status_code: int) -> None:
                self.status_code = status_code

        waits: list[float] = []
        outcomes = [ApiError(429), ApiError(503), {"ok": True}]

        def flaky(kwargs):
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        self.assertEqual(qwen_api.call_with_retries(flaky, {}, sleep=waits.append), {"ok": True})
        self.assertEqual(len(waits), 2)
        with self.assertRaises(ApiError):  # a bad request is not retried
            qwen_api.call_with_retries(lambda kwargs: (_ for _ in ()).throw(ApiError(400)), {}, sleep=waits.append)
        self.assertEqual(len(waits), 2)

    def test_stream_is_joined(self) -> None:
        def chunk(content=None, reasoning=None, finish=None, usage=None):
            delta = types.SimpleNamespace(content=content, reasoning_content=reasoning)
            choices = [] if usage else [types.SimpleNamespace(delta=delta, finish_reason=finish)]
            return types.SimpleNamespace(id="r1", model=SERVED, choices=choices, usage=usage)

        usage = types.SimpleNamespace(model_dump=lambda: {"total_tokens": 9})
        result = qwen_api.collect_stream([chunk(reasoning="hmm"), chunk('{"joy": '), chunk("1}", finish="stop"), chunk(usage=usage)])
        self.assertEqual((result["text"], result["reasoning"], result["finish_reason"]), ('{"joy": 1}', "hmm", "stop"))
        self.assertEqual((result["model"], result["usage"]), (SERVED, {"total_tokens": 9}))

    def test_settings_come_from_env_file_not_code(self) -> None:
        env = self.tmp / ".env"
        env.write_text("# comment\nDASHSCOPE_API_KEY='sk-test'\nexport QWEN_BASE_URL=https://example.invalid/v1\n")
        with mock.patch.object(qwen_api, "ENV_FILE", env), mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(qwen_api.api_settings(), ("sk-test", "https://example.invalid/v1"))
        with mock.patch.object(qwen_api, "ENV_FILE", self.tmp / "missing"), mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "DASHSCOPE_API_KEY is not set"):
                qwen_api.api_settings()

    def test_recorded_real_qwen_output_still_parses(self) -> None:
        """Replay real answers from Model Studio (2026-11-24), not the fake's assumptions.

        tests/fixtures/qwen_raw_output_v1.jsonl holds three recorded lines, changed only
        in custom_id (placeholder ids) and, for Flash, qwen.reasoning (it quoted lyrics):
        1. qwen3.5-omni-plus without JSON mode: the answer is wrapped in ```json ... ```
        2. qwen3.5-omni-plus with JSON mode
        3. qwen3.8-omni-flash with JSON mode, which also returned reasoning text
        """
        index = result_parser.load_request_metadata_index(FIXTURES / "qwen_request_metadata_v1.csv")
        rows = result_parser.parse_output_jsonl_text(
            (FIXTURES / "qwen_raw_output_v1.jsonl").read_text(), job_id="j", request_metadata_index=index,
            run_columns={"backend": "qwen", "model_served": "", "settings_source": "s", "code_commit": "c",
                         "hardware": "alibaba_hosted"},
            schema_fields={"gems_felt_simple_v1__base": GEMS_FIELDS},
        )
        fenced, plus, flash = rows
        raw = [json.loads(line) for line in (FIXTURES / "qwen_raw_output_v1.jsonl").read_text().splitlines()]

        # Without JSON mode the real models fence their JSON, which the shared parser
        # deliberately does not read: flagged, never silently filled in.
        self.assertTrue(raw[0]["response"]["candidates"][0]["content"]["parts"][0]["text"].startswith("```json"))
        self.assertNotIn("response_format", raw[0]["qwen"]["sent"])
        self.assertTrue(fenced["schema_keys"].startswith("missing: " + ", ".join(GEMS_FIELDS)))
        self.assertEqual(fenced.get("answer_wonder", ""), "")

        # With JSON mode (what the backend now sends) both models parse cleanly.
        for row, line, model in ((plus, raw[1], "qwen3.5-omni-plus"), (flash, raw[2], "qwen3.8-omni-flash")):
            self.assertEqual(line["qwen"]["sent"]["response_format"], {"type": "json_object"})
            answer = json.loads(line["response"]["candidates"][0]["content"]["parts"][0]["text"])
            self.assertEqual(row["schema_keys"], "exact")
            self.assertEqual({field: row["answer_" + field] for field in GEMS_FIELDS}, {f: str(answer[f]) for f in GEMS_FIELDS})
            self.assertEqual((row["model_served"], row["backend"]), (model, "qwen"))
        self.assertEqual((plus["answer_wonder"], flash["answer_wonder"]), ("15.0", "61.0"))
        self.assertTrue(raw[2]["qwen"]["reasoning"])  # Flash thinks by default; Plus does not
        self.assertEqual(raw[1]["qwen"]["reasoning"], "")

    def test_cli_requires_qwen_flags(self) -> None:
        for argv, message in [
            (["--prompt-spec", "p", "--trials", "1", "--model", "qwen3.5-omni-plus"], "--dataset-dir is required with --backend qwen"),
            (["--prompt-spec", "p", "--trials", "1", "--dataset-dir", "d"], "--model is required with --backend qwen"),
        ]:
            stderr = io.StringIO()
            with mock.patch.object(sys, "argv", ["cli.py", "submit-experiment", "--backend", "qwen", *argv]), redirect_stderr(stderr):
                with self.assertRaises(SystemExit):
                    cli.main()
            self.assertIn(message, stderr.getvalue())


class ApiStatusError(Exception):
    """Like openai.APIStatusError: carries the HTTP status."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}: rejected")
        self.status_code = status_code


class QwenReviewFixesTest(QwenTestBase):
    """Regressions for the PR #3 review: no double billing, nothing lost, provenance complete."""

    def output_paths(self, experiment_id: str) -> tuple[Path, Path]:
        results = self.artifacts / experiment_id / "results"
        return Path(read_registry(self.registry)[0].input_jsonl_uri), results / qwen_api.OUTPUT_FILENAME

    def test_finished_run_is_not_run_again(self) -> None:
        experiment_id = self.submit(FakeQwen(), trials=1)
        requests, output = self.output_paths(experiment_id)
        before = output.read_text()
        again = FakeQwen()
        with self.assertRaisesRegex(ValueError, "already exists"):
            qwen_api.run(requests, output, call=again)
        self.assertEqual((len(again.calls), output.read_text()), (0, before))  # nothing re-billed or replaced

    def test_ctrl_c_still_registers_the_experiment(self) -> None:
        class Interrupting(FakeQwen):
            def __call__(self, kwargs):
                if len(self.calls) == 1:
                    raise KeyboardInterrupt
                return super().__call__(kwargs)

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            experiment_id = self.submit(Interrupting(), trials=2)
        self.assertIn("interrupted (Ctrl-C)", stderr.getvalue())
        record = read_registry(self.registry)[0]
        self.assertEqual((record.experiment_id, record.status), (experiment_id, qwen_api.STOPPED))

    def test_resume_keeps_every_attempt_and_flags_mixed_code(self) -> None:
        with redirect_stderr(io.StringIO()):
            experiment_id = self.submit(FakeQwen(fail_on_call=3))
        requests, output = self.output_paths(experiment_id)
        later = {"commit": "b" * 40, "branch": "x", "uncommitted_changes": False}
        with mock.patch.object(qwen_api, "code_version", return_value=later):
            qwen_api.run(requests, output, call=FakeQwen())
        info = json.loads((output.parent / qwen_api.RUN_INFO_FILENAME).read_text())
        self.assertEqual(len(info["attempts"]), 2)
        self.assertIn("failed_at", info["attempts"][0])  # the first attempt is not lost
        self.assertEqual([a["answered_this_attempt"] for a in info["attempts"]], [2, 2])
        commits = {json.loads(line)["qwen"]["code_commit"] for line in output.read_text().splitlines()}
        self.assertEqual(len(commits), 2)  # each answer says which code produced it
        run_collect_results(self.registry, self.artifacts, project=None, location="global")
        for row in self.results(experiment_id):
            self.assertTrue(row["code_commit"].startswith("mixed: "), row["code_commit"])

    def test_bad_audio_is_found_before_anything_is_sent(self) -> None:
        (self.audio / "c3.wav").write_bytes(b"x" * 100)
        fake = FakeQwen()
        stderr = io.StringIO()
        with mock.patch.object(qwen_api, "MAX_BASE64_BYTES", 50), redirect_stderr(stderr):
            self.submit(fake, trials=1)
        self.assertEqual(len(fake.calls), 0)  # nothing paid for
        self.assertIn("c3.wav is too large", stderr.getvalue())

    def test_a_rejected_request_is_logged_and_the_run_goes_on(self) -> None:
        class RejectsOne(FakeQwen):
            def __call__(self, kwargs):
                if len(self.calls) == 1:
                    self.calls.append(kwargs)
                    raise ApiStatusError(400)
                return super().__call__(kwargs)

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            experiment_id = self.submit(RejectsOne(), trials=2)
            self.assertEqual(read_registry(self.registry)[0].status, "JOB_STATE_SUCCEEDED")
            run_collect_results(self.registry, self.artifacts, project=None, location="global")
        self.assertEqual(len(self.results(experiment_id)), 3)  # 4 requests, 1 rejected
        rejected = (self.artifacts / experiment_id / "results" / qwen_api.REJECTED_FILENAME).read_text()
        self.assertEqual(json.loads(rejected)["status_code"], 400)
        self.assertIn("1 requests in", stderr.getvalue())  # collect warns about the missing row

    def test_rejections_before_any_answer_stop_the_run(self) -> None:
        class RejectsAll(FakeQwen):
            def __call__(self, kwargs):
                self.calls.append(kwargs)
                raise ApiStatusError(400)

        fake = RejectsAll()
        with redirect_stderr(io.StringIO()):
            self.submit(fake, trials=2)
        self.assertEqual(len(fake.calls), qwen_api.MAX_REJECTED_BEFORE_ANY_ANSWER)
        self.assertEqual(read_registry(self.registry)[0].status, qwen_api.STOPPED)

    def test_collect_names_stopped_runs(self) -> None:
        with redirect_stderr(io.StringIO()):
            experiment_id = self.submit(FakeQwen(fail_on_call=2))
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            run_collect_results(self.registry, self.artifacts, project=None, location="global")
        self.assertIn(f"Not collected: {experiment_id} stopped part-way", stderr.getvalue())
        self.assertIn("python -m qwen_api run --requests", stderr.getvalue())

    def test_a_line_cut_off_mid_write_does_not_block_resume(self) -> None:
        with redirect_stderr(io.StringIO()):
            experiment_id = self.submit(FakeQwen(fail_on_call=3))
        requests, output = self.output_paths(experiment_id)
        partial = output.with_suffix(output.suffix + ".partial")
        partial.write_text(partial.read_text() + '{"custom_id": "r0", "resp')  # killed mid-write
        resumed = FakeQwen()
        qwen_api.run(requests, output, call=resumed)
        self.assertEqual(len(resumed.calls), 2)
        self.assertEqual(len(output.read_text().splitlines()), 4)
        info = json.loads((output.parent / qwen_api.RUN_INFO_FILENAME).read_text())
        self.assertTrue(info["dropped_incomplete_last_line"])


if __name__ == "__main__":
    unittest.main()
