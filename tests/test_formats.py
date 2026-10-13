"""Format ("golden file") tests: lock the shape of every file the CLI reads or writes.

These tests fail if a change alters a file format, a column name or order, or
how model output is parsed. If a change is intentional, regenerate the golden
files and review the diff before committing:

    UPDATE_GOLDEN=1 python -m unittest tests.test_formats

All fixtures use placeholder names and random file ids only (public repo).
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import music_flamingo  # noqa: E402
import parser as result_parser  # noqa: E402
from prompts import expand_prompt_instances, load_prompt_spec  # noqa: E402
from registry import append_registry_record, read_registry  # noqa: E402
from request_builder import build_requests, save_request_metadata_csv, save_requests_jsonl  # noqa: E402
from schemas import BatchJobRecord, InputRecord  # noqa: E402

GOLDEN = REPO / "tests" / "golden"
FIXTURES = REPO / "tests" / "fixtures"
UPDATE = os.environ.get("UPDATE_GOLDEN") == "1"
FIXED_TIME = "2026-01-01T00:00:00+00:00"

# The contracts, written out literally so any change shows up in review.
REGISTRY_COLUMNS = [
    "experiment_id", "job_id", "submitted_at", "status", "input_jsonl_uri", "output_uri_prefix",
    "result_collected", "dataset_id", "model", "prompt_id", "trials", "temperature", "base_seed", "backend",
]
METADATA_COLUMNS = [
    "request_id", "experiment_id", "dataset_id", "file_id", "file_uri", "model", "prompt_id",
    "prompt_instance_id", "trial_index", "variable_name", "variable_value", "temperature", "seed",
    "prompt_text_hash",
]
RESULT_BASE_COLUMNS = [
    "experiment_id", "dataset_id", "file_id", "file_uri", "model", "prompt_id", "prompt_instance_id",
    "trial_index", "raw_response_json", "job_id", "created_at", "submitted_at", "temperature", "seed",
    "prompt_text_hash",
]

SIMPLE_PROMPT = """prompt_id: golden_simple_v1
kind: simple
prompt_text: |
  Rate how this excerpt makes you feel, from 0 to 100.
  Return JSON with "joy" and "sadness".
response_schema:
  joy: number
  sadness: number
"""
TEMPLATED_TEXT_FIRST = """prompt_id: golden_templated_textfirst_v1
kind: templated
audio_position: after
prompt_text: |
  Rate this felt emotion from 0 to 100: {{item}}
variables:
  item:
    - joy
    - sadness
response_schema:
  score: number
"""


def check_golden(case: unittest.TestCase, name: str, actual: str) -> None:
    path = GOLDEN / name
    if UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual, encoding="utf-8")
        return
    case.assertTrue(path.exists(), f"Missing golden file {name}; run with UPDATE_GOLDEN=1")
    case.assertEqual(actual, path.read_text(encoding="utf-8"), f"{name} changed")


def header(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle))


class FormatTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def build(self, prompt_yaml: str, name: str):
        spec_path = self.tmp / f"{name}.yaml"
        spec_path.write_text(prompt_yaml)
        records = [
            InputRecord("audio", "3f9a1c7e5b2d8046", "gs://example-bucket/audio/3f9a1c7e5b2d8046.wav"),
            InputRecord("audio", "8c1d2e3f4a5b6c7d", "/data/example/8c1d2e3f4a5b6c7d.mp3"),
        ]
        return build_requests(
            experiment_id=f"example__{name}__1700000000", dataset_id="example", model="example-model",
            input_records=records, prompt_instances=expand_prompt_instances(load_prompt_spec(spec_path)),
            trials=2, temperature=1.0, base_seed=10,
        )

    # 1 + 2: request files ------------------------------------------------------
    def test_requests_and_metadata_files(self) -> None:
        for prompt, name in ((SIMPLE_PROMPT, "simple"), (TEMPLATED_TEXT_FIRST, "templated_textfirst")):
            requests = self.build(prompt, name)
            save_requests_jsonl(requests, self.tmp / "requests.jsonl")
            save_request_metadata_csv(requests, self.tmp / "request_metadata.csv")
            check_golden(self, f"{name}_requests.jsonl", (self.tmp / "requests.jsonl").read_text())
            check_golden(self, f"{name}_request_metadata.csv", (self.tmp / "request_metadata.csv").read_text())
            self.assertEqual(header(self.tmp / "request_metadata.csv"), METADATA_COLUMNS)

    def test_audio_position_orders_parts(self) -> None:
        for prompt, name, expected in (
            (SIMPLE_PROMPT, "simple", ["file_data", "text"]),
            (TEMPLATED_TEXT_FIRST, "templated", ["text", "file_data"]),
        ):
            save_requests_jsonl(self.build(prompt, name), self.tmp / "r.jsonl")
            for line in (self.tmp / "r.jsonl").read_text().splitlines():
                parts = json.loads(line)["request"]["contents"][0]["parts"]
                self.assertEqual([next(iter(p)) for p in parts], expected)

    # 3: parsed results ---------------------------------------------------------
    def test_parsed_results_file(self) -> None:
        requests = self.build(SIMPLE_PROMPT, "simple")
        save_request_metadata_csv(requests, self.tmp / "request_metadata.csv")
        answers = ['{"joy": 71.5, "sadness": 4}', '```json\n{"joy": 70}\n```', "not json at all", '{"joy": 69, "sadness": 6}']
        output = "".join(json.dumps(music_flamingo.output_line(r.request_id, a)) + "\n" for r, a in zip(requests, answers))
        index = result_parser.load_request_metadata_index(self.tmp / "request_metadata.csv")
        with mock.patch.object(result_parser, "utc_now_iso", return_value=FIXED_TIME):
            rows = result_parser.parse_output_jsonl_text(output, job_id="job-1", request_metadata_index=index, submitted_at=FIXED_TIME)
        result_parser.save_parsed_results_csv(rows, self.tmp / "parsed.csv")
        check_golden(self, "simple_parsed_results.csv", (self.tmp / "parsed.csv").read_text())
        self.assertEqual(header(self.tmp / "parsed.csv")[: len(RESULT_BASE_COLUMNS)], RESULT_BASE_COLUMNS)
        first = rows[0]
        self.assertEqual((first["joy"], first["sadness"], first["file_id"]), ("71.5", "4", "3f9a1c7e5b2d8046"))

    # 4: registry upgrades ------------------------------------------------------
    def test_registry_upgrades_from_every_older_header(self) -> None:
        old_headers = {"v1_11_columns": REGISTRY_COLUMNS[:11], "v2_13_columns": REGISTRY_COLUMNS[:13]}
        for name, columns in old_headers.items():
            path = self.tmp / f"{name}.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(columns)
                writer.writerow(["old-exp", "old-job", FIXED_TIME, "JOB_STATE_SUCCEEDED", "gs://b/in", "gs://b/out",
                                 "true", "old", "old-model", "old_prompt", "1", "1.0", ""][: len(columns)])
            append_registry_record(path, BatchJobRecord("new", "slurm:1", FIXED_TIME, "JOB_STATE_PENDING", "/r", "/o",
                                                        "false", "d", "m", "p", 1, "1.0", "0", "music-flamingo"))
            self.assertEqual(header(path), REGISTRY_COLUMNS, name)
            old, new = read_registry(path)
            self.assertEqual((old.experiment_id, old.job_id, old.backend), ("old-exp", "old-job", "vertex"), name)
            self.assertEqual(new.backend, "music-flamingo", name)

    # 5: real past Gemini output still parses -----------------------------------
    def test_real_gemini_output_v1_still_parses(self) -> None:
        index = result_parser.load_request_metadata_index(FIXTURES / "gemini_request_metadata_v1.csv")
        rows = result_parser.parse_output_jsonl_text(
            (FIXTURES / "gemini_raw_output_v1.jsonl").read_text(), job_id="j", request_metadata_index=index
        )
        self.assertEqual(len(rows), 3)
        by_uri = {Path(r["file_uri"]).stem: r for r in rows}
        self.assertEqual((by_uri["a606d6d5e11290be"]["mood"], by_uri["a606d6d5e11290be"]["energy"]), ("25.5", "96.0"))
        # Pre-change metadata: run settings blank, model's own file_id kept as returned.
        self.assertEqual(by_uri["a606d6d5e11290be"]["temperature"], "")
        self.assertEqual(by_uri["a606d6d5e11290be"]["file_id"], "audio_file")

    # 6: both backends produce output the parser treats the same ----------------
    def test_music_flamingo_output_matches_gemini_shape(self) -> None:
        gemini = json.loads((FIXTURES / "gemini_raw_output_v1.jsonl").read_text().splitlines()[0])
        text = gemini["response"]["candidates"][0]["content"]["parts"][0]["text"]
        flamingo = music_flamingo.output_line(gemini["custom_id"], text)
        self.assertEqual(result_parser.extract_request_id(flamingo), result_parser.extract_request_id(gemini))
        self.assertEqual(result_parser.extract_response_json(flamingo), result_parser.extract_response_json(gemini))

    def test_runner_keeps_part_order_for_both_positions(self) -> None:
        for prompt, name, expected in (
            (SIMPLE_PROMPT, "simple", ["audio", "text"]),
            (TEMPLATED_TEXT_FIRST, "templated", ["text", "audio"]),
        ):
            save_requests_jsonl(self.build(prompt, name), self.tmp / "r.jsonl")
            request = json.loads((self.tmp / "r.jsonl").read_text().splitlines()[0])
            conversation, _, seed = music_flamingo.request_to_conversation(request)
            self.assertEqual([p["type"] for p in conversation[0]["content"]], expected)
            self.assertEqual(seed, 10)


if __name__ == "__main__":
    unittest.main()
