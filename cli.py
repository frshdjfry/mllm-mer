from __future__ import annotations

import argparse
import os
from pathlib import Path

from collect import run_collect_results
from submit import run_submit_experiment, run_submit_text_experiment

DEFAULT_REGISTRY_PATH = Path("outputs/registry/experiments_registry.csv")
DEFAULT_ARTIFACTS_ROOT = Path("outputs/experiments")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gemini batch experiments on audio files in GCS.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit-experiment", help="Create and submit a batch experiment.")
    submit_parser.add_argument("--dataset-uri", required=True, help="GCS dataset URI such as gs://bucket/prefix")
    submit_parser.add_argument("--prompt-spec", required=True, help="Path to YAML prompt spec file")
    submit_parser.add_argument("--model", required=True, help="Vertex model resource name")
    submit_parser.add_argument("--trials", required=True, type=int, help="Number of trials per file/prompt")
    submit_parser.add_argument("--output-uri-prefix", required=True, help="GCS prefix for batch inputs/outputs")
    add_generation_args(submit_parser)
    add_shared_args(submit_parser)

    submit_text_parser = subparsers.add_parser(
        "submit-text-experiment",
        help="Create and submit a text-only batch experiment from a CSV.",
    )
    submit_text_parser.add_argument("--input-csv", required=True, help="CSV path with file_id and transcription")
    submit_text_parser.add_argument("--prompt-spec", required=True, help="Path to YAML prompt spec file")
    submit_text_parser.add_argument("--model", required=True, help="Vertex model resource name")
    submit_text_parser.add_argument("--trials", required=True, type=int, help="Number of trials per row/prompt")
    submit_text_parser.add_argument("--output-uri-prefix", required=True, help="GCS prefix for batch inputs/outputs")
    add_generation_args(submit_text_parser)
    add_shared_args(submit_text_parser)

    collect_parser = subparsers.add_parser("collect-results", help="Collect completed batch results.")
    add_shared_args(collect_parser)

    return parser


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature sent with every request (default: 1.0, Gemini 3's default).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed; trial i uses seed + i. Omit to let the model choose a random seed.",
    )


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--registry-path",
        default=str(DEFAULT_REGISTRY_PATH),
        help=f"CSV registry path (default: {DEFAULT_REGISTRY_PATH})",
    )
    parser.add_argument(
        "--artifacts-root",
        default=str(DEFAULT_ARTIFACTS_ROOT),
        help=f"Local experiment artifact root (default: {DEFAULT_ARTIFACTS_ROOT})",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        help="Google Cloud project id. Defaults to GOOGLE_CLOUD_PROJECT.",
    )
    parser.add_argument(
        "--location",
        default=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
        help="Vertex AI location. Defaults to GOOGLE_CLOUD_LOCATION or global.",
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    project = args.project
    if not project:
        parser.error("--project is required, or set GOOGLE_CLOUD_PROJECT")

    if args.command in {"submit-experiment", "submit-text-experiment"}:
        if not 0.0 <= args.temperature <= 2.0:
            parser.error("--temperature must be between 0.0 and 2.0")
        if args.seed is not None and not (
            -(2**31) <= args.seed and args.seed + args.trials - 1 <= 2**31 - 1
        ):
            parser.error("--seed plus trials must stay within the 32-bit integer range")

    registry_path = Path(args.registry_path)
    artifacts_root = Path(args.artifacts_root)

    if args.command == "submit-experiment":
        experiment_id = run_submit_experiment(
            dataset_uri=args.dataset_uri,
            prompt_spec_path=args.prompt_spec,
            model=args.model,
            trials=args.trials,
            output_uri_prefix=args.output_uri_prefix,
            registry_path=registry_path,
            local_artifacts_root=artifacts_root,
            project=project,
            location=args.location,
            temperature=args.temperature,
            base_seed=args.seed,
        )
        print(experiment_id)
        return

    if args.command == "submit-text-experiment":
        experiment_id = run_submit_text_experiment(
            input_csv=args.input_csv,
            prompt_spec_path=args.prompt_spec,
            model=args.model,
            trials=args.trials,
            output_uri_prefix=args.output_uri_prefix,
            registry_path=registry_path,
            local_artifacts_root=artifacts_root,
            project=project,
            location=args.location,
            temperature=args.temperature,
            base_seed=args.seed,
        )
        print(experiment_id)
        return

    if args.command == "collect-results":
        collected = run_collect_results(
            registry_path=registry_path,
            local_artifacts_root=artifacts_root,
            project=project,
            location=args.location,
        )
        if collected:
            for experiment_id in collected:
                print(experiment_id)
        else:
            print("No completed jobs were collected.")
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
