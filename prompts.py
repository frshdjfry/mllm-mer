from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
from typing import Any

from schemas import PromptInstance, PromptSpec

_VARIABLE_PATTERN = re.compile(r"{{\s*([a-zA-Z0-9_]+)\s*}}")


def load_prompt_spec(path: str | Path) -> PromptSpec:
    import yaml

    spec_path = Path(path)
    raw_text = spec_path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw_text)
    if not isinstance(data, dict):
        raise ValueError("Prompt spec must be a mapping.")

    prompt_id = require_string(data, "prompt_id")
    kind = require_string(data, "kind")
    prompt_text = require_string(data, "prompt_text")
    response_schema = require_mapping(data, "response_schema")
    variables = data.get("variables") or {}

    if kind not in {"simple", "templated"}:
        raise ValueError(f"Unsupported prompt kind: {kind}")
    if kind == "simple" and variables:
        raise ValueError("Simple prompt specs must not define variables.")
    if kind == "templated":
        if not isinstance(variables, dict) or not variables:
            raise ValueError("Templated prompt specs must define variables.")
        validate_template_variables(prompt_text, variables)

    return PromptSpec(
        prompt_id=prompt_id,
        kind=kind,
        prompt_text=prompt_text,
        response_schema=response_schema,
        variables={key: list(value) for key, value in variables.items()},
    )


def expand_prompt_instances(spec: PromptSpec) -> list[PromptInstance]:
    if spec.kind == "simple":
        return [
            PromptInstance(
                prompt_id=spec.prompt_id,
                prompt_instance_id=f"{spec.prompt_id}__base",
                prompt_text=spec.prompt_text,
                response_schema=spec.response_schema,
                variables={},
            )
        ]

    variable_names = list(spec.variables.keys())
    value_lists = [spec.variables[name] for name in variable_names]
    instances: list[PromptInstance] = []

    for values in itertools.product(*value_lists):
        variables = dict(zip(variable_names, values, strict=True))
        rendered = render_template(spec.prompt_text, variables)
        suffix = "__".join(f"{name}-{slugify(value)}" for name, value in variables.items())
        instances.append(
            PromptInstance(
                prompt_id=spec.prompt_id,
                prompt_instance_id=f"{spec.prompt_id}__{suffix}",
                prompt_text=rendered,
                response_schema=spec.response_schema,
                variables=variables,
            )
        )

    return instances


def save_prompt_instances(prompt_instances: list[PromptInstance], path: Path) -> None:
    payload = [instance.to_dict() for instance in prompt_instances]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_template(template: str, variables: dict[str, str]) -> str:
    rendered = template
    for name, value in variables.items():
        rendered = re.sub(r"{{\s*" + re.escape(name) + r"\s*}}", value, rendered)
    return rendered


def validate_template_variables(prompt_text: str, variables: dict[str, list[str]]) -> None:
    referenced = set(_VARIABLE_PATTERN.findall(prompt_text))
    declared = set(variables.keys())
    missing = referenced - declared
    extra = declared - referenced
    if missing:
        raise ValueError(f"Prompt template references undeclared variables: {sorted(missing)}")
    if extra:
        raise ValueError(f"Prompt spec declares unused variables: {sorted(extra)}")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return slug.strip("-") or "value"


def require_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty string for {key}")
    return value


def require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Expected non-empty mapping for {key}")
    return value
