"""Feature token policies for final-project ablation experiments."""

from __future__ import annotations

from typing import Any

import run_faers_pipeline as pipeline


ABLATION_CONFIGS: dict[str, dict[str, Any]] = {
    "all_tokens": {
        "description": "Structured, binned, and all text token families",
        "include_structured": True,
        "include_prefixes": None,
        "exclude_prefixes": (),
    },
    "without_reaction_pt": {
        "description": "All features except reaction PT tokens",
        "include_structured": True,
        "include_prefixes": None,
        "exclude_prefixes": ("reac:",),
    },
    "without_reaction_outcome": {
        "description": "All features except reaction outcome tokens",
        "include_structured": True,
        "include_prefixes": None,
        "exclude_prefixes": ("reactionoutcome:",),
    },
    "drug_indication_only": {
        "description": "Drug and indication text tokens only",
        "include_structured": False,
        "include_prefixes": ("drug:", "indi:"),
        "exclude_prefixes": (),
    },
    "structured_only": {
        "description": "Demographic, source, quarter, and count-bin tokens only",
        "include_structured": True,
        "include_prefixes": (),
        "exclude_prefixes": (),
    },
}


def _matches_prefix(token: str, prefixes: tuple[str, ...]) -> bool:
    return any(token.startswith(prefix) for prefix in prefixes)


def build_ablation_text(row: dict[str, str], experiment_name: str) -> str:
    config = ABLATION_CONFIGS.get(experiment_name)
    if config is None:
        raise ValueError(f"Unknown ablation experiment: {experiment_name}")

    tokens: list[str] = []
    if config["include_structured"]:
        structured_row = dict(row)
        structured_row["text_tokens"] = ""
        tokens.extend(pipeline.build_hash_text(structured_row).split())

    include_prefixes = config["include_prefixes"]
    exclude_prefixes = tuple(config["exclude_prefixes"])
    for token in row.get("text_tokens", "").split():
        if include_prefixes is not None and not _matches_prefix(token, tuple(include_prefixes)):
            continue
        if _matches_prefix(token, exclude_prefixes):
            continue
        tokens.append(token)

    return " ".join(tokens)
