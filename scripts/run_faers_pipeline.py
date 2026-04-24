#!/usr/bin/env python
"""FAERS XML risk prediction pipeline.

The implementation intentionally avoids pandas/lightgbm/snorkel so it can run in
the provided environment. It streams the FDA XML files, writes case-level CSV
features, trains two sklearn baselines, and renders audit reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ.setdefault(_thread_var, "1")

import numpy as np


SERIOUSNESS_FIELDS = [
    "serious",
    "seriousnessdeath",
    "seriousnesslifethreatening",
    "seriousnesshospitalization",
    "seriousnessdisabling",
    "seriousnesscongenitalanomali",
    "seriousnessother",
]

Q4_VALID_FRACTION = 0.5
SPLIT_POLICY = "2025Q1-Q3 train, earliest 50% of 2025Q4 valid, latest 50% of 2025Q4 test by receivedate"

LEAKAGE_FIELDS = set(SERIOUSNESS_FIELDS + ["label_serious"])

CASE_COLUMNS = [
    "quarter",
    "split",
    "source_file",
    "safetyreportid",
    "safetyreportversion",
    "receivedate",
    "receiptdate",
    "transmissiondate",
    "primarysourcecountry",
    "occurcountry",
    "reporttype",
    "fulfillexpeditecriteria",
    "duplicate",
    "reportercountry",
    "qualification",
    "sendertype",
    "patientsex",
    "patientagegroup",
    "age_years",
    "patientweight",
    *SERIOUSNESS_FIELDS,
    "label_serious",
    "drug_count",
    "suspect_drug_count",
    "concomitant_drug_count",
    "interacting_drug_count",
    "dose_available_count",
    "dose_value_mean",
    "duration_days_min",
    "duration_days_mean",
    "duration_days_max",
    "drug_name_count",
    "drug_name_covered_count",
    "active_substance_count",
    "canonical_drug_count",
    "reaction_count",
    "reaction_unique_count",
    "indication_count",
    "has_drug_start",
    "has_drug_end",
    "text_tokens",
]

NUMERIC_FEATURES = [
    "age_years",
    "patientweight",
    "reporttype",
    "duplicate",
    "qualification",
    "sendertype",
    "patientsex",
    "patientagegroup",
    "drug_count",
    "suspect_drug_count",
    "concomitant_drug_count",
    "interacting_drug_count",
    "dose_available_count",
    "dose_value_mean",
    "duration_days_min",
    "duration_days_mean",
    "duration_days_max",
    "canonical_drug_count",
    "reaction_count",
    "reaction_unique_count",
    "indication_count",
    "has_drug_start",
    "has_drug_end",
]

TOKEN_RE = re.compile(r"[^A-Z0-9]+")
QUARTER_RE = re.compile(r"(20\d{2})q([1-4])", re.IGNORECASE)


@dataclass(frozen=True)
class QuarterData:
    label: str
    root: Path
    xml_files: list[Path]
    deleted_file: Path | None


def ensure_dirs(out_dir: Path) -> dict[str, Path]:
    paths = {
        "out": out_dir,
        "interim": out_dir / "interim",
        "models": out_dir / "models",
        "reports": out_dir / "reports",
        "figures": out_dir / "reports" / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def direct_child(elem: ET.Element | None, tag: str) -> ET.Element | None:
    if elem is None:
        return None
    for child in elem:
        if local_name(child.tag) == tag:
            return child
    return None


def children(elem: ET.Element | None, tag: str) -> list[ET.Element]:
    if elem is None:
        return []
    return [child for child in elem if local_name(child.tag) == tag]


def direct_text(elem: ET.Element | None, tag: str, default: str = "") -> str:
    child = direct_child(elem, tag)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def text_or_empty(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_float(value: Any) -> float | None:
    text = text_or_empty(value)
    if not text:
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
    return None


def format_number(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return f"{float(value):.6g}"


def parse_yyyymmdd(value: str) -> date | None:
    text = text_or_empty(value)
    if len(text) < 8 or not text[:8].isdigit():
        return None
    try:
        return date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    except ValueError:
        return None


def age_to_years(value: Any, unit: Any) -> float | None:
    age = parse_float(value)
    if age is None or age < 0:
        return None
    code = text_or_empty(unit)
    factors = {
        "800": 10.0,  # decade
        "801": 1.0,  # year
        "802": 1.0 / 12.0,  # month
        "803": 1.0 / 52.1429,  # week
        "804": 1.0 / 365.25,  # day
        "805": 1.0 / 8766.0,  # hour
        "806": 1.0 / 525960.0,  # minute
    }
    factor = factors.get(code)
    if factor is None:
        return None
    years = age * factor
    if years > 130:
        return None
    return years


def age_bin(age_text: str) -> str:
    age = parse_float(age_text)
    if age is None:
        return "unknown"
    if age < 18:
        return "child"
    if age < 65:
        return "adult"
    return "senior"


def normalize_text(value: str) -> str:
    text = text_or_empty(value).upper()
    text = TOKEN_RE.sub(" ", text)
    return " ".join(text.split())


def make_token(prefix: str, value: str) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    token_value = text.replace(" ", "_")
    if len(token_value) > 120:
        token_value = token_value[:120]
    return f"{prefix}:{token_value}"


def canonical_drug_name(drug: ET.Element) -> tuple[str, bool]:
    active_names = []
    for active in children(drug, "activesubstance"):
        name = normalize_text(direct_text(active, "activesubstancename"))
        if name:
            active_names.append(name)
    if active_names:
        return active_names[0], True
    medicinal = normalize_text(direct_text(drug, "medicinalproduct"))
    return medicinal, False


def label_from_flags(flags: dict[str, str]) -> int:
    if flags.get("serious") == "1":
        return 1
    for field in SERIOUSNESS_FIELDS:
        if field != "serious" and flags.get(field) == "1":
            return 1
    return 0


def discover_quarters(data_dir: Path) -> list[QuarterData]:
    quarters: list[QuarterData] = []
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        match = QUARTER_RE.search(child.name)
        if not match:
            continue
        label = f"{match.group(1)}Q{match.group(2)}"
        xml_dir = child / "XML"
        xml_files = sorted(xml_dir.glob("*.xml")) if xml_dir.exists() else []
        deleted_files = sorted((child / "Deleted").glob("DELETE*.txt")) if (child / "Deleted").exists() else []
        quarters.append(QuarterData(label=label, root=child, xml_files=xml_files, deleted_file=deleted_files[0] if deleted_files else None))
    if not quarters:
        raise FileNotFoundError(f"No faers_xml_YYYYqN quarter directories found under {data_dir}")
    return sorted(quarters, key=lambda item: item.label)


def split_for_quarter(label: str) -> str:
    q = int(label[-1])
    if q <= 3:
        return "train"
    return "q4_pending"


def report_id_sort_parts(report_id: str) -> tuple[int, int, str]:
    text = text_or_empty(report_id)
    if text.isdigit():
        return (0, int(text), "")
    return (1, 0, text)


def q4_split_sort_key(row: dict[str, str]) -> tuple[int, int, int, str]:
    parsed_date = parse_yyyymmdd(row.get("receivedate", ""))
    date_key = parsed_date.toordinal() if parsed_date else 10**9
    id_kind, id_num, id_text = report_id_sort_parts(row.get("safetyreportid", ""))
    return (date_key, id_kind, id_num, id_text)


def assign_q4_splits(rows: list[dict[str, str]], valid_fraction: float = Q4_VALID_FRACTION) -> Counter:
    ordered_indices = sorted(range(len(rows)), key=lambda index: q4_split_sort_key(rows[index]))
    cutoff = int(len(rows) * valid_fraction)
    counts: Counter = Counter()
    for rank, index in enumerate(ordered_indices):
        split = "valid" if rank < cutoff else "test"
        rows[index]["split"] = split
        counts[split] += 1
    return counts


def count_safetyreports(path: Path) -> int:
    pattern = b"<safetyreport>"
    count = 0
    tail = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024 * 8)
            if not chunk:
                break
            data = tail + chunk
            count += data.count(pattern)
            tail = data[-(len(pattern) - 1) :]
    return count


def load_deleted_ids(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    deleted = set()
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            item = line.strip()
            if item:
                deleted.add(item)
    return deleted


def run_inventory(data_dir: Path, out_dir: Path) -> dict[str, Any]:
    paths = ensure_dirs(out_dir)
    quarters = discover_quarters(data_dir)
    inventory: dict[str, Any] = {"data_dir": str(data_dir), "quarters": {}, "total_reports": 0, "total_deleted_ids": 0}
    for quarter in quarters:
        deleted_ids = load_deleted_ids(quarter.deleted_file)
        quarter_info = {
            "root": str(quarter.root),
            "deleted_file": str(quarter.deleted_file) if quarter.deleted_file else "",
            "deleted_ids": len(deleted_ids),
            "xml_files": [],
            "reports": 0,
        }
        for xml_file in quarter.xml_files:
            reports = count_safetyreports(xml_file)
            quarter_info["xml_files"].append({"path": str(xml_file), "bytes": xml_file.stat().st_size, "reports": reports})
            quarter_info["reports"] += reports
            print(f"[inventory] {quarter.label} {xml_file.name}: {reports:,} reports")
        inventory["quarters"][quarter.label] = quarter_info
        inventory["total_reports"] += quarter_info["reports"]
        inventory["total_deleted_ids"] += len(deleted_ids)
    (paths["interim"] / "inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    (paths["reports"] / "data_inventory.md").write_text(render_inventory_markdown(inventory), encoding="utf-8")
    print(f"[inventory] total reports: {inventory['total_reports']:,}")
    return inventory


def render_inventory_markdown(inventory: dict[str, Any]) -> str:
    lines = [
        "# FAERS XML 数据清点",
        "",
        f"- 数据目录：`{inventory['data_dir']}`",
        f"- XML safetyreport 总数：{inventory['total_reports']:,}",
        f"- 删除列表 ID 总数：{inventory['total_deleted_ids']:,}",
        "",
        "| 季度 | XML 文件数 | safetyreport | 删除列表 ID |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, info in inventory["quarters"].items():
        lines.append(f"| {label} | {len(info['xml_files'])} | {info['reports']:,} | {info['deleted_ids']:,} |")
    lines.append("")
    return "\n".join(lines)


def iter_safetyreports(xml_file: Path) -> Iterable[ET.Element]:
    context = ET.iterparse(xml_file, events=("start", "end"))
    _, root = next(context)
    for event, elem in context:
        if event == "end" and local_name(elem.tag) == "safetyreport":
            yield elem
            root.clear()


def extract_case(report: ET.Element, quarter: str, source_file: str) -> dict[str, str]:
    patient = direct_child(report, "patient")
    primarysource = direct_child(report, "primarysource")
    sender = direct_child(report, "sender")

    flags = {field: direct_text(report, field) for field in SERIOUSNESS_FIELDS}
    label = label_from_flags(flags)

    age_years = age_to_years(direct_text(patient, "patientonsetage"), direct_text(patient, "patientonsetageunit"))
    patientweight = parse_float(direct_text(patient, "patientweight"))

    drugs = children(patient, "drug")
    reactions = children(patient, "reaction")

    drug_tokens: set[str] = set()
    reaction_tokens: set[str] = set()
    indication_tokens: set[str] = set()
    misc_tokens: set[str] = set()

    drug_name_count = 0
    drug_name_covered_count = 0
    active_substance_count = 0
    canonical_names: set[str] = set()
    dose_values: list[float] = []
    durations: list[int] = []
    indication_names: set[str] = set()
    has_start = 0
    has_end = 0
    drug_characterization = Counter()

    for drug in drugs:
        drug_name_count += 1
        characterization = direct_text(drug, "drugcharacterization")
        if characterization:
            drug_characterization[characterization] += 1
            misc_tokens.add(f"drugchar:{characterization}")
        canonical, from_active = canonical_drug_name(drug)
        if canonical:
            drug_name_covered_count += 1
            canonical_names.add(canonical)
            token = make_token("drug", canonical)
            if token:
                drug_tokens.add(token)
        if from_active:
            active_substance_count += 1

        dose = parse_float(direct_text(drug, "drugstructuredosagenumb"))
        if dose is not None:
            dose_values.append(dose)

        start = parse_yyyymmdd(direct_text(drug, "drugstartdate"))
        end = parse_yyyymmdd(direct_text(drug, "drugenddate"))
        if start:
            has_start = 1
        if end:
            has_end = 1
        if start and end and end >= start:
            durations.append((end - start).days)

        indication = direct_text(drug, "drugindication")
        if indication:
            normalized_indication = normalize_text(indication)
            indication_names.add(normalized_indication)
            token = make_token("indi", indication)
            if token:
                indication_tokens.add(token)

        route = direct_text(drug, "drugadministrationroute")
        if route:
            misc_tokens.add(f"route:{normalize_text(route).replace(' ', '_')}")
        actiondrug = direct_text(drug, "actiondrug")
        if actiondrug:
            misc_tokens.add(f"actiondrug:{normalize_text(actiondrug).replace(' ', '_')}")

    reaction_names: set[str] = set()
    for reaction in reactions:
        name = direct_text(reaction, "reactionmeddrapt")
        normalized = normalize_text(name)
        if normalized:
            reaction_names.add(normalized)
            token = make_token("reac", normalized)
            if token:
                reaction_tokens.add(token)
        outcome = direct_text(reaction, "reactionoutcome")
        if outcome:
            misc_tokens.add(f"reactionoutcome:{normalize_text(outcome).replace(' ', '_')}")

    all_tokens = sorted(drug_tokens | reaction_tokens | indication_tokens | misc_tokens)

    row: dict[str, str] = {
        "quarter": quarter,
        "split": split_for_quarter(quarter),
        "source_file": source_file,
        "safetyreportid": direct_text(report, "safetyreportid"),
        "safetyreportversion": direct_text(report, "safetyreportversion"),
        "receivedate": direct_text(report, "receivedate"),
        "receiptdate": direct_text(report, "receiptdate"),
        "transmissiondate": direct_text(report, "transmissiondate"),
        "primarysourcecountry": direct_text(report, "primarysourcecountry"),
        "occurcountry": direct_text(report, "occurcountry"),
        "reporttype": direct_text(report, "reporttype"),
        "fulfillexpeditecriteria": direct_text(report, "fulfillexpeditecriteria"),
        "duplicate": direct_text(report, "duplicate"),
        "reportercountry": direct_text(primarysource, "reportercountry"),
        "qualification": direct_text(primarysource, "qualification"),
        "sendertype": direct_text(sender, "sendertype"),
        "patientsex": direct_text(patient, "patientsex"),
        "patientagegroup": direct_text(patient, "patientagegroup"),
        "age_years": format_number(age_years),
        "patientweight": format_number(patientweight),
        **flags,
        "label_serious": str(label),
        "drug_count": str(len(drugs)),
        "suspect_drug_count": str(drug_characterization.get("1", 0)),
        "concomitant_drug_count": str(drug_characterization.get("2", 0)),
        "interacting_drug_count": str(drug_characterization.get("3", 0)),
        "dose_available_count": str(len(dose_values)),
        "dose_value_mean": format_number(float(np.mean(dose_values)) if dose_values else None),
        "duration_days_min": format_number(float(np.min(durations)) if durations else None),
        "duration_days_mean": format_number(float(np.mean(durations)) if durations else None),
        "duration_days_max": format_number(float(np.max(durations)) if durations else None),
        "drug_name_count": str(drug_name_count),
        "drug_name_covered_count": str(drug_name_covered_count),
        "active_substance_count": str(active_substance_count),
        "canonical_drug_count": str(len(canonical_names)),
        "reaction_count": str(len(reactions)),
        "reaction_unique_count": str(len(reaction_names)),
        "indication_count": str(len(indication_names)),
        "has_drug_start": str(has_start),
        "has_drug_end": str(has_end),
        "text_tokens": " ".join(all_tokens),
    }
    return row


def run_etl(data_dir: Path, out_dir: Path, sample: int | None = None) -> dict[str, Any]:
    paths = ensure_dirs(out_dir)
    quarters = discover_quarters(data_dir)
    parse_log: dict[str, Any] = {"sample_per_quarter": sample, "quarters": {}}
    for quarter in quarters:
        start_time = time.time()
        deleted_ids = load_deleted_ids(quarter.deleted_file)
        out_csv = paths["interim"] / f"cases_{quarter.label}.csv"
        stats = {
            "xml_files": len(quarter.xml_files),
            "parsed_reports": 0,
            "kept_reports": 0,
            "deleted_skipped": 0,
            "missing_id": 0,
            "split_counts": {},
            "output": str(out_csv),
        }
        q4_rows: list[dict[str, str]] = []
        is_q4 = quarter.label.endswith("Q4")
        with out_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CASE_COLUMNS)
            writer.writeheader()
            for xml_file in quarter.xml_files:
                if sample is not None and stats["kept_reports"] >= sample:
                    break
                print(f"[etl] {quarter.label} parsing {xml_file.name}")
                for report in iter_safetyreports(xml_file):
                    stats["parsed_reports"] += 1
                    report_id = direct_text(report, "safetyreportid")
                    if not report_id:
                        stats["missing_id"] += 1
                    if report_id in deleted_ids:
                        stats["deleted_skipped"] += 1
                        continue
                    row = extract_case(report, quarter.label, xml_file.name)
                    if is_q4:
                        q4_rows.append(row)
                    else:
                        writer.writerow(row)
                        stats["split_counts"][row["split"]] = stats["split_counts"].get(row["split"], 0) + 1
                    stats["kept_reports"] += 1
                    if sample is not None and stats["kept_reports"] >= sample:
                        break
            if is_q4:
                split_counts = assign_q4_splits(q4_rows)
                for row in q4_rows:
                    writer.writerow(row)
                stats["split_counts"] = dict(split_counts)
        stats["elapsed_seconds"] = round(time.time() - start_time, 2)
        parse_log["quarters"][quarter.label] = stats
        print(f"[etl] {quarter.label}: kept {stats['kept_reports']:,}, skipped deleted {stats['deleted_skipped']:,}")
    (paths["interim"] / "parse_log.json").write_text(json.dumps(parse_log, ensure_ascii=False, indent=2), encoding="utf-8")
    return parse_log


def feature_paths(interim_dir: Path) -> list[Path]:
    paths = sorted(interim_dir.glob("cases_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No case CSV files found in {interim_dir}. Run --mode etl first.")
    return paths


def iter_feature_rows(paths: Iterable[Path], split: str | None = None) -> Iterable[dict[str, str]]:
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if split is None or row.get("split") == split:
                    yield row


def numeric_value(row: dict[str, str], field: str) -> float:
    value = parse_float(row.get(field))
    return np.nan if value is None else float(value)


def build_hash_text(row: dict[str, str]) -> str:
    tokens: list[str] = []
    categorical_fields = [
        "quarter",
        "primarysourcecountry",
        "occurcountry",
        "reporttype",
        "fulfillexpeditecriteria",
        "duplicate",
        "reportercountry",
        "qualification",
        "sendertype",
        "patientsex",
        "patientagegroup",
    ]
    for field in categorical_fields:
        value = normalize_text(row.get(field, ""))
        if value:
            tokens.append(f"{field}:{value.replace(' ', '_')}")
    tokens.append(f"age_bin:{age_bin(row.get('age_years', ''))}")
    tokens.append(f"drug_count_bin:{count_bin(row.get('drug_count', ''))}")
    tokens.append(f"reaction_count_bin:{count_bin(row.get('reaction_count', ''))}")
    tokens.append(f"indication_count_bin:{count_bin(row.get('indication_count', ''))}")
    if row.get("text_tokens"):
        tokens.append(row["text_tokens"])
    return " ".join(tokens)


def count_bin(value: str) -> str:
    parsed = parse_float(value)
    if parsed is None:
        return "unknown"
    if parsed <= 0:
        return "0"
    if parsed == 1:
        return "1"
    if parsed <= 4:
        return "2_4"
    if parsed <= 9:
        return "5_9"
    return "10_plus"


def read_hash_chunks(paths: list[Path], split: str, chunk_size: int) -> Iterable[tuple[list[str], np.ndarray]]:
    texts: list[str] = []
    labels: list[int] = []
    for row in iter_feature_rows(paths, split):
        texts.append(build_hash_text(row))
        labels.append(int(row["label_serious"]))
        if len(texts) >= chunk_size:
            yield texts, np.asarray(labels, dtype=np.int8)
            texts, labels = [], []
    if texts:
        yield texts, np.asarray(labels, dtype=np.int8)


def split_label_counts(paths: list[Path]) -> dict[str, Counter]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in iter_feature_rows(paths):
        counts[row["split"]][int(row["label_serious"])] += 1
    return counts


def class_weights_from_counts(counts: Counter) -> dict[int, float]:
    total = counts[0] + counts[1]
    weights = {}
    for label in (0, 1):
        if counts[label] == 0:
            weights[label] = 1.0
        else:
            weights[label] = total / (2.0 * counts[label])
    return weights


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40, 40)
    return 1.0 / (1.0 + np.exp(-clipped))


def train_logistic(paths: list[Path], model_dir: Path, chunk_size: int, n_features: int, random_state: int) -> dict[str, Any]:
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import SGDClassifier

    counts = split_label_counts(paths)
    train_counts = counts["train"]
    if train_counts[0] == 0 or train_counts[1] == 0:
        raise ValueError(f"Training split must contain both classes, got {dict(train_counts)}")
    weights = class_weights_from_counts(train_counts)

    vectorizer = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        token_pattern=r"(?u)\b\S+\b",
    )
    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-5, random_state=random_state)
    classes = np.asarray([0, 1], dtype=np.int8)
    seen = 0
    for texts, y in read_hash_chunks(paths, "train", chunk_size):
        x = vectorizer.transform(texts)
        sample_weight = np.asarray([weights[int(label)] for label in y], dtype=np.float64)
        if seen == 0:
            clf.partial_fit(x, y, classes=classes, sample_weight=sample_weight)
        else:
            clf.partial_fit(x, y, sample_weight=sample_weight)
        seen += len(y)
        print(f"[train] logistic seen {seen:,} train rows", end="\r")
    print()
    model_path = model_dir / "logistic_sgd.pkl"
    with model_path.open("wb") as handle:
        pickle.dump({"model": clf, "vectorizer": vectorizer, "numeric_features": [], "type": "hash_logistic"}, handle)
    return {"model": clf, "vectorizer": vectorizer, "path": str(model_path), "train_rows": seen}


def metadata_from_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "patientsex": row.get("patientsex", ""),
        "age_years": row.get("age_years", ""),
        "quarter": row.get("quarter", ""),
    }


def collect_metadata(paths: list[Path], split: str) -> list[dict[str, str]]:
    return [metadata_from_row(row) for row in iter_feature_rows(paths, split)]


def predict_logistic(
    paths: list[Path],
    split: str,
    model_bundle: dict[str, Any],
    chunk_size: int,
    collect_rows: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    clf = model_bundle["model"]
    vectorizer = model_bundle["vectorizer"]
    labels: list[int] = []
    probs: list[float] = []
    for chunk_texts, y in read_hash_chunks(paths, split, chunk_size):
        x = vectorizer.transform(chunk_texts)
        if hasattr(clf, "predict_proba"):
            p = clf.predict_proba(x)[:, 1]
        else:
            p = sigmoid(clf.decision_function(x))
        labels.extend(int(v) for v in y)
        probs.extend(float(v) for v in p)
    rows = collect_metadata(paths, split) if collect_rows else []
    return np.asarray(labels, dtype=np.int8), np.asarray(probs, dtype=np.float64), rows


def row_to_numeric(row: dict[str, str]) -> list[float]:
    return [numeric_value(row, field) for field in NUMERIC_FEATURES]


def load_numeric_split(
    paths: list[Path],
    split: str,
    max_rows: int | None = None,
    random_state: int = 42,
    collect_rows: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    rng = random.Random(random_state)
    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    meta_rows: list[dict[str, str]] = []
    seen = 0
    for row in iter_feature_rows(paths, split):
        seen += 1
        item_x = row_to_numeric(row)
        item_y = int(row["label_serious"])
        item_meta = metadata_from_row(row) if collect_rows else {}
        if max_rows is None or len(x_rows) < max_rows:
            x_rows.append(item_x)
            y_rows.append(item_y)
            if collect_rows:
                meta_rows.append(item_meta)
        else:
            index = rng.randint(0, seen - 1)
            if index < max_rows:
                x_rows[index] = item_x
                y_rows[index] = item_y
                if collect_rows:
                    meta_rows[index] = item_meta
    if not x_rows:
        raise ValueError(f"No rows found for split={split}")
    return np.asarray(x_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.int8), meta_rows


def train_numeric_hgb(paths: list[Path], model_dir: Path, max_rows: int, random_state: int) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier

    x_train, y_train, _ = load_numeric_split(paths, "train", max_rows=max_rows, random_state=random_state)
    counts = Counter(int(v) for v in y_train)
    if counts[0] == 0 or counts[1] == 0:
        raise ValueError(f"Numeric training sample must contain both classes, got {dict(counts)}")
    weights = class_weights_from_counts(counts)
    sample_weight = np.asarray([weights[int(label)] for label in y_train], dtype=np.float64)
    clf = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=random_state,
    )
    print(f"[train] numeric_hgb fitting {len(y_train):,} sampled train rows")
    clf.fit(x_train, y_train, sample_weight=sample_weight)
    model_path = model_dir / "numeric_hgb.pkl"
    with model_path.open("wb") as handle:
        pickle.dump({"model": clf, "numeric_features": NUMERIC_FEATURES, "type": "numeric_hgb"}, handle)
    return {"model": clf, "path": str(model_path), "train_rows": len(y_train)}


def predict_numeric(paths: list[Path], split: str, model_bundle: dict[str, Any], collect_rows: bool = False) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    x, y, rows = load_numeric_split(paths, split, max_rows=None, collect_rows=collect_rows)
    clf = model_bundle["model"]
    p = clf.predict_proba(x)[:, 1]
    return y, p, rows


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.5
    candidates = sorted(set(np.linspace(0.05, 0.95, 19).tolist() + np.quantile(y_prob, np.linspace(0.05, 0.95, 19)).tolist()))
    best_t = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        metrics = classification_metrics(y_true, y_prob, threshold)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_t = float(threshold)
    return best_t


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float | int | None]:
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

    y_pred = (y_prob >= threshold).astype(np.int8)
    metrics: dict[str, float | int | None] = {
        "n": int(len(y_true)),
        "positives": int(np.sum(y_true)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else None,
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)) if len(y_true) else None,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)) if len(y_true) else None,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) else None,
    }
    if len(set(int(v) for v in y_true)) == 2:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["auroc"] = None
        metrics["auprc"] = None
    for pct in (0.01, 0.05, 0.10):
        metrics[f"recall_at_top_{int(pct * 100)}pct"] = recall_at_top_pct(y_true, y_prob, pct)
        metrics[f"hit_rate_top_{int(pct * 100)}pct"] = hit_rate_top_pct(y_true, y_prob, pct)
    return metrics


def recall_at_top_pct(y_true: np.ndarray, y_prob: np.ndarray, pct: float) -> float | None:
    positives = int(np.sum(y_true))
    if positives == 0 or len(y_true) == 0:
        return None
    k = max(1, int(math.ceil(len(y_true) * pct)))
    top_idx = np.argsort(-y_prob)[:k]
    return float(np.sum(y_true[top_idx]) / positives)


def hit_rate_top_pct(y_true: np.ndarray, y_prob: np.ndarray, pct: float) -> float | None:
    if len(y_true) == 0:
        return None
    k = max(1, int(math.ceil(len(y_true) * pct)))
    top_idx = np.argsort(-y_prob)[:k]
    return float(np.mean(y_true[top_idx]))


def stratified_metrics(rows: list[dict[str, str]], y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    strata: dict[str, Any] = {}
    group_defs = {
        "sex": lambda row: row.get("patientsex") or "unknown",
        "age_bin": lambda row: age_bin(row.get("age_years", "")),
        "quarter": lambda row: row.get("quarter") or "unknown",
    }
    for group_name, getter in group_defs.items():
        group_values: dict[str, list[int]] = defaultdict(list)
        for idx, row in enumerate(rows):
            group_values[getter(row)].append(idx)
        strata[group_name] = {}
        for value, indices in sorted(group_values.items()):
            if len(indices) < 20:
                continue
            idx = np.asarray(indices, dtype=np.int64)
            strata[group_name][value] = classification_metrics(y_true[idx], y_prob[idx], threshold)
    return strata


def train_models(out_dir: Path, chunk_size: int, n_features: int, tree_max_rows: int, random_state: int) -> dict[str, Any]:
    paths = ensure_dirs(out_dir)
    cases = feature_paths(paths["interim"])
    metrics: dict[str, Any] = {"models": {}, "splits": ["train", "valid", "test"]}

    logistic = train_logistic(cases, paths["models"], chunk_size, n_features, random_state)
    logistic_results: dict[str, Any] = {"model_path": logistic["path"], "train_rows": logistic["train_rows"], "split_metrics": {}}
    valid_y, valid_p, _ = predict_logistic(cases, "valid", logistic, chunk_size)
    logistic_threshold = best_threshold(valid_y, valid_p)
    logistic_results["threshold_from_valid"] = logistic_threshold
    for split in ("train", "valid", "test"):
        collect_rows = split in {"valid", "test"}
        y, p, rows = predict_logistic(cases, split, logistic, chunk_size, collect_rows=collect_rows)
        logistic_results["split_metrics"][split] = classification_metrics(y, p, logistic_threshold)
        if split in {"valid", "test"}:
            logistic_results.setdefault("strata", {})[split] = stratified_metrics(rows, y, p, logistic_threshold)
    metrics["models"]["logistic_sgd"] = logistic_results

    numeric = train_numeric_hgb(cases, paths["models"], tree_max_rows, random_state)
    numeric_results: dict[str, Any] = {"model_path": numeric["path"], "train_rows": numeric["train_rows"], "numeric_features": NUMERIC_FEATURES, "split_metrics": {}}
    valid_y, valid_p, _ = predict_numeric(cases, "valid", numeric)
    numeric_threshold = best_threshold(valid_y, valid_p)
    numeric_results["threshold_from_valid"] = numeric_threshold
    for split in ("train", "valid", "test"):
        collect_rows = split in {"valid", "test"}
        y, p, rows = predict_numeric(cases, split, numeric, collect_rows=collect_rows)
        numeric_results["split_metrics"][split] = classification_metrics(y, p, numeric_threshold)
        if split in {"valid", "test"}:
            numeric_results.setdefault("strata", {})[split] = stratified_metrics(rows, y, p, numeric_threshold)
    metrics["models"]["numeric_hgb"] = numeric_results

    (paths["reports"] / "model_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (paths["models"] / "feature_config.json").write_text(
        json.dumps(
            {
                "leakage_fields_excluded_from_model": sorted(LEAKAGE_FIELDS),
                "numeric_features": NUMERIC_FEATURES,
                "hash_n_features": n_features,
                "split_policy": SPLIT_POLICY,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics


def scan_feature_stats(cases: list[Path]) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "total": 0,
        "by_quarter": defaultdict(lambda: {"n": 0, "positive": 0}),
        "by_split": defaultdict(lambda: {"n": 0, "positive": 0}),
        "missing": Counter(),
        "sums": Counter(),
        "weak_rules": {
            "coverage": 0,
            "conflicts": 0,
            "agreement": 0,
            "voted": 0,
            "rules": defaultdict(lambda: {"fires": 0, "positive_labels": 0}),
        },
    }
    missing_fields = ["age_years", "patientsex", "patientweight", "primarysourcecountry", "reportercountry", "qualification", "text_tokens"]
    for row in iter_feature_rows(cases):
        stats["total"] += 1
        y = int(row["label_serious"])
        quarter = row["quarter"]
        split = row["split"]
        stats["by_quarter"][quarter]["n"] += 1
        stats["by_quarter"][quarter]["positive"] += y
        stats["by_split"][split]["n"] += 1
        stats["by_split"][split]["positive"] += y
        for field in missing_fields:
            if not row.get(field):
                stats["missing"][field] += 1
        for field in ["drug_count", "drug_name_covered_count", "active_substance_count", "reaction_count", "indication_count"]:
            stats["sums"][field] += parse_float(row.get(field)) or 0.0
        update_weak_rule_stats(stats["weak_rules"], row, y)

    stats["by_quarter"] = dict(stats["by_quarter"])
    stats["by_split"] = dict(stats["by_split"])
    stats["missing"] = dict(stats["missing"])
    stats["sums"] = dict(stats["sums"])
    stats["weak_rules"]["rules"] = dict(stats["weak_rules"]["rules"])
    return stats


def update_weak_rule_stats(weak_stats: dict[str, Any], row: dict[str, str], label: int) -> None:
    tokens = row.get("text_tokens", "")
    age = parse_float(row.get("age_years"))
    drug_count = parse_float(row.get("drug_count")) or 0
    suspect_count = parse_float(row.get("suspect_drug_count")) or 0
    reaction_count = parse_float(row.get("reaction_count")) or 0

    rules = {
        "death_or_fatal_reaction_term": 1 if ("DEATH" in tokens or "FATAL" in tokens or "CARDIAC_ARREST" in tokens) else None,
        "high_polypharmacy_10plus": 1 if drug_count >= 10 else None,
        "senior_with_multiple_suspect_drugs": 1 if (age is not None and age >= 65 and suspect_count >= 2) else None,
        "low_complexity_younger_case": 0 if (age is not None and age < 50 and drug_count <= 2 and reaction_count <= 1) else None,
    }
    votes = [vote for vote in rules.values() if vote is not None]
    for name, vote in rules.items():
        if vote is not None:
            weak_stats["rules"][name]["fires"] += 1
            weak_stats["rules"][name]["positive_labels"] += label
    if not votes:
        return
    weak_stats["coverage"] += 1
    has_positive = any(vote == 1 for vote in votes)
    has_negative = any(vote == 0 for vote in votes)
    if has_positive and has_negative:
        weak_stats["conflicts"] += 1
        return
    weak_stats["voted"] += 1
    predicted = 1 if has_positive else 0
    if predicted == label:
        weak_stats["agreement"] += 1


def build_reports(out_dir: Path) -> dict[str, Any]:
    paths = ensure_dirs(out_dir)
    cases = feature_paths(paths["interim"])
    stats = scan_feature_stats(cases)
    metrics_path = paths["reports"] / "model_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    audit_md = render_data_audit(stats, metrics)
    summary_md = render_final_summary(stats, metrics)
    (paths["reports"] / "data_audit.md").write_text(audit_md, encoding="utf-8")
    (paths["reports"] / "final_summary.md").write_text(summary_md, encoding="utf-8")
    (paths["reports"] / "feature_audit.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    render_figures(stats, metrics, paths["figures"])
    return {"stats": stats, "metrics": metrics}


def pct(num: float, den: float) -> str:
    if den == 0:
        return "NA"
    return f"{100.0 * num / den:.2f}%"


def render_data_audit(stats: dict[str, Any], metrics: dict[str, Any]) -> str:
    total = stats["total"]
    drug_count = stats["sums"].get("drug_count", 0.0)
    covered = stats["sums"].get("drug_name_covered_count", 0.0)
    active = stats["sums"].get("active_substance_count", 0.0)
    lines = [
        "# 资料审计报告",
        "",
        "## 数据口径",
        "",
        "- 使用本地 `FAERS/faers_xml_2025q1` 至 `FAERS/faers_xml_2025q4` XML 数据。",
        "- 切分策略：2025Q1-Q3 为训练集；2025Q4 按 `receivedate` 前后 50/50 拆为验证集与测试集，同日用 `safetyreportid` 稳定排序。",
        "- 重症标签由 `serious == 1` 或任一 seriousness flag 为 `1` 生成；这些字段不进入模型特征。",
        "",
        "## 样本与标签",
        "",
        "| 分组 | 样本数 | 重症数 | 重症率 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for quarter, item in sorted(stats["by_quarter"].items()):
        lines.append(f"| {quarter} | {item['n']:,} | {item['positive']:,} | {pct(item['positive'], item['n'])} |")
    lines.extend(["", "## 训练切分", "", "| 切分 | 样本数 | 重症数 | 重症率 |", "| --- | ---: | ---: | ---: |"])
    for split, item in sorted(stats["by_split"].items()):
        lines.append(f"| {split} | {item['n']:,} | {item['positive']:,} | {pct(item['positive'], item['n'])} |")
    lines.extend(["", "## 缺失与正规化", "", "| 指标 | 数值 |", "| --- | ---: |"])
    for field, missing in sorted(stats["missing"].items()):
        lines.append(f"| `{field}` 缺失率 | {pct(missing, total)} |")
    lines.append(f"| 药物名称覆盖率 | {pct(covered, drug_count)} |")
    lines.append(f"| active substance 覆盖率 | {pct(active, drug_count)} |")

    weak = stats["weak_rules"]
    lines.extend(["", "## 弱监督规则审计", "", "| 指标 | 数值 |", "| --- | ---: |"])
    lines.append(f"| 规则覆盖率 | {pct(weak['coverage'], total)} |")
    lines.append(f"| 规则冲突率（相对覆盖样本） | {pct(weak['conflicts'], weak['coverage'])} |")
    lines.append(f"| 非冲突规则与标签一致率 | {pct(weak['agreement'], weak['voted'])} |")
    lines.extend(["", "| 规则 | 触发数 | 触发样本重症率 |", "| --- | ---: | ---: |"])
    for name, item in sorted(weak["rules"].items()):
        lines.append(f"| `{name}` | {item['fires']:,} | {pct(item['positive_labels'], item['fires'])} |")

    if metrics:
        lines.extend(["", "## 模型指标概览", "", "| 模型 | 切分 | AUROC | AUPRC | F1 | Recall@Top5% |", "| --- | --- | ---: | ---: | ---: | ---: |"])
        for model_name, model in metrics.get("models", {}).items():
            for split, item in model.get("split_metrics", {}).items():
                lines.append(
                    f"| {model_name} | {split} | {fmt_metric(item.get('auroc'))} | {fmt_metric(item.get('auprc'))} | "
                    f"{fmt_metric(item.get('f1'))} | {fmt_metric(item.get('recall_at_top_5pct'))} |"
                )
    lines.append("")
    return "\n".join(lines)


def render_final_summary(stats: dict[str, Any], metrics: dict[str, Any]) -> str:
    lines = [
        "# 基于本地 FAERS XML 的药物风险预测摘要",
        "",
        "本项目已按本地真实数据口径构建可复现流水线：XML 流式解析、删除列表过滤、病例级特征工程、数据质量审计、弱监督规则审计，以及两个 sklearn 基线模型。",
        "",
        "## 关键结果",
        "",
        f"- 病例级样本数：{stats['total']:,}",
        f"- 药物名称规则正规化覆盖率：{pct(stats['sums'].get('drug_name_covered_count', 0.0), stats['sums'].get('drug_count', 0.0))}",
        f"- 弱监督规则覆盖率：{pct(stats['weak_rules']['coverage'], stats['total'])}",
    ]
    if metrics:
        for model_name, model in metrics.get("models", {}).items():
            test = model.get("split_metrics", {}).get("test", {})
            lines.append(
                f"- `{model_name}` 测试集 AUROC={fmt_metric(test.get('auroc'))}, "
                f"AUPRC={fmt_metric(test.get('auprc'))}, Recall@Top5%={fmt_metric(test.get('recall_at_top_5pct'))}"
            )
    lines.extend(
        [
            "",
            "## 可复现命令",
            "",
            "```powershell",
            "python scripts/run_faers_pipeline.py --data FAERS --out outputs --mode full",
            "```",
            "",
            "若只做快速检查，可使用：",
            "",
            "```powershell",
            "python scripts/run_faers_pipeline.py --data FAERS --out outputs_sample --mode full --sample 1000",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def fmt_metric(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "NA"


def render_figures(stats: dict[str, Any], metrics: dict[str, Any], figure_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[report] matplotlib unavailable, skipping figures: {exc}")
        return

    quarters = sorted(stats["by_quarter"])
    rates = [stats["by_quarter"][q]["positive"] / stats["by_quarter"][q]["n"] for q in quarters]
    counts = [stats["by_quarter"][q]["n"] for q in quarters]
    plt.figure(figsize=(7, 4))
    plt.bar(quarters, rates, color="#4C78A8")
    plt.ylabel("Serious label rate")
    plt.title("Serious label rate by quarter")
    plt.tight_layout()
    plt.savefig(figure_dir / "label_rate_by_quarter.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(quarters, counts, color="#59A14F")
    plt.ylabel("Rows")
    plt.title("Case rows by quarter")
    plt.tight_layout()
    plt.savefig(figure_dir / "rows_by_quarter.png", dpi=150)
    plt.close()

    missing_fields = sorted(stats["missing"])
    missing_rates = [stats["missing"][field] / stats["total"] for field in missing_fields]
    plt.figure(figsize=(9, 4))
    plt.bar(missing_fields, missing_rates, color="#E15759")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Missing rate")
    plt.title("Missingness audit")
    plt.tight_layout()
    plt.savefig(figure_dir / "missing_rates.png", dpi=150)
    plt.close()

    if metrics:
        names = []
        aurocs = []
        auprcs = []
        for model_name, model in metrics.get("models", {}).items():
            test = model.get("split_metrics", {}).get("test", {})
            if test.get("auroc") is not None:
                names.append(model_name)
                aurocs.append(test.get("auroc"))
                auprcs.append(test.get("auprc"))
        if names:
            x = np.arange(len(names))
            width = 0.35
            plt.figure(figsize=(7, 4))
            plt.bar(x - width / 2, aurocs, width, label="AUROC", color="#4C78A8")
            plt.bar(x + width / 2, auprcs, width, label="AUPRC", color="#F28E2B")
            plt.xticks(x, names, rotation=20, ha="right")
            plt.ylim(0, 1)
            plt.legend()
            plt.title("Test metrics")
            plt.tight_layout()
            plt.savefig(figure_dir / "test_model_metrics.png", dpi=150)
            plt.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FAERS XML risk prediction pipeline")
    parser.add_argument("--data", default="FAERS", type=Path, help="Path to FAERS data directory")
    parser.add_argument("--out", default="outputs", type=Path, help="Output directory")
    parser.add_argument("--mode", choices=["inventory", "etl", "train", "report", "full"], default="full")
    parser.add_argument("--sample", type=int, default=None, help="Optional max kept reports per quarter for quick runs")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Training/evaluation chunk size for hashed model")
    parser.add_argument("--n-features", type=int, default=2**18, help="HashingVectorizer feature count")
    parser.add_argument("--tree-max-rows", type=int, default=200000, help="Max sampled train rows for numeric HGB")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    ensure_dirs(args.out)
    if args.mode in {"inventory", "full"}:
        run_inventory(args.data, args.out)
    if args.mode in {"etl", "full"}:
        run_etl(args.data, args.out, sample=args.sample)
    if args.mode in {"train", "full"}:
        train_models(args.out, args.chunk_size, args.n_features, args.tree_max_rows, args.random_state)
    if args.mode in {"report", "full"}:
        build_reports(args.out)
    print(f"[done] mode={args.mode} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
