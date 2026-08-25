#!/usr/bin/env python3
"""Generate TRL-compatible conversational JSONL from vision prediction CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_CLASS_ORDER = ["Inductor", "Resistor", "Transformer", "solenoid"]
OUTPUT_FILES = {
    "train": "electronics_train.jsonl",
    "validation": "electronics_validation.jsonl",
    "test": "electronics_test.jsonl",
}
DEFAULT_CSV_CANDIDATES = [
    Path("vision_predictions_large_model.csv"),
    Path("vision_predictions_small_model.csv"),
    Path("vision_predictions.csv"),
]
QUESTION_VARIANTS = [
    "Interpret this vision result for a manufacturing operator.",
    "Summarize the predicted component and review decision.",
    "Return the component explanation and whether human review is required.",
    "Explain this classifier result using the required JSON schema.",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)

    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")

    return value


def load_threshold(path: Path) -> tuple[float, list[str]]:
    config = load_json(path)
    threshold = float(config.get("confidence_threshold", 0.7))

    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1.")

    class_order = config.get("class_order", DEFAULT_CLASS_ORDER)
    if not isinstance(class_order, list) or not all(isinstance(x, str) for x in class_order):
        raise ValueError("class_order must be a list of strings.")

    return threshold, class_order


def normalize_split(value: str) -> str:
    split = value.strip().lower()
    if split == "val":
        return "validation"
    return split


def parse_probability(row: dict[str, str], class_name: str) -> float:
    field = f"prob_{class_name}"
    if field not in row:
        raise KeyError(f"Missing required CSV column: {field}")
    return float(row[field])


def build_user_message(
    row: dict[str, str],
    class_order: list[str],
    threshold: float,
    question: str,
) -> str:
    probabilities = "\n".join(
        f"{class_name}={parse_probability(row, class_name):.4f}"
        for class_name in class_order
    )

    return f"""Vision model result:
predicted_class={row["predicted_label"]}
confidence={float(row["confidence"]):.4f}
confidence_threshold={threshold:.4f}

class_probabilities:
{probabilities}

Operator request:
{question}""".strip()


def build_target(
    row: dict[str, str],
    knowledge: dict[str, Any],
    threshold: float,
) -> str:
    predicted_label = row["predicted_label"]
    confidence = float(row["confidence"])
    requires_review = confidence < threshold

    component_info = knowledge.get(predicted_label, {})
    if not isinstance(component_info, dict):
        component_info = {}

    function = str(component_info.get("function", "N/A"))
    action_key = (
        "operator_action_if_uncertain"
        if requires_review
        else "operator_action_if_accepted"
    )
    operator_action = str(component_info.get(action_key, "Send the image to human review."))
    limitation = str(
        component_info.get(
            "limitation",
            "Visual classification does not verify electrical functionality.",
        )
    )

    target = {
        "identified_component": "uncertain" if requires_review else predicted_label,
        "predicted_component": predicted_label,
        "vision_confidence": round(confidence, 4),
        "confidence_threshold": round(threshold, 4),
        "requires_human_review": requires_review,
        "function": function,
        "operator_action": operator_action,
        "limitation": limitation,
    }

    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def build_record(
    row: dict[str, str],
    knowledge: dict[str, Any],
    class_order: list[str],
    threshold: float,
    row_index: int,
) -> dict[str, Any]:
    question = QUESTION_VARIANTS[row_index % len(QUESTION_VARIANTS)]

    system_message = (
        "You are an electronics manufacturing assistant. "
        "Interpret the output of a vision classifier for an operator. "
        "Return valid JSON only. "
        "Do not claim electrical functionality was verified from the image. "
        "If confidence is below the threshold, require human review."
    )

    return {
        "messages": [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": build_user_message(row, class_order, threshold, question),
            },
            {
                "role": "assistant",
                "content": build_target(row, knowledge, threshold),
            },
        ]
    }


def validate_csv_header(fieldnames: list[str] | None, class_order: list[str]) -> None:
    if fieldnames is None:
        raise ValueError("CSV file is empty.")

    required = {
        "image_path",
        "true_label",
        "predicted_label",
        "confidence",
        "split",
    }
    required.update(f"prob_{class_name}" for class_name in class_order)

    missing = sorted(required.difference(fieldnames))
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")


def get_output_files(output_suffix: str = "") -> dict[str, str]:
    return {
        split: filename.replace(".jsonl", f"{output_suffix}.jsonl")
        for split, filename in OUTPUT_FILES.items()
    }


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_csv_path(csv_path: Path | None) -> Path:
    if csv_path is not None:
        return csv_path

    existing = [path for path in DEFAULT_CSV_CANDIDATES if path.exists()]
    model_specific = [
        path
        for path in existing
        if path.name in {
            "vision_predictions_large_model.csv",
            "vision_predictions_small_model.csv",
        }
    ]

    if len(model_specific) == 1:
        return model_specific[0]

    if len(model_specific) > 1:
        names = ", ".join(path.name for path in model_specific)
        raise ValueError(
            "Multiple model-specific CSV files found: "
            f"{names}. Please specify one with --csv."
        )

    if len(existing) == 1:
        return existing[0]

    raise FileNotFoundError(
        "No prediction CSV found. Run inference.exe --batch-csv first, "
        "or pass --csv <path> explicitly."
    )


def generate_jsonl(
    csv_path: Path | None,
    knowledge_path: Path,
    threshold_path: Path,
    out_dir: Path,
    output_suffix: str = "",
) -> dict[str, int]:
    csv_path = resolve_csv_path(csv_path)
    threshold, class_order = load_threshold(threshold_path)
    knowledge = load_json(knowledge_path)

    grouped_records: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        validate_csv_header(reader.fieldnames, class_order)

        for row_index, row in enumerate(reader):
            split = normalize_split(row["split"])
            if split not in grouped_records:
                continue

            grouped_records[split].append(
                build_record(row, knowledge, class_order, threshold, row_index)
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    output_files = get_output_files(output_suffix)

    for split, filename in output_files.items():
        output_path = out_dir / filename
        write_jsonl(grouped_records[split], output_path)
        counts[split] = len(grouped_records[split])

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert vision prediction CSV into conversational JSONL files.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        type=Path,
        help="Input CSV generated by inference.exe --batch-csv.",
    )
    parser.add_argument(
        "--knowledge",
        default=Path("knowledge/component_knowledge.json"),
        type=Path,
        help="Component knowledge JSON path.",
    )
    parser.add_argument(
        "--threshold",
        default=Path("configs/threshold.json"),
        type=Path,
        help="Confidence threshold config JSON path.",
    )
    parser.add_argument(
        "--out-dir",
        default=Path("."),
        type=Path,
        help="Output directory for electronics_*.jsonl files.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix appended before .jsonl, for example _large_model or _small_model.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        counts = generate_jsonl(
            csv_path=args.csv,
            knowledge_path=args.knowledge,
            threshold_path=args.threshold,
            out_dir=args.out_dir,
            output_suffix=args.output_suffix,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    print("JSONL generation complete.")
    output_files = get_output_files(args.output_suffix)
    for split in ("train", "validation", "test"):
        print(f"{output_files[split]}: {counts[split]} records")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
