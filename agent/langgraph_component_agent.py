#!/usr/bin/env python3
"""
Local LangGraph workflow for the electronic component diagnostic system.

This file corresponds to the Slide 44-48 design:
1. classify_component tool: call the local C++ ONNX Runtime executable.
2. get_component_knowledge tool: read verified local component knowledge.
3. confidence_route: keep the confidence threshold as deterministic logic.
4. human_review node: pause uncertain cases through LangGraph interrupt.
5. llm_response node: ask the local Ollama / Qwen3 model for diagnostic JSON.

Example:
  python agent/langgraph_component_agent.py \
    --image test/Inductor/new_file_137.jpeg \
    --variant large_model \
    --llm-variant large

Baseline comparison:
  python agent/langgraph_component_agent.py \
    --image test/Inductor/new_file_137.jpeg \
    --variant large_model \
    --baseline
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Literal, TypedDict

try:
    from langchain_core.tools import tool
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt

    try:
        from langgraph.checkpoint.memory import InMemorySaver
    except ImportError:  # Older LangGraph versions used MemorySaver.
        from langgraph.checkpoint.memory import MemorySaver as InMemorySaver
except ModuleNotFoundError as exc:
    missing_package = exc.name or "langgraph"
    print(
        f"Missing Python package: {missing_package}\n"
        "Install the local agent dependencies first, for example:\n\n"
        "  python -m pip install langgraph langchain-core\n",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["Inductor", "Resistor", "Transformer", "solenoid"]
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/v1/chat/completions"
DEFAULT_THRESHOLD_PATH = PROJECT_ROOT / "configs" / "threshold.json"
DEFAULT_KNOWLEDGE_PATH = PROJECT_ROOT / "knowledge" / "component_knowledge.json"
BASELINE_OLLAMA_MODEL = "llama3.2:1b"


class ComponentState(TypedDict, total=False):
    image_path: str
    model_variant: str
    model_path: str
    ollama_model: str
    ollama_url: str
    confidence_threshold: float
    vision_result: dict[str, Any]
    component_knowledge: dict[str, Any]
    human_decision: dict[str, Any]
    llm_raw_content: str
    diagnostic_json: dict[str, Any]


def project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_project_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return PROJECT_ROOT / path_obj


def load_confidence_threshold() -> float:
    if not DEFAULT_THRESHOLD_PATH.exists():
        return 0.70

    with DEFAULT_THRESHOLD_PATH.open("r", encoding="utf-8") as f:
        config = json.load(f)

    threshold = float(config.get("confidence_threshold", 0.70))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1.")
    return threshold


def model_path_for_variant(model_variant: str) -> Path:
    if model_variant == "small_model":
        return PROJECT_ROOT / "model" / "component_classifier_small.onnx"
    if model_variant == "large_model":
        return PROJECT_ROOT / "model" / "component_classifier_large.onnx"
    raise ValueError("model_variant must be small_model or large_model.")


def default_ollama_model(model_variant: str) -> str:
    if model_variant == "small_model":
        return "electronics-qwen3-4b-instruct-2507-small"
    if model_variant == "large_model":
        return "electronics-qwen3-4b-instruct-2507-large"
    raise ValueError("model_variant must be small_model or large_model.")


def ollama_model_for_llm_variant(llm_variant: str) -> str:
    if llm_variant == "small":
        return "electronics-qwen3-4b-instruct-2507-small"
    if llm_variant == "large":
        return "electronics-qwen3-4b-instruct-2507-large"
    raise ValueError("llm_variant must be small or large.")


@tool
def classify_component(
    image_path: str,
    model_variant: Literal["small_model", "large_model"] = "large_model",
) -> dict[str, Any]:
    """Run the local C++ ONNX component classifier and return JSON output."""

    image = resolve_project_path(image_path)
    if not image.exists():
        raise FileNotFoundError(image_path)

    executable = PROJECT_ROOT / "build" / "inference.exe"
    if not executable.exists():
        raise FileNotFoundError(
            "build/inference.exe was not found. Run `cmake --build build` first."
        )

    model_path = model_path_for_variant(model_variant)
    if not model_path.exists():
        raise FileNotFoundError(project_relative(model_path))

    result = subprocess.run(
        [
            str(executable),
            "--json",
            project_relative(model_path),
            project_relative(image),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )

    return json.loads(result.stdout)


@tool
def get_component_knowledge(
    component: Literal["Inductor", "Resistor", "Transformer", "solenoid"],
) -> dict[str, Any]:
    """Retrieve verified local component knowledge."""

    with DEFAULT_KNOWLEDGE_PATH.open("r", encoding="utf-8") as f:
        knowledge = json.load(f)

    return knowledge[component]


def confidence_route(state: ComponentState) -> str:
    confidence = float(state["vision_result"]["confidence"])
    threshold = float(state["confidence_threshold"])

    if confidence < threshold:
        return "human_review"

    return "llm_response"


def run_vision(state: ComponentState) -> dict[str, Any]:
    vision_result = classify_component.invoke(
        {
            "image_path": state["image_path"],
            "model_variant": state["model_variant"],
        }
    )

    return {
        "vision_result": vision_result,
        "confidence_threshold": load_confidence_threshold(),
    }


def load_knowledge(state: ComponentState) -> dict[str, Any]:
    component = state["vision_result"]["predicted_label"]
    if component not in CLASS_NAMES:
        raise ValueError(f"Unknown predicted component: {component}")

    knowledge = get_component_knowledge.invoke({"component": component})
    return {"component_knowledge": knowledge}


def human_review(state: ComponentState) -> dict[str, Any]:
    vision_result = state["vision_result"]
    threshold = state["confidence_threshold"]

    decision = interrupt(
        {
            "reason": "Low confidence prediction",
            "message": "Please approve, reject, or correct the predicted component.",
            "predicted_label": vision_result["predicted_label"],
            "confidence": vision_result["confidence"],
            "confidence_threshold": threshold,
            "probabilities": vision_result["probabilities"],
            "allowed_actions": ["approve", "reject", "correct"],
        }
    )

    return {"human_decision": decision}


def build_prompt(state: ComponentState) -> str:
    vision_result = state["vision_result"]
    knowledge = state["component_knowledge"]
    probabilities = vision_result["probabilities"]
    human_decision = state.get("human_decision")

    human_review_text = "none"
    if human_decision:
        human_review_text = json.dumps(human_decision, ensure_ascii=False)

    return (
        "Return valid JSON only. No markdown. No explanation.\n\n"
        "Use the required schema:\n"
        "{"
        '"identified_component": string, '
        '"predicted_component": string, '
        '"vision_confidence": number, '
        '"confidence_threshold": number, '
        '"requires_human_review": boolean, '
        '"function": string, '
        '"operator_action": string, '
        '"limitation": string'
        "}\n\n"
        "Vision model result:\n"
        f"image_path={vision_result['image_path']}\n"
        f"model_variant={state['model_variant']}\n"
        f"predicted_class={vision_result['predicted_label']}\n"
        f"confidence={float(vision_result['confidence']):.4f}\n"
        f"confidence_threshold={float(state['confidence_threshold']):.4f}\n\n"
        "class_probabilities:\n"
        f"Inductor={float(probabilities['Inductor']):.4f}\n"
        f"Resistor={float(probabilities['Resistor']):.4f}\n"
        f"Transformer={float(probabilities['Transformer']):.4f}\n"
        f"solenoid={float(probabilities['solenoid']):.4f}\n\n"
        "Verified component knowledge:\n"
        f"display_name={knowledge['display_name']}\n"
        f"function={knowledge['function']}\n"
        f"operator_action_if_accepted={knowledge['operator_action_if_accepted']}\n"
        f"operator_action_if_uncertain={knowledge['operator_action_if_uncertain']}\n"
        f"limitation={knowledge['limitation']}\n\n"
        "Human review decision:\n"
        f"{human_review_text}\n\n"
        "Policy:\n"
        "If confidence is below confidence_threshold, requires_human_review must be true. "
        "If confidence is equal to or above confidence_threshold, requires_human_review must be false."
    )


def call_ollama_chat(
    *,
    ollama_url: str,
    ollama_model: str,
    prompt: str,
    timeout_seconds: int = 60,
) -> str:
    payload = {
        "model": ollama_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an electronics manufacturing assistant. "
                    "Use the verified knowledge and deterministic threshold policy. "
                    "Return valid JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "stream": False,
    }

    request = urllib.request.Request(
        ollama_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    response_json = json.loads(body)
    return response_json["choices"][0]["message"]["content"]


def extract_json_object(raw_content: str) -> dict[str, Any]:
    content = raw_content.strip()

    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM response did not contain a JSON object.")

    parsed = json.loads(content[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON is not an object.")
    return parsed


def build_fallback_diagnostic(state: ComponentState, warning: str | None = None) -> dict[str, Any]:
    vision_result = state["vision_result"]
    knowledge = state["component_knowledge"]
    confidence = float(vision_result["confidence"])
    threshold = float(state["confidence_threshold"])
    requires_review = confidence < threshold

    diagnostic = {
        "identified_component": "uncertain" if requires_review else knowledge["display_name"],
        "predicted_component": vision_result["predicted_label"],
        "vision_confidence": round(confidence, 4),
        "confidence_threshold": round(threshold, 4),
        "requires_human_review": requires_review,
        "function": knowledge["function"],
        "operator_action": (
            knowledge["operator_action_if_uncertain"]
            if requires_review
            else knowledge["operator_action_if_accepted"]
        ),
        "limitation": knowledge["limitation"],
    }

    if warning:
        diagnostic["llm_warning"] = warning

    return diagnostic


def normalize_diagnostic_with_policy(
    state: ComponentState,
    diagnostic: dict[str, Any],
) -> dict[str, Any]:
    """Make safety-critical fields deterministic after the LLM responds.

    The LLM may phrase the diagnostic text, but it must not decide the
    threshold policy. A low-confidence sample required human review even if
    the operator later approved it.
    """

    vision_result = state["vision_result"]
    knowledge = state["component_knowledge"]
    confidence = float(vision_result["confidence"])
    threshold = float(state["confidence_threshold"])
    review_required = confidence < threshold
    predicted_component = vision_result["predicted_label"]
    human_decision = state.get("human_decision") or {}
    human_action = human_decision.get("action")

    normalized = dict(diagnostic)
    normalized.update(
        {
            "predicted_component": predicted_component,
            "vision_confidence": round(confidence, 4),
            "confidence_threshold": round(threshold, 4),
            "requires_human_review": review_required,
            "function": knowledge["function"],
            "limitation": knowledge["limitation"],
        }
    )

    if not review_required:
        normalized["identified_component"] = knowledge["display_name"]
        normalized["operator_action"] = knowledge["operator_action_if_accepted"]
        normalized["human_review_completed"] = False
        return normalized

    normalized["human_review_completed"] = bool(human_decision)

    if human_action == "approve":
        normalized["identified_component"] = predicted_component
        normalized["human_action"] = "approve"
        normalized["operator_action"] = (
            "Human review completed; operator approved the predicted component. "
            "Use the approved component label for downstream documentation."
        )
    elif human_action == "correct":
        corrected_component = human_decision.get("corrected_component", predicted_component)
        normalized["identified_component"] = corrected_component
        normalized["human_action"] = "correct"
        normalized["corrected_component"] = corrected_component
        normalized["operator_action"] = (
            "Human review completed; use the corrected component label "
            "for downstream documentation."
        )
    elif human_action == "reject":
        normalized["identified_component"] = "uncertain"
        normalized["human_action"] = "reject"
        normalized["operator_action"] = (
            "Human review completed; operator rejected the prediction. "
            "Do not use the predicted component label."
        )
    else:
        normalized["identified_component"] = "uncertain"
        normalized["operator_action"] = knowledge["operator_action_if_uncertain"]

    if human_decision.get("operator_note"):
        normalized["operator_note"] = human_decision["operator_note"]

    return normalized


def validate_diagnostic_schema(diagnostic: dict[str, Any]) -> None:
    required_fields = {
        "identified_component": str,
        "predicted_component": str,
        "vision_confidence": (int, float),
        "confidence_threshold": (int, float),
        "requires_human_review": bool,
        "function": str,
        "operator_action": str,
        "limitation": str,
    }

    for field, expected_type in required_fields.items():
        if field not in diagnostic:
            raise ValueError(f"Missing LLM diagnostic field: {field}")
        if not isinstance(diagnostic[field], expected_type):
            raise ValueError(f"Invalid type for LLM diagnostic field: {field}")


def generate_llm_response(state: ComponentState) -> dict[str, Any]:
    prompt = build_prompt(state)

    try:
        raw_content = call_ollama_chat(
            ollama_url=state["ollama_url"],
            ollama_model=state["ollama_model"],
            prompt=prompt,
        )
        diagnostic = extract_json_object(raw_content)
        validate_diagnostic_schema(diagnostic)
    except Exception as exc:  # Keep the workflow usable even if local LLM fails.
        raw_content = ""
        diagnostic = build_fallback_diagnostic(state, warning=str(exc))

    diagnostic = normalize_diagnostic_with_policy(state, diagnostic)

    return {
        "llm_raw_content": raw_content,
        "diagnostic_json": diagnostic,
    }


def build_graph():
    builder = StateGraph(ComponentState)

    builder.add_node("classify_component", run_vision)
    builder.add_node("get_component_knowledge", load_knowledge)
    builder.add_node("human_review", human_review)
    builder.add_node("llm_response", generate_llm_response)

    builder.add_edge(START, "classify_component")
    builder.add_edge("classify_component", "get_component_knowledge")
    builder.add_conditional_edges(
        "get_component_knowledge",
        confidence_route,
        {
            "human_review": "human_review",
            "llm_response": "llm_response",
        },
    )
    builder.add_edge("human_review", "llm_response")
    builder.add_edge("llm_response", END)

    return builder.compile(checkpointer=InMemorySaver())


def get_interrupt_payload(result: Any) -> Any | None:
    if not isinstance(result, dict) or "__interrupt__" not in result:
        return None

    interrupts = result["__interrupt__"]
    if not interrupts:
        return None

    first_interrupt = interrupts[0]
    return getattr(first_interrupt, "value", first_interrupt)


def ask_human_decision(payload: Any) -> dict[str, Any]:
    print("\n========== Human Review Required ==========")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("\nActions: approve, reject, correct")

    action = input("Action: ").strip().lower()
    while action not in {"approve", "reject", "correct"}:
        action = input("Please type approve, reject, or correct: ").strip().lower()

    decision: dict[str, Any] = {"action": action}

    if action == "correct":
        corrected = input("Corrected component label: ").strip()
        if corrected not in CLASS_NAMES:
            raise ValueError(f"Corrected component must be one of: {', '.join(CLASS_NAMES)}")
        decision["corrected_component"] = corrected

    note = input("Operator note (optional): ").strip()
    if note:
        decision["operator_note"] = note

    return decision


def print_result(result: ComponentState) -> None:
    vision_result = result["vision_result"]
    diagnostic = result["diagnostic_json"]

    print("\n========== LangGraph Diagnostic Workflow ==========")
    print(f"Image Path          : {vision_result['image_path']}")
    print(f"Vision Variant      : {result['model_variant']}")
    print(f"Vision Model Path   : {result['model_path']}")
    print(f"Ollama Model        : {result['ollama_model']}")
    print(f"Predicted Component : {vision_result['predicted_label']}")
    print(f"Top Confidence      : {float(vision_result['confidence']) * 100:.2f} %")
    print(f"Review Threshold    : {float(result['confidence_threshold']) * 100:.2f} %")
    print(f"Review Required     : {diagnostic['requires_human_review']}")
    print(f"Human Review Done   : {diagnostic.get('human_review_completed', False)}")
    if diagnostic.get("human_action"):
        print(f"Human Action        : {diagnostic['human_action']}")
    print("\nDiagnostic JSON")
    print(json.dumps(diagnostic, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local LangGraph component diagnostic workflow.",
        epilog=(
            "Examples:\n"
            "  Fine-tuned Qwen large with ONNX large:\n"
            "    python agent/langgraph_component_agent.py --image test/Inductor/new_file_137.jpeg "
            "--variant large_model --llm-variant large\n\n"
            "  Fine-tuned Qwen small with ONNX small:\n"
            "    python agent/langgraph_component_agent.py --image test/Resistor/new_file_111.jpeg "
            "--variant small_model --llm-variant small\n\n"
            "  Baseline LLM comparison:\n"
            "    python agent/langgraph_component_agent.py --image test/Inductor/new_file_137.jpeg "
            "--variant large_model --baseline\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", required=True, help="Input image path under the project root.")
    parser.add_argument(
        "--variant",
        "--vision-variant",
        dest="model_variant",
        choices=["small_model", "large_model"],
        default="large_model",
        help=(
            "ONNX vision classifier variant to run. "
            "This selects component_classifier_small.onnx or component_classifier_large.onnx. "
            "It is not the Qwen/Ollama model."
        ),
    )
    parser.add_argument(
        "--llm-variant",
        choices=["small", "large"],
        default=None,
        help=(
            "Fine-tuned Qwen/Ollama variant to use. "
            "small -> electronics-qwen3-4b-instruct-2507-small; "
            "large -> electronics-qwen3-4b-instruct-2507-large."
        ),
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help=f"Use baseline Ollama model {BASELINE_OLLAMA_MODEL} instead of fine-tuned Qwen.",
    )
    parser.add_argument(
        "--ollama-model",
        default=None,
        help=(
            "Explicit local Ollama model name. "
            "Use this to override --llm-variant or compare another model."
        ),
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_API_URL", DEFAULT_OLLAMA_URL),
        help="OpenAI-compatible Ollama chat completions endpoint.",
    )
    parser.add_argument(
        "--thread-id",
        default="component-diagnostic-demo",
        help="LangGraph checkpoint thread id.",
    )
    parser.add_argument(
        "--auto-review-action",
        choices=["approve", "reject"],
        default=None,
        help="Automatically resume low-confidence interrupts for non-interactive demos.",
    )
    args = parser.parse_args()
    explicit_llm_modes = [
        bool(args.ollama_model),
        bool(args.llm_variant),
        bool(args.baseline),
    ]
    if sum(explicit_llm_modes) > 1:
        parser.error("Choose only one of --ollama-model, --llm-variant, or --baseline.")

    return args


def choose_ollama_model(args: argparse.Namespace) -> str:
    if args.ollama_model:
        return args.ollama_model

    if args.baseline:
        return BASELINE_OLLAMA_MODEL

    if args.llm_variant:
        return ollama_model_for_llm_variant(args.llm_variant)

    env_model = os.environ.get("OLLAMA_MODEL")
    if env_model:
        return env_model

    return default_ollama_model(args.model_variant)


def main() -> int:
    args = parse_args()
    ollama_model = choose_ollama_model(args)

    graph = build_graph()
    config = {"configurable": {"thread_id": args.thread_id}}

    initial_state: ComponentState = {
        "image_path": args.image,
        "model_variant": args.model_variant,
        "model_path": project_relative(model_path_for_variant(args.model_variant)),
        "ollama_model": ollama_model,
        "ollama_url": args.ollama_url,
        "confidence_threshold": load_confidence_threshold(),
    }

    result = graph.invoke(initial_state, config=config)
    interrupt_payload = get_interrupt_payload(result)

    if interrupt_payload is not None:
        if args.auto_review_action:
            human_decision = {"action": args.auto_review_action}
        else:
            human_decision = ask_human_decision(interrupt_payload)

        result = graph.invoke(Command(resume=human_decision), config=config)

    print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
