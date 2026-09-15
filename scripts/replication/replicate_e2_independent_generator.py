import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq


EXP1_SUMMARY_PATH = Path(
    "benchmark_results/Exp1/formal/experiment_1_summary.json"
)
EXP2_RESULTS_PATH = Path(
    "benchmark_results/Exp2/formal/experiment_2_generation.jsonl"
)

AVAILABLE_CORPUS = Path(
    "benchmarks/Exp1/corpora/formal_v1/available"
)
ABSENT_CORPUS = Path(
    "benchmarks/Exp1/corpora/formal_v1/absent"
)

OUTPUT_ROOT = Path(
    "benchmark_results/independent_generator_replication"
)

MODEL_NAME = "qwen/qwen3.6-27b"
MAX_RETRIES = 0
MAX_OUTPUT_TOKENS = 64
INTER_CALL_SECONDS = 3

CONDITIONS = (
    "exposed",
    "available_not_exposed",
    "absent",
)

SYSTEM_PROMPT = """You are a claim-verification system.

Judge the claim using ONLY the supplied evidence.
Do not use outside knowledge.

Return exactly ONE word and nothing else:
TRUE
or
FALSE
"""

Verdict = Literal["TRUE", "FALSE"]

VERDICT_PATTERN = re.compile(
    r"^\s*(TRUE|FALSE)[.!]?\s*$",
    re.IGNORECASE,
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def ai_message_text(message) -> str:
    content = message.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)

    return str(content)


def parse_verdict(text: str) -> Verdict:
    match = VERDICT_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError(
            "Independent generator returned non-canonical verdict: "
            f"{text!r}"
        )
    return match.group(1).upper()  # type: ignore[return-value]


def truth_verdict(truth_label: bool) -> Verdict:
    return "TRUE" if truth_label else "FALSE"


def corpus_root(corpus_state: str) -> Path:
    if corpus_state == "available":
        return AVAILABLE_CORPUS
    if corpus_state == "absent":
        return ABSENT_CORPUS
    raise ValueError(f"Unknown corpus_state: {corpus_state}")


def load_context_texts(row: dict) -> list[str]:
    root = corpus_root(row["corpus_state"])
    source_files = list(row["context_source_files"])

    texts: list[str] = []
    for filename in source_files:
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Frozen context file missing: {path}"
            )
        texts.append(path.read_text(encoding="utf-8").strip())

    return texts


def run_case(*, llm: ChatGroq, row: dict) -> dict:
    query = row["query"]
    truth_label = bool(row["truth_label"])
    condition = row["condition"]
    source_files = list(row["context_source_files"])

    context_texts = load_context_texts(row)

    blocks = [
        f"[Evidence {rank}]\n{text}"
        for rank, text in enumerate(context_texts, start=1)
    ]
    context = "\n\n".join(blocks)

    prompt = (
        f"Claim:\n{query}\n\n"
        f"Evidence:\n{context}\n\n"
        "Determine whether the claim is TRUE or FALSE "
        "based only on the supplied evidence."
    )

    message = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )

    raw_output = ai_message_text(message)
    verdict = parse_verdict(raw_output)
    correct = verdict == truth_verdict(truth_label)

    return {
        "analysis": "independent_generator_replication",
        "source_experiment": "E2",
        "generator_model": MODEL_NAME,
        "generator_family": "qwen",
        "stimuli_selected_with_this_model": False,
        "stimuli_preflighted_with_this_model": False,
        "temperature": 0,
        "reasoning_effort": "none",
        "claim_id": row["claim_id"],
        "truth_label": truth_label,
        "query": query,
        "condition": condition,
        "corpus_state": row["corpus_state"],
        "context_source_files": source_files,
        "verdict": verdict,
        "correct": correct,
        "raw_model_output": raw_output,
    }


def summarize(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        grouped[row["condition"]].append(row)

    summary: dict[str, dict] = {}

    for condition in CONDITIONS:
        items = grouped[condition]
        if len(items) != 6:
            raise RuntimeError(
                f"Expected 6 rows for {condition}, got {len(items)}."
            )

        true_items = [
            item for item in items if item["truth_label"]
        ]
        false_items = [
            item for item in items if not item["truth_label"]
        ]

        correct = sum(item["correct"] for item in items)

        summary[condition] = {
            "n": len(items),
            "correct": correct,
            "accuracy": correct / len(items),
            "true_claim_accuracy": (
                sum(item["correct"] for item in true_items)
                / len(true_items)
            ),
            "false_claim_accuracy": (
                sum(item["correct"] for item in false_items)
                / len(false_items)
            ),
        }

    return summary


def paired_contrast(
    rows: list[dict],
    left: str,
    right: str,
) -> dict:
    index = {
        (row["claim_id"], row["condition"]): row
        for row in rows
    }
    claim_ids = sorted({row["claim_id"] for row in rows})

    switches = 0
    left_correct_right_wrong = 0
    left_wrong_right_correct = 0
    pairs: list[dict] = []

    for claim_id in claim_ids:
        a = index[(claim_id, left)]
        b = index[(claim_id, right)]

        switched = a["verdict"] != b["verdict"]
        deterioration = a["correct"] and not b["correct"]
        improvement = (not a["correct"]) and b["correct"]

        switches += int(switched)
        left_correct_right_wrong += int(deterioration)
        left_wrong_right_correct += int(improvement)

        pairs.append(
            {
                "claim_id": claim_id,
                "left_verdict": a["verdict"],
                "right_verdict": b["verdict"],
                "verdict_switched": switched,
                "left_correct": a["correct"],
                "right_correct": b["correct"],
            }
        )

    return {
        "n_claims": len(claim_ids),
        "verdict_switch_count": switches,
        "verdict_switch_rate": switches / len(claim_ids),
        "left_correct_right_wrong_count": (
            left_correct_right_wrong
        ),
        "left_wrong_right_correct_count": (
            left_wrong_right_correct
        ),
        "pairs": pairs,
    }


def validate_inputs(
    exp1: dict,
    exp2: list[dict],
) -> None:
    if exp1.get("formal_acceptance_pass") is not True:
        raise RuntimeError(
            "Formal E1 acceptance is not True."
        )

    if len(exp2) != 18:
        raise RuntimeError(
            f"Expected 18 frozen E2 rows, got {len(exp2)}."
        )

    counts: dict[str, int] = defaultdict(int)

    for row in exp2:
        condition = row.get("condition")
        if condition not in CONDITIONS:
            raise RuntimeError(
                f"Unexpected E2 condition: {condition!r}"
            )

        if row.get("formal_experiment") is not True:
            raise RuntimeError(
                "E2 row is not marked as a formal experiment."
            )

        counts[condition] += 1

    for condition in CONDITIONS:
        if counts[condition] != 6:
            raise RuntimeError(
                f"Expected 6 E2 rows for {condition}; "
                f"got {counts[condition]}."
            )


def main() -> None:
    exp1 = load_json(EXP1_SUMMARY_PATH)
    exp2 = load_jsonl(EXP2_RESULTS_PATH)

    validate_inputs(exp1, exp2)

    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=0,
        max_retries=MAX_RETRIES,
        max_tokens=MAX_OUTPUT_TOKENS,
        reasoning_effort="none",
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    condition_order = {
        condition: index
        for index, condition in enumerate(CONDITIONS)
    }
    exp2 = sorted(
        exp2,
        key=lambda row: (
            condition_order[row["condition"]],
            row["claim_id"],
        ),
    )

    print()
    print("INDEPENDENT GENERATOR REPLICATION — FROZEN E2")
    print("=" * 92)
    print(f"Generator:        {MODEL_NAME}")
    print("Model family:     Qwen")
    print("Temperature:      0")
    print("Reasoning effort: none")
    print("Fallbacks:        none")
    print("Stimulus selection on this model: NONE")
    print("Stimulus preflight on this model: NONE")
    print("Frozen source:    Formal E2")
    print()

    rows: list[dict] = []

    for row in exp2:
        result = run_case(llm=llm, row=row)
        rows.append(result)

        print(
            f"{result['condition']:<24}"
            f"{result['claim_id']:<10}"
            f"truth={str(result['truth_label']):<6}"
            f"verdict={result['verdict']:<6}"
            f"correct={result['correct']}"
        )

        time.sleep(INTER_CALL_SECONDS)

    condition_summary = summarize(rows)

    contrasts = {
        "exposed_vs_available_not_exposed": paired_contrast(
            rows,
            "exposed",
            "available_not_exposed",
        ),
        "exposed_vs_absent": paired_contrast(
            rows,
            "exposed",
            "absent",
        ),
    }

    detail_path = (
        OUTPUT_ROOT / "qwen_replication_results.jsonl"
    )
    with detail_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    summary = {
        "analysis": "independent_generator_replication",
        "formal_experiment": False,
        "analysis_of_frozen_formal_contexts": True,
        "source_experiment": "E2",
        "stimulus_version": exp1["stimulus_version"],
        "generator_model": MODEL_NAME,
        "generator_family": "qwen",
        "stimuli_selected_with_this_model": False,
        "stimuli_preflighted_with_this_model": False,
        "temperature": 0,
        "reasoning_effort": "none",
        "conditions": condition_summary,
        "contrasts": contrasts,
    }

    summary_path = (
        OUTPUT_ROOT / "qwen_replication_summary.json"
    )
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 92)
    print("SUMMARY")
    print("-" * 92)

    for condition in CONDITIONS:
        item = condition_summary[condition]
        print(
            f"{condition:<24}"
            f"{item['correct']}/{item['n']} correct "
            f"({item['accuracy']:.3f})"
        )

    print()
    print(f"Detailed results: {detail_path}")
    print(f"Summary:          {summary_path}")


if __name__ == "__main__":
    main()
