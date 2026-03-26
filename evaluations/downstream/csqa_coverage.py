"""
CommonsenseQA Coverage Analysis.

Measures how much of the human-validated novel ConceptNet entries produced
by the pipeline (Meta-Llama-3-70B-Instruct-FP8, one-shot JSON) cover
answer choices that are not yet reachable via 1-hop in the original ConceptNet.

Methodology:
  1. Load CommonsenseQA (train + validation splits).
  2. For each question, extract the source concept and the correct answer concept.
  3. Normalize to ConceptNet format (lowercase, underscores).
  4. Build the original 1-hop RelatedTo neighbourhood from related_concepts.csv.
  5. Build the augmented neighbourhood: original + all generated entries from the
     primary model (novel status inferred from the HLoop annotation approval rate;
     entries that appear in original ConceptNet are excluded as non-novel).
  6. For each CSQA question whose source concept was prompted by our pipeline,
     compute baseline coverage (correct answer reachable in original ConceptNet)
     and augmented coverage (correct answer reachable after adding generated entries).
  7. Report statistics.
"""

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO / "datasets" / "results" / "conceptnet" / "RelatedTo"
PRIMARY_MODEL = "Meta-Llama-3-70B-Instruct-FP8"
CN_TRIPLES = REPO / "evaluations" / "baselines" / "data" / "related_concepts.csv"
OUTPUT_DIR = REPO / "evaluations" / "downstream"

# Human approval rate from the paper (novel candidates approved by annotators)
HUMAN_APPROVAL_RATE = 0.8665


def normalize(concept: str) -> str:
    """Normalize a concept to ConceptNet format: lowercase, underscores."""
    return concept.lower().strip().replace(" ", "_").replace("-", "_")


def parse_result(result_str: str) -> list[str]:
    """Parse a JSON array or comma-separated list from an LLM result string."""
    s = result_str.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    s = s.strip()
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(x).strip().lower() for x in parsed if str(x).strip()]
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*?\]", s, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [str(x).strip().lower() for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            pass
    # Fallback: comma-separated
    items = re.split(r"[,\n]+", s)
    return [x.strip().strip('"').strip("'").lower() for x in items if x.strip()]


def load_original_cn(path: Path) -> dict[str, set[str]]:
    """Load ConceptNet RelatedTo triples into a source→{targets} dict."""
    print("Loading original ConceptNet RelatedTo triples...")
    graph = defaultdict(set)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            c1 = normalize(row["concept1"])
            c2 = normalize(row["concept2"])
            graph[c1].add(c2)
            graph[c2].add(c1)   # RelatedTo is symmetric
    print(f"  Loaded {sum(len(v) for v in graph.values())} directed edges "
          f"for {len(graph)} concepts.")
    return graph


def load_generated_entries(results_dir: Path, model: str) -> dict[str, list[str]]:
    """Load all generated entries for the primary model (one-shot JSON results)."""
    model_dir = results_dir / model
    generated = defaultdict(list)
    pattern = "*oneshot*json*.jsonl"
    files = list(model_dir.glob(pattern))
    if not files:
        # Also try without the glob pattern
        files = list(model_dir.glob("*.jsonl"))
    print(f"Loading generated entries from {model} ({len(files)} files)...")
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                concept = normalize(entry.get("concept_name", ""))
                candidates = parse_result(entry.get("result", ""))
                for c in candidates:
                    c_norm = normalize(c)
                    if c_norm:
                        generated[concept].append(c_norm)
    total = sum(len(v) for v in generated.values())
    print(f"  Loaded {total} generated entries for {len(generated)} source concepts.")
    return generated


def load_csqa() -> list[dict]:
    """Load CommonsenseQA train and validation splits via HuggingFace datasets."""
    from datasets import load_dataset
    questions = []
    for split in ("train", "validation"):
        ds = load_dataset("tau/commonsense_qa", split=split)
        for item in ds:
            source = normalize(item["question_concept"])
            labels = item["choices"]["label"]
            texts = item["choices"]["text"]
            answer_key = item["answerKey"]
            # Map answer key to answer text
            answer_text = None
            for lbl, txt in zip(labels, texts):
                if lbl == answer_key:
                    answer_text = normalize(txt)
                    break
            if answer_text:
                questions.append({
                    "source": source,
                    "answer": answer_text,
                    "all_choices": [normalize(t) for t in texts],
                    "id": item["id"],
                })
    print(f"Loaded {len(questions)} CSQA questions (train + validation).")
    return questions


def run_coverage_analysis(
    questions: list[dict],
    cn_graph: dict[str, set[str]],
    generated: dict[str, list[str]],
) -> dict:
    """
    For each CSQA question whose source concept was prompted by our pipeline,
    determine whether the correct answer is reachable:
      a) in the original ConceptNet 1-hop neighbourhood
      b) in the augmented neighbourhood (original + all generated entries)
    """
    prompted_concepts = set(generated.keys())

    # Subset of questions whose source concept we prompted
    relevant = [q for q in questions if q["source"] in prompted_concepts]

    baseline_covered = 0      # correct answer in original ConceptNet
    augmented_covered = 0     # correct answer in augmented (original + generated)
    novel_covered = 0         # correct answer ONLY in generated (not in original)

    for q in relevant:
        src = q["source"]
        ans = q["answer"]

        in_original = ans in cn_graph.get(src, set())
        generated_for_src = set(generated.get(src, []))
        in_generated = ans in generated_for_src

        if in_original:
            baseline_covered += 1
            augmented_covered += 1
        elif in_generated:
            augmented_covered += 1
            novel_covered += 1

    total = len(relevant)
    return {
        "total_csqa_questions": len(questions),
        "prompted_source_concepts": len(prompted_concepts),
        "csqa_questions_with_prompted_source": total,
        "baseline_covered": baseline_covered,
        "augmented_covered": augmented_covered,
        "novel_only_covered": novel_covered,
        "baseline_coverage_pct": round(100 * baseline_covered / total, 2) if total else 0,
        "augmented_coverage_pct": round(100 * augmented_covered / total, 2) if total else 0,
        "coverage_gain_pct": round(100 * novel_covered / total, 2) if total else 0,
        "coverage_gain_absolute": novel_covered,
    }


def report(results: dict) -> None:
    print("\n" + "=" * 60)
    print("CommonsenseQA Coverage Analysis — Results")
    print("=" * 60)
    print(f"Total CSQA questions (train+val):         {results['total_csqa_questions']:>6}")
    print(f"Distinct prompted ConceptNet concepts:    {results['prompted_source_concepts']:>6}")
    print(f"CSQA questions with prompted source:      {results['csqa_questions_with_prompted_source']:>6}")
    print()
    print(f"Original ConceptNet 1-hop coverage:       {results['baseline_covered']:>6}  "
          f"({results['baseline_coverage_pct']:.1f}%)")
    print(f"Augmented ConceptNet 1-hop coverage:      {results['augmented_covered']:>6}  "
          f"({results['augmented_coverage_pct']:.1f}%)")
    print(f"Novel entries covering correct answer:    {results['novel_only_covered']:>6}  "
          f"({results['coverage_gain_pct']:.1f}%)")
    print()
    print(f"Coverage gain from augmentation:          +{results['coverage_gain_pct']:.1f} pp")
    print("=" * 60)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cn_graph = load_original_cn(CN_TRIPLES)
    generated = load_generated_entries(RESULTS_DIR, PRIMARY_MODEL)
    questions = load_csqa()

    results = run_coverage_analysis(questions, cn_graph, generated)
    report(results)

    out_path = OUTPUT_DIR / "csqa_coverage_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    main()
