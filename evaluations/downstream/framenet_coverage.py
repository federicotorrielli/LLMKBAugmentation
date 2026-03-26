"""
FrameNet Lexical Unit (LU) Coverage Analysis.

Measures how many novel LU candidates produced by the pipeline
(Meta-Llama-3-70B-Instruct-FP8, one-shot JSON, gloss condition)
expand the coverage of FrameNet beyond its existing 13,572 LUs,
and how many target word tokens in FrameNet's annotated exemplar
sentences would gain frame-level coverage from the augmented resource.

Methodology:
  1. Build the reference LU set from NLTK FrameNet 1.7:
     a set of (frame_name, lemma) pairs.
  2. Load generated entries from the primary model for all three POS
     types (adjectives, nouns, verbs) under the gloss condition.
     Frame names are extracted from the `frame_name` field (the part
     before the first ' - ' in the gloss-enriched concept name).
  3. Identify novel candidates: (frame_name, lemma) pairs NOT already
     in the reference LU set.
  4. Report: frames gaining ≥1 new LU candidate, new LUs per POS.
  5. Cross-reference with FrameNet's annotated exemplar sentences:
     for each annotated target word token, check whether the lemma
     already has an LU in the original FrameNet and in the augmented
     version.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import nltk

REPO = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO / "datasets" / "results" / "framenet"
PRIMARY_MODEL = "Meta-Llama-3-70B-Instruct-FP8"
OUTPUT_DIR = REPO / "evaluations" / "downstream"

POS_TYPES = ["adjectives", "nouns", "verbs"]


def normalize_lemma(word: str) -> str:
    """Lowercase and strip punctuation/whitespace."""
    return word.lower().strip().strip("'\".,;:!?()[]{}").strip()


def extract_frame_name(concept_name: str) -> str:
    """
    Extract the bare frame name from a gloss-enriched concept_name.
    Format: "FrameName - frame description here"
    We take everything before the first ' - '.
    Then normalize to match NLTK (replace spaces with underscores).
    """
    # concept_name may be "Abandonment - An Agent leaves..."
    # or "Abounding_with - A Location is filled..."
    if " - " in concept_name:
        name = concept_name.split(" - ", 1)[0].strip()
    else:
        name = concept_name.strip()
    # Normalize spaces → underscores to match NLTK frame names
    return name.replace(" ", "_")


def build_fn_lu_set() -> tuple[set[tuple[str, str]], dict[str, set[str]]]:
    """
    Build reference sets from NLTK FrameNet 1.7.
    Returns:
      lu_set:    set of (frame_name, lemma) for all existing LUs
      frame_lus: dict frame_name → set of lemmas
    """
    nltk.download("framenet_v17", quiet=True)
    from nltk.corpus import framenet as fn

    lu_set: set[tuple[str, str]] = set()
    frame_lus: dict[str, set[str]] = defaultdict(set)

    for lu in fn.lus():
        frame_name = lu.frame.name  # e.g., "Abandonment", "Abounding_with"
        # LU names like "run.v", "bank.n" — strip the POS suffix
        lemma = normalize_lemma(lu.name.rsplit(".", 1)[0])
        lu_set.add((frame_name, lemma))
        frame_lus[frame_name].add(lemma)

    print(f"Original FrameNet: {len(lu_set)} LU entries across {len(frame_lus)} frames.")
    return lu_set, dict(frame_lus)


def parse_result(result_str: str) -> list[str]:
    """Parse a JSON array or comma-separated list from an LLM result string."""
    s = result_str.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    s = s.strip()
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*?\]", s, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            pass
    items = re.split(r"[,\n]+", s)
    return [x.strip().strip('"').strip("'") for x in items if x.strip()]


def load_generated_fn_entries(results_dir: Path, model: str) -> dict[str, dict[str, list[str]]]:
    """
    Load generated FrameNet entries from the primary model (one-shot JSON gloss).
    Returns: pos → {frame_name → [generated lemmas]}
    Frame names are normalized to match NLTK format.
    """
    model_dir = results_dir / "gloss" / model
    generated: dict[str, dict[str, list[str]]] = {pos: defaultdict(list) for pos in POS_TYPES}

    for pos in POS_TYPES:
        files = sorted(model_dir.glob(f"*{pos}_oneshot*json*GLOSS*.jsonl"))
        if not files:
            files = sorted(model_dir.glob(f"*{pos}*oneshot*GLOSS*.jsonl"))
        count_entries = 0
        count_frames = 0
        for f in files:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    raw_name = entry.get("frame_name", entry.get("concept_name", ""))
                    frame = extract_frame_name(raw_name)
                    if not frame:
                        continue
                    candidates = parse_result(entry.get("result", ""))
                    for c in candidates:
                        lemma = normalize_lemma(c)
                        if lemma:
                            generated[pos][frame].append(lemma)
                            count_entries += 1
                    count_frames = len(generated[pos])
        print(f"  {pos}: {count_entries} generated entries for {count_frames} frames.")

    return generated


def compute_novel_lu_stats(
    lu_set: set[tuple[str, str]],
    frame_lus: dict[str, set[str]],
    generated: dict[str, dict[str, list[str]]],
) -> dict:
    """Identify novel LU candidates not in original FrameNet."""
    novel_by_pos: dict[str, set[tuple[str, str]]] = {}
    frames_gaining_coverage: set[str] = set()

    for pos, frame_dict in generated.items():
        novel_pairs: set[tuple[str, str]] = set()
        for frame, lemmas in frame_dict.items():
            for lemma in set(lemmas):  # deduplicate
                pair = (frame, lemma)
                if pair not in lu_set:
                    novel_pairs.add(pair)
                    frames_gaining_coverage.add(frame)
        novel_by_pos[pos] = novel_pairs

    all_novel = set().union(*novel_by_pos.values())
    print(f"  Total novel LU candidates: {len(all_novel)}")
    return {
        "original_lu_count": len(lu_set),
        "original_frame_count": len(frame_lus),
        "novel_lu_candidates_total": len(all_novel),
        "novel_lu_by_pos": {pos: len(pairs) for pos, pairs in novel_by_pos.items()},
        "frames_gaining_coverage": len(frames_gaining_coverage),
        "relative_lu_expansion_pct": round(100 * len(all_novel) / len(lu_set), 2),
    }


def compute_exemplar_coverage(
    lu_set: set[tuple[str, str]],
    frame_lus: dict[str, set[str]],
    generated: dict[str, dict[str, list[str]]],
) -> dict:
    """
    For each annotated target word token in FrameNet exemplar sentences,
    check whether it has LU coverage in original vs augmented FrameNet.
    """
    nltk.download("framenet_v17", quiet=True)
    from nltk.corpus import framenet as fn

    # Build augmented frame_lus (original + generated)
    aug_frame_lus: dict[str, set[str]] = {k: set(v) for k, v in frame_lus.items()}
    for pos, frame_dict in generated.items():
        for frame, lemmas in frame_dict.items():
            if frame not in aug_frame_lus:
                aug_frame_lus[frame] = set()
            for lemma in lemmas:
                aug_frame_lus[frame].add(normalize_lemma(lemma))

    total_targets = 0
    baseline_covered = 0
    augmented_covered = 0

    for sent in fn.exemplars():
        frame_obj = sent.get("frame")
        target_spans = sent.get("Target", [])
        text = sent.get("text", "")

        if not frame_obj or not target_spans or not text:
            continue

        frame_name = frame_obj.name  # e.g., "Abandonment"

        for start, end in target_spans:
            raw_word = text[start:end]
            lemma = normalize_lemma(raw_word)
            if not lemma:
                continue

            total_targets += 1
            pair = (frame_name, lemma)
            in_orig = pair in lu_set
            in_aug = lemma in aug_frame_lus.get(frame_name, set())

            if in_orig:
                baseline_covered += 1
                augmented_covered += 1
            elif in_aug:
                augmented_covered += 1

    return {
        "total_target_tokens": total_targets,
        "baseline_covered": baseline_covered,
        "augmented_covered": augmented_covered,
        "novel_covered": augmented_covered - baseline_covered,
        "baseline_pct": round(100 * baseline_covered / total_targets, 2) if total_targets else 0,
        "augmented_pct": round(100 * augmented_covered / total_targets, 2) if total_targets else 0,
        "coverage_gain_pp": round(
            100 * (augmented_covered - baseline_covered) / total_targets, 2
        ) if total_targets else 0,
    }


def report(novel_stats: dict, exemplar_stats: dict) -> None:
    print("\n" + "=" * 60)
    print("FrameNet LU Coverage Analysis — Results")
    print("=" * 60)
    print(f"Original FrameNet LUs:                    {novel_stats['original_lu_count']:>6}")
    print(f"Novel LU candidates (total, unique):      {novel_stats['novel_lu_candidates_total']:>6}  "
          f"(+{novel_stats['relative_lu_expansion_pct']:.1f}%)")
    print(f"  Adjectives:                             {novel_stats['novel_lu_by_pos'].get('adjectives', 0):>6}")
    print(f"  Nouns:                                  {novel_stats['novel_lu_by_pos'].get('nouns', 0):>6}")
    print(f"  Verbs:                                  {novel_stats['novel_lu_by_pos'].get('verbs', 0):>6}")
    print(f"Frames gaining ≥1 new LU candidate:       "
          f"{novel_stats['frames_gaining_coverage']:>6} / {novel_stats['original_frame_count']}")
    print()
    ex = exemplar_stats
    print("Exemplar sentence target word coverage:")
    print(f"  Total annotated target tokens:          {ex['total_target_tokens']:>6}")
    print(f"  Baseline (original FrameNet):           {ex['baseline_covered']:>6}  ({ex['baseline_pct']:.1f}%)")
    print(f"  Augmented (+ pipeline entries):         {ex['augmented_covered']:>6}  ({ex['augmented_pct']:.1f}%)")
    print(f"  Novel entries covering new tokens:      {ex['novel_covered']:>6}  "
          f"(+{ex['coverage_gain_pp']:.1f} pp)")
    print("=" * 60)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lu_set, frame_lus = build_fn_lu_set()

    print("\nLoading generated FrameNet entries...")
    generated = load_generated_fn_entries(RESULTS_DIR, PRIMARY_MODEL)

    print("\nComputing novel LU statistics...")
    novel_stats = compute_novel_lu_stats(lu_set, frame_lus, generated)

    print("\nComputing exemplar sentence coverage...")
    exemplar_stats = compute_exemplar_coverage(lu_set, frame_lus, generated)

    report(novel_stats, exemplar_stats)

    results = {"novel_lu_stats": novel_stats, "exemplar_coverage": exemplar_stats}
    out_path = OUTPUT_DIR / "framenet_coverage_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    main()
