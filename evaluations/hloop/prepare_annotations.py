"""
Prepare HLoop annotation datasets for cross-model validation.

Extracts LLM-generated concepts from Phi-3 and Gemma results (one-shot JSON config),
formats them as annotation questions following the existing HLoop patterns, and outputs
JSON files in the annotation tool format:
  [{"content": "...", "description": "..."}]
"""

import json
import re
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "datasets" / "results"
OUTPUT = REPO / "evaluations" / "hloop" / "cross_model" / "prompts"

MODELS = {
    "Phi-3-medium-4k-instruct": "Phi-3",
    "gemma-2-27b-it": "Gemma-2",
}

random.seed(42)


def parse_json_result(result_str: str) -> list[str]:
    """Parse a JSON array from an LLM result string, handling markdown fences and noise."""
    s = result_str.strip()
    # Strip markdown code fences
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    s = s.strip()

    # Try direct JSON parse
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except json.JSONDecodeError:
        pass

    # Try to find a JSON array within the string
    match = re.search(r"\[.*?\]", s, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            pass

    return []


def parse_commasep_result(result_str: str) -> list[str]:
    """Parse a comma-separated list from an LLM result string."""
    s = result_str.strip()
    # Remove markdown fences
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    # Remove numbered prefixes like "1. "
    s = re.sub(r"\d+\.\s*", "", s)
    # Split on commas or newlines
    items = re.split(r"[,\n]+", s)
    return [x.strip().strip('"').strip("'").strip() for x in items if x.strip().strip('"').strip("'").strip()]


def load_results(model: str, kb: str, subtask: str, fmt: str = "json") -> list[dict]:
    """Load one-shot result file for a given model/KB/subtask."""
    if kb == "framenet":
        # FrameNet has gloss/nogloss subdirectories
        base = RESULTS / "framenet" / "gloss" / model
        pattern = f"*oneshot_{fmt}_GLOSS*"
        # subtask is "adjectives", "nouns", or "verbs"
        files = sorted(base.glob(f"*{subtask}_oneshot_{fmt}_GLOSS*"))
    elif kb == "conceptnet":
        base = RESULTS / "conceptnet" / subtask / model
        files = sorted(base.glob(f"*oneshot_{subtask.lower()}_{fmt}*"))
    elif kb == "multialignet":
        base = RESULTS / "multialignet" / model
        files = sorted(base.glob(f"*{subtask}_oneshot_{fmt}*"))
    elif kb == "semagram":
        base = RESULTS / "semagram" / model
        files = sorted(base.glob(f"*oneshot_{fmt}*"))
    else:
        return []

    results = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    return results


def make_conceptnet_items(model: str) -> list[dict]:
    """Generate annotation items for ConceptNet (RelatedTo + UsedFor)."""
    items = []
    for relation in ["RelatedTo", "UsedFor"]:
        # Try JSON first, fall back to commasep
        results = load_results(model, "conceptnet", relation, "json")
        if not results:
            results = load_results(model, "conceptnet", relation, "commasep")
        parser = parse_json_result if results and '"result"' in json.dumps(results[0]) else parse_commasep_result

        for entry in results:
            concept = entry.get("concept_name", "").replace("_", " ")
            concepts = parse_json_result(entry["result"])
            if not concepts:
                concepts = parse_commasep_result(entry["result"])

            rel_text = "related to" if relation == "RelatedTo" else "used for"
            for c in concepts:
                if relation == "UsedFor":
                    content = f"Is {concept} used for {c}?"
                else:
                    content = f"Is {c} related to {concept}?"
                items.append({
                    "content": content,
                    "description": f"ConceptNet | {relation} | source: {concept} | generated: {c}"
                })
    return items


def make_framenet_items(model: str) -> list[dict]:
    """Generate annotation items for FrameNet (gloss, all POS)."""
    items = []
    for pos in ["adjectives", "nouns", "verbs"]:
        results = load_results(model, "framenet", pos, "json")
        if not results:
            results = load_results(model, "framenet", pos, "commasep")

        for entry in results:
            concept = entry.get("concept_name", "").replace("_", " ")
            concepts = parse_json_result(entry["result"])
            if not concepts:
                concepts = parse_commasep_result(entry["result"])

            # Extract gloss from prompt if present
            prompt = entry.get("prompt", "")
            gloss_match = re.search(r"Given the concept '([^']+)'", prompt)
            frame_desc = gloss_match.group(1) if gloss_match else concept

            for c in concepts:
                content = f"Is '{c}' associated with the frame '{frame_desc}'?"
                items.append({
                    "content": content,
                    "description": f"FrameNet | {pos} | frame: {concept} | generated: {c}"
                })
    return items


def make_semagram_items(model: str) -> list[dict]:
    """Generate annotation items for Semagram."""
    items = []
    results = load_results(model, "semagram", "", "json")
    if not results:
        results = load_results(model, "semagram", "", "commasep")

    for entry in results:
        category = entry.get("concept_name", "").replace("_", " ")
        criterion = entry.get("concept_criterion", "")
        concepts = parse_json_result(entry["result"])
        if not concepts:
            concepts = parse_commasep_result(entry["result"])

        for c in concepts:
            content = f"Are {c} from {category} that {criterion}?"
            items.append({
                "content": content,
                "description": f"Semagram | category: {category} | criterion: {criterion} | generated: {c}"
            })
    return items


def make_multialignet_items(model: str) -> list[dict]:
    """Generate annotation items for MultiAligNet."""
    items = []
    for pos in ["adjectives", "nouns", "verbs"]:
        results = load_results(model, "multialignet", pos, "json")
        if not results:
            results = load_results(model, "multialignet", pos, "commasep")

        for entry in results:
            concept = entry.get("concept_name", "").replace("_", " ")
            concepts = parse_json_result(entry["result"])
            if not concepts:
                concepts = parse_commasep_result(entry["result"])

            for c in concepts:
                content = f"Is '{c}' associated with '{concept}'?"
                items.append({
                    "content": content,
                    "description": f"MultiAligNet | {pos} | concept: {concept} | generated: {c}"
                })
    return items


def sample_items(items: list[dict], n: int) -> list[dict]:
    """Randomly sample n items, or return all if fewer than n."""
    if len(items) <= n:
        return items
    return random.sample(items, n)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for model_dir, model_label in MODELS.items():
        print(f"\nProcessing {model_label} ({model_dir})...")

        all_items = {}
        kb_generators = {
            "conceptnet": make_conceptnet_items,
            "framenet": make_framenet_items,
            "semagram": make_semagram_items,
            "multialignet": make_multialignet_items,
        }

        for kb, generator in kb_generators.items():
            items = generator(model_dir)
            all_items[kb] = items
            print(f"  {kb}: {len(items)} total items")

        # Write per-KB files (500 sampled items each, matching revision strategy)
        for kb, items in all_items.items():
            sampled = sample_items(items, 500)
            random.shuffle(sampled)
            outfile = OUTPUT / f"{model_label.lower()}_{kb}.json"
            with open(outfile, "w") as f:
                json.dump(sampled, f, indent=2, ensure_ascii=False)
            print(f"  -> wrote {outfile.name} ({len(sampled)} items)")

        # Write combined file
        combined = []
        for kb, items in all_items.items():
            combined.extend(sample_items(items, 500))
        random.shuffle(combined)
        outfile = OUTPUT / f"{model_label.lower()}_all.json"
        with open(outfile, "w") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        print(f"  -> wrote {outfile.name} ({len(combined)} items)")


if __name__ == "__main__":
    main()
