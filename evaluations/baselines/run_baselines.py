#!/usr/bin/env python3
"""External baselines for ConceptNet RelatedTo evaluation.

Evaluates GloVe, FastText, TransE, and ComplEx on the same 1000 prompt concepts
used for LLM evaluation, using identical metrics (P@10, R@10, F1@10, MRR).
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PROMPTS_PATH = Path(
    "datasets/prompts/conceptnet/RelatedTo/conceptnet_oneshot_relatedto_json.json"
)
GT_PATH = DATA_DIR / "related_concepts.csv"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def normalize(text):
    """Lowercase and replace underscores with spaces."""
    return text.lower().replace("_", " ").strip()


def load_ground_truth(csv_path):
    """Load ConceptNet RelatedTo edges as {normalized_concept: set(normalized)}."""
    gt = defaultdict(set)
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            c1, c2 = normalize(row[0]), normalize(row[1])
            gt[c1].add(c2)
            gt[c2].add(c1)  # undirected relation
    return gt


def load_prompt_concepts(json_path):
    """Return list of concept_name strings from prompt file."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return [item["concept_name"] for item in data]


# ---------------------------------------------------------------------------
# Metrics (matching paper definitions, Section 5.2)
# ---------------------------------------------------------------------------


def precision_at_k(predicted, gold, k):
    top_k = predicted[:k]
    if not top_k:
        return 0.0
    return sum(1 for p in top_k if p in gold) / len(top_k)


def recall_at_k(predicted, gold, k):
    if not gold:
        return 0.0
    top_k = predicted[:k]
    return sum(1 for p in top_k if p in gold) / len(gold)


def mrr(predicted, gold):
    for rank, item in enumerate(predicted, start=1):
        if item in gold:
            return 1.0 / rank
    return 0.0


def evaluate_predictions(predictions, ground_truth, k=10):
    """Compute average P@k, R@k, F1@k, MRR over all evaluated concepts."""
    p_scores, r_scores, mrr_scores = [], [], []

    for concept, pred_list in predictions.items():
        concept_norm = normalize(concept)
        gold = ground_truth.get(concept_norm, set())
        if not gold:
            continue

        pred_normalized = [normalize(p) for p in pred_list]

        p_scores.append(precision_at_k(pred_normalized, gold, k))
        r_scores.append(recall_at_k(pred_normalized, gold, k))
        mrr_scores.append(mrr(pred_normalized, gold))

    avg_p = float(np.mean(p_scores)) if p_scores else 0.0
    avg_r = float(np.mean(r_scores)) if r_scores else 0.0
    avg_f1 = (2 * avg_p * avg_r / (avg_p + avg_r)) if (avg_p + avg_r) > 0 else 0.0
    avg_mrr = float(np.mean(mrr_scores)) if mrr_scores else 0.0

    return {
        "P@10": round(avg_p, 2),
        "R@10": round(avg_r, 2),
        "F1@10": round(avg_f1, 2),
        "MRR": round(avg_mrr, 2),
        "num_evaluated": len(p_scores),
    }


# ---------------------------------------------------------------------------
# GloVe / FastText baselines
# ---------------------------------------------------------------------------


def run_embedding_baseline(model_key, concepts, ground_truth):
    """Retrieve top-10 nearest neighbors from a pre-trained embedding model."""
    import gensim.downloader as api

    model_map = {
        "glove": "glove-wiki-gigaword-300",
        "fasttext": "fasttext-wiki-news-subwords-300",
    }
    print(f"\nLoading {model_key} ({model_map[model_key]})...")
    model = api.load(model_map[model_key])

    predictions = {}
    skipped = 0

    for concept in tqdm(concepts, desc=f"{model_key}"):
        concept_norm = normalize(concept)
        query = None

        # Try underscore form, then space form
        if concept in model:
            query = concept
        elif concept_norm in model:
            query = concept_norm
        else:
            # Average component word vectors
            words = concept_norm.split()
            valid = [w for w in words if w in model]
            if valid:
                avg_vec = np.mean([model[w] for w in valid], axis=0)
                similar = model.similar_by_vector(avg_vec, topn=10)
                predictions[concept] = [w for w, _ in similar]
                continue
            skipped += 1
            continue

        similar = model.most_similar(query, topn=10)
        predictions[concept] = [w for w, _ in similar]

    print(f"  Skipped {skipped}/{len(concepts)} concepts (OOV)")
    return predictions


# ---------------------------------------------------------------------------
# Sentence-Transformer baseline
# ---------------------------------------------------------------------------


def run_sbert_baseline(concepts, ground_truth):
    """Retrieve top-10 nearest concepts using a sentence-transformer model."""
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    print("\nLoading sentence-transformer (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Build vocabulary from ground truth keys (same entity set KGE models use)
    vocab = list(ground_truth.keys())
    print(f"  Encoding {len(vocab)} vocabulary concepts...")
    vocab_embeddings = model.encode(vocab, batch_size=512, show_progress_bar=True)

    predictions = {}
    skipped = 0

    # Encode query concepts
    query_norms = [normalize(c) for c in concepts]
    print("  Encoding query concepts...")
    query_embeddings = model.encode(query_norms, batch_size=512, show_progress_bar=True)

    for i, concept in enumerate(tqdm(concepts, desc="SBERT")):
        concept_norm = query_norms[i]
        if concept_norm not in ground_truth:
            skipped += 1
            continue

        # Compute similarities and get top-11 (excluding self)
        sims = cosine_similarity([query_embeddings[i]], vocab_embeddings)[0]
        top_indices = np.argsort(sims)[::-1]

        results = []
        for idx in top_indices:
            if vocab[idx] != concept_norm and len(results) < 10:
                results.append(vocab[idx])
            if len(results) >= 10:
                break

        predictions[concept] = results

    print(f"  Skipped {skipped}/{len(concepts)} concepts")
    return predictions


# ---------------------------------------------------------------------------
# TransE / ComplEx baselines (PyKEEN)
# ---------------------------------------------------------------------------


def _load_triples(triples_path):
    """Load triples and create a TriplesFactory (cached across models)."""
    import torch
    from pykeen.triples import TriplesFactory

    print(f"\nLoading triples from {triples_path}...")
    triples = []
    with open(triples_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            triples.append([row[0], "RelatedTo", row[1]])
            triples.append([row[1], "RelatedTo", row[0]])  # symmetric relation

    # Deduplicate
    triples = list({tuple(t) for t in triples})
    triples_array = np.array(triples, dtype=str)
    print(f"  {len(triples_array)} triples loaded (symmetric)")

    tf = TriplesFactory.from_labeled_triples(triples_array)
    # PyKEEN pipeline requires a testing set. We use a 99/1 split so
    # nearly all triples are seen during training; our own metrics
    # (not PyKEEN's built-in evaluation) are used for scoring.
    training, testing = tf.split([0.99, 0.01], random_state=42)
    print(f"  Training: {training.num_triples}, Testing: {testing.num_triples}")
    return tf, training, testing


def run_kge_baseline(model_name, concepts, triples_path, tf=None, training=None, testing=None):
    """Train a KGE model on ConceptNet RelatedTo and predict top-10 tails."""
    import torch
    from pykeen.pipeline import pipeline
    from pykeen.predict import predict_target

    if tf is None:
        tf, training, testing = _load_triples(triples_path)

    print(f"  Training {model_name} (this may take a while)...")

    result = pipeline(
        model=model_name,
        training=training,
        testing=testing,
        epochs=50,
        model_kwargs={"embedding_dim": 200},
        training_kwargs={"batch_size": 4096},
        evaluation_kwargs={"batch_size": 4096},
        random_seed=42,
    )

    trained_model = result.model

    predictions = {}
    skipped = 0

    for concept in tqdm(concepts, desc=f"{model_name}"):
        if concept not in tf.entity_to_id:
            skipped += 1
            continue

        try:
            # Use PyKEEN's predict_target API which handles score semantics
            pred_df = predict_target(
                model=trained_model,
                head=concept,
                relation="RelatedTo",
                triples_factory=tf,
            ).df

            # Filter out self-predictions and take top 10
            pred_df = pred_df[pred_df["tail_label"] != concept].head(10)
            predictions[concept] = pred_df["tail_label"].tolist()
        except Exception as e:
            skipped += 1

    print(f"  Skipped {skipped}/{len(concepts)} concepts")
    return predictions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Loading ground truth...")
    ground_truth = load_ground_truth(GT_PATH)
    print(f"  {len(ground_truth)} concepts with relations")

    print("Loading prompt concepts...")
    concepts = load_prompt_concepts(PROMPTS_PATH)
    print(f"  {len(concepts)} prompt concepts")

    found = sum(1 for c in concepts if normalize(c) in ground_truth)
    print(f"  {found}/{len(concepts)} found in ground truth")

    results = {}

    # --- Distributional baselines ---
    for model_key in ["glove", "fasttext"]:
        preds = run_embedding_baseline(model_key, concepts, ground_truth)
        metrics = evaluate_predictions(preds, ground_truth)
        results[model_key] = metrics
        print(f"\n{model_key.upper()}: {metrics}")

    # --- Sentence-Transformer baseline ---
    preds = run_sbert_baseline(concepts, ground_truth)
    metrics = evaluate_predictions(preds, ground_truth)
    results["SBERT"] = metrics
    print(f"\nSBERT: {metrics}")

    # --- KGE baselines (share triples factory) ---
    tf, training, testing = _load_triples(GT_PATH)
    for model_name in ["TransE", "RotatE"]:
        preds = run_kge_baseline(model_name, concepts, GT_PATH, tf, training, testing)
        metrics = evaluate_predictions(preds, ground_truth)
        results[model_name] = metrics
        print(f"\n{model_name}: {metrics}")

    # --- Summary table ---
    print("\n" + "=" * 70)
    header = f"{'Method':<20} {'P@10':>6} {'R@10':>6} {'F1@10':>7} {'MRR':>6} {'#Eval':>6}"
    print(header)
    print("-" * 70)
    for name, m in results.items():
        print(
            f"{name:<20} {m['P@10']:>6.2f} {m['R@10']:>6.2f} "
            f"{m['F1@10']:>7.2f} {m['MRR']:>6.2f} {m['num_evaluated']:>6d}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
