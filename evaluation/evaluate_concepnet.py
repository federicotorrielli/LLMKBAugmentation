import os
import tqdm
import glob
import ujson
import eval_measures as em
import query_conceptnet as qc


def compute_model_scores(model_name, model_file, output_folder):
    p_1 = 0.0
    p_2 = 0.0
    p_5 = 0.0
    p_10 = 0.0

    h_1 = 0.0
    h_2 = 0.0
    h_5 = 0.0
    h_10 = 0.0

    acc_1 = 0.0
    acc_2 = 0.0
    acc_5 = 0.0
    acc_10 = 0.0

    rec_1 = 0.0
    rec_2 = 0.0
    rec_5 = 0.0
    rec_10 = 0.0

    mrr = 0.0

    num_q = 0.0

    with open(model_file, "r", encoding="utf-8") as reader:
        for line in reader:
            num_q += 1
            data = ujson.loads(line.strip())
            concepts = em.read_generated_concepts(data["result"])
            concepts_eval = qc.query_concept_1(data["concept"])

            if len(concepts) > 0 and concepts_eval and len(concepts_eval) > 0:
                p_1 += em.precision_at_k(concepts, concepts_eval, 1)
                p_2 += em.precision_at_k(concepts, concepts_eval, 2)
                p_5 += em.precision_at_k(concepts, concepts_eval, 5)
                p_10 += em.precision_at_k(concepts, concepts_eval, 10)

                h_1 += em.hits_at_k(concepts, concepts_eval, 1)
                h_2 += em.hits_at_k(concepts, concepts_eval, 2)
                h_5 += em.hits_at_k(concepts, concepts_eval, 5)
                h_10 += em.hits_at_k(concepts, concepts_eval, 10)

                acc_1 += em.accuracy_at_k(concepts, concepts_eval, 1)
                acc_2 += em.accuracy_at_k(concepts, concepts_eval, 2)
                acc_5 += em.accuracy_at_k(concepts, concepts_eval, 5)
                acc_10 += em.accuracy_at_k(concepts, concepts_eval, 10)

                rec_1 += em.recall_at_k(concepts, concepts_eval, 1)
                rec_2 += em.recall_at_k(concepts, concepts_eval, 2)
                rec_5 += em.recall_at_k(concepts, concepts_eval, 5)
                rec_10 += em.recall_at_k(concepts, concepts_eval, 10)

                mrr += em.MRR(concepts, concepts_eval)

    p_1 /= num_q
    p_2 /= num_q
    p_5 /= num_q
    p_10 /= num_q

    h_1 /= num_q
    h_2 /= num_q
    h_5 /= num_q
    h_10 /= num_q

    acc_1 /= num_q
    acc_2 /= num_q
    acc_5 /= num_q
    acc_10 /= num_q

    rec_1 /= num_q
    rec_2 /= num_q
    rec_5 /= num_q
    rec_10 /= num_q

    mrr /= num_q

    output_file = os.path.join(output_folder, f"{model_name}.txt")
    with open(output_file, "w") as writer:
        writer.write(f"P@1: {p_1}\n")
        writer.write(f"P@2: {p_2}\n")
        writer.write(f"P@5: {p_5}\n")
        writer.write(f"P@10: {p_10}\n")
        writer.write("\n============================\n")
        writer.write(f"H@1: {h_1}\n")
        writer.write(f"H@2: {h_2}\n")
        writer.write(f"H@5: {h_5}\n")
        writer.write(f"H@10: {h_10}\n")
        writer.write("\n============================\n")
        writer.write(f"ACC@1: {acc_1}\n")
        writer.write(f"ACC@2: {acc_2}\n")
        writer.write(f"ACC@5: {acc_5}\n")
        writer.write(f"ACC@10: {acc_10}\n")
        writer.write("\n============================\n")
        writer.write(f"R@1: {rec_1}\n")
        writer.write(f"R@2: {rec_2}\n")
        writer.write(f"R@5: {rec_5}\n")
        writer.write(f"R@10: {rec_10}\n")
        writer.write("\n============================\n")
        writer.write(f"MRR: {mrr}\n")


def read_model_files_and_write_results(folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    t = tqdm.tqdm()
    output_file = os.path.join(output_folder, "{0}.txt")

    for json_file in glob.glob(os.path.join(folder, "**/*.jsonl"), recursive=True):
        model_name = em.get_model_name(json_file)
        if model_name == "to_annotate" or os.path.exists(
            output_file.format(model_name)
        ):
            continue

        compute_model_scores(model_name, json_file, output_folder)

        t.update(1)


if __name__ == "__main__":
    read_model_files_and_write_results(
        "../results/conceptnet", "../results/conceptnet_scores"
    )
