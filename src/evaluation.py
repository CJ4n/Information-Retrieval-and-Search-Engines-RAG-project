import time
from functools import partial
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from metrics import *
from retrieve import *


def evaluate_combination(
    combo,
    queries,
    recipes,
    recipe_ids,
    k_values,
    thresholds,
    vec_uni,
    vec_bi,
    X_uni,
    X_bi,
):
    i, j = combo
    k = k_values[i]
    threshold = thresholds[j]

    metrics = evaluate_ir_system(
        queries=queries,
        recipies=recipes,
        recipe_ids=recipe_ids,
        k=int(k),
        threshold=threshold,
        vec_uni=vec_uni,
        vec_bi=vec_bi,
        X_uni=X_uni,
        X_bi=X_bi,
    )

    return (i, j, metrics["macro_f1"])


def create_parameter_heatmap(
    queries, recipes, recipe_ids, vec_uni, vec_bi, X_uni, X_bi
):
    thresholds = np.arange(0.1, 0.30, 0.05)
    k_values = np.arange(20, 60, 5)
    total_combinations = len(k_values) * len(thresholds)
    f1_matrix = np.zeros((len(k_values), len(thresholds)))

    combinations = [
        (i, j) for i in range(len(k_values)) for j in range(len(thresholds))
    ]

    evaluate_func = partial(
        evaluate_combination,
        queries=queries,
        recipes=recipes,
        recipe_ids=recipe_ids,
        k_values=k_values,
        thresholds=thresholds,
        vec_uni=vec_uni,
        vec_bi=vec_bi,
        X_uni=X_uni,
        X_bi=X_bi,
    )

    num_processes = min(cpu_count(), total_combinations)
    print(f"Running parameter search using {num_processes} processes...")
    with Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(evaluate_func, combinations),
                total=total_combinations,
                desc="Evaluating combinations",
            )
        )

    for i, j, f1_score in results:
        f1_matrix[i, j] = f1_score

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        f1_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=[f"{t:.2f}" for t in thresholds],
        yticklabels=[f"{int(k)}" for k in k_values],
    )

    plt.title("Macro F1 Scores for Combinations of k and Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("k")
    plt.tight_layout()
    plt.savefig(f"ir_parameter_heatmap{int(time.time())}.png", dpi=300)
    plt.show()

    # Find the best combination
    best_i, best_j = np.unravel_index(f1_matrix.argmax(), f1_matrix.shape)
    best_k = k_values[best_i]
    best_threshold = thresholds[best_j]
    best_f1 = f1_matrix[best_i, best_j]

    print("\nBest parameter combination:")
    print(f"k = {int(best_k)}, threshold = {best_threshold:.2f}")
    print(f"Macro F1 = {best_f1:.4f}")

    return {
        "f1_matrix": f1_matrix,
        "best_k": int(best_k),
        "best_threshold": best_threshold,
        "best_f1": best_f1,
    }


def evaluate_ir_system(
    queries, recipies, recipe_ids, k, threshold, vec_uni, vec_bi, X_uni, X_bi
):
    metrics_per_query = []
    all_relevant_doc_ids = []
    all_retrieved_doc_ids = []

    for _, row in tqdm(queries.iterrows()):
        query_text = row["q"]
        relevant_doc_ids = [doc[0] for doc in row["r"]]

        results = retrieve_documents(
            query_text, recipies, recipe_ids, k, threshold, vec_uni, vec_bi, X_uni, X_bi
        )
        retrieved_doc_ids = [result[1] for result in results]
        query_metrics = calculate_precision_recall_f1_optimized(
            relevant_doc_ids, retrieved_doc_ids
        )
        metrics_per_query.append(query_metrics)

        all_relevant_doc_ids.append(relevant_doc_ids)
        all_retrieved_doc_ids.append(retrieved_doc_ids)

    macro_metrics = calculate_macro_averages(metrics_per_query)
    micro_metrics = calculate_micro_averages_optimized(
        all_relevant_doc_ids, all_retrieved_doc_ids
    )
    MAP_metric = calculate_mean_average_precision(
        all_relevant_doc_ids, all_retrieved_doc_ids
    )

    all_metrics = {**macro_metrics, **micro_metrics, **MAP_metric}
    return all_metrics
