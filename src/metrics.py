import numpy as np

from evaluation import *
from retrieve import *


def calculate_average_precision(relevant_doc_ids, retrieved_doc_ids):
    hit_count = 0
    sum_precisions = 0.0
    for i, doc_id in enumerate(retrieved_doc_ids):
        if doc_id in relevant_doc_ids:
            hit_count += 1
            precision_at_i = hit_count / (i + 1)
            sum_precisions += precision_at_i
        # else: sum_precisions += 0.0

    if len(relevant_doc_ids) == 0:
        return 0.0

    return sum_precisions / len(relevant_doc_ids)


def calculate_mean_average_precision(all_relevant_doc_ids, all_retrieved_doc_ids):
    average_precisions = []
    for relevant, retrieved in zip(all_relevant_doc_ids, all_retrieved_doc_ids):
        ap = calculate_average_precision(relevant, retrieved)
        average_precisions.append(ap)

    return {
        "map": (
            sum(average_precisions) / len(average_precisions)
            if average_precisions
            else 0.0
        )
    }


def calculate_precision_recall_f1_optimized(relevant_doc_ids, retrieved_doc_ids):
    relevant_set = set(relevant_doc_ids)
    retrieved_set = set(retrieved_doc_ids)
    true_positives = len(relevant_set.intersection(retrieved_set))

    if len(retrieved_set) == 0:
        precision = 0.0
        recall = 0.0 if len(relevant_set) > 0 else 1.0
        f1 = 0.0
    elif len(relevant_set) == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precision = true_positives / len(retrieved_set)
        recall = true_positives / len(relevant_set)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def calculate_macro_averages(metrics_per_query):
    precision_values = [metrics["precision"] for metrics in metrics_per_query]
    recall_values = [metrics["recall"] for metrics in metrics_per_query]
    f1_values = [metrics["f1"] for metrics in metrics_per_query]

    macro_precision = np.mean(precision_values)
    macro_recall = np.mean(recall_values)
    macro_f1 = np.mean(f1_values)

    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def calculate_micro_averages_optimized(all_relevant_doc_ids, all_retrieved_doc_ids):
    all_relevant = [
        doc_id for query_relevant in all_relevant_doc_ids for doc_id in query_relevant
    ]
    all_retrieved = [
        doc_id
        for query_retrieved in all_retrieved_doc_ids
        for doc_id in query_retrieved
    ]

    relevant_set = set(all_relevant)
    retrieved_set = set(all_retrieved)
    true_positives = len(relevant_set.intersection(retrieved_set))

    if len(retrieved_set) == 0:
        micro_precision = 0.0
        micro_recall = 0.0 if len(relevant_set) > 0 else 1.0
        micro_f1 = 0.0
    elif len(relevant_set) == 0:
        micro_precision = 0.0
        micro_recall = 1.0
        micro_f1 = 0.0
    else:
        micro_precision = true_positives / len(retrieved_set)
        micro_recall = true_positives / len(relevant_set)
        if micro_precision + micro_recall > 0:
            micro_f1 = (
                2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            )
        else:
            micro_f1 = 0.0

    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
    }
