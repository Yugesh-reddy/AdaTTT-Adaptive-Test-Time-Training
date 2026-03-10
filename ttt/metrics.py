"""
Evaluation metrics for the Efficient TTT VQA system.

Includes:
- VQA accuracy (top-1 match)
- Per-question-type accuracy breakdown
- Pareto frontier computation
- Gate routing statistics aggregation
"""

from typing import Any, Dict, List, Optional

import numpy as np


def vqa_accuracy(
    predictions: List[int],
    ground_truth: List[int],
) -> float:
    """Standard VQA accuracy metric (top-1).

    For simplicity (common practice): check if predicted answer
    matches the most-frequent ground truth answer.

    Args:
        predictions: List of predicted answer indices.
        ground_truth: List of ground truth answer indices.

    Returns:
        Accuracy in [0, 1].
    """
    if len(predictions) == 0:
        return 0.0
    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    return correct / len(predictions)


def accuracy_by_question_type(
    predictions: List[int],
    ground_truth: List[int],
    question_types: List[str],
) -> Dict[str, float]:
    """Break down accuracy by question type.

    Args:
        predictions: Predicted answer indices.
        ground_truth: Ground truth answer indices.
        question_types: Question type for each sample ("yes/no", "number", "other").

    Returns:
        Dict mapping question type to accuracy.
    """
    type_correct: Dict[str, int] = {}
    type_total: Dict[str, int] = {}

    for pred, gt, qtype in zip(predictions, ground_truth, question_types):
        type_total[qtype] = type_total.get(qtype, 0) + 1
        if pred == gt:
            type_correct[qtype] = type_correct.get(qtype, 0) + 1

    results = {}
    for qtype in sorted(type_total.keys()):
        total = type_total[qtype]
        correct = type_correct.get(qtype, 0)
        results[qtype] = correct / total if total > 0 else 0.0

    return results


def pareto_frontier(
    results_list: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compute Pareto-optimal configurations.

    A point is Pareto-optimal if no other point has both higher accuracy
    AND lower FLOPs.

    Args:
        results_list: List of dicts with keys:
            - "accuracy": float
            - "avg_flops": float (GFLOPs)
            - "config": str (description)

    Returns:
        Filtered list of Pareto-optimal points, sorted by FLOPs (ascending).
    """
    if not results_list:
        return []

    # Sort by FLOPs ascending
    sorted_results = sorted(results_list, key=lambda x: x["avg_flops"])

    pareto = []
    max_acc = -1.0

    for result in sorted_results:
        if result["accuracy"] > max_acc:
            pareto.append(result)
            max_acc = result["accuracy"]

    return pareto


def compute_gate_statistics(
    routing_infos: List[Dict[str, Any]],
    predictions: Optional[List[int]] = None,
    ground_truth: Optional[List[int]] = None,
    skip_masks: Optional[List[List[bool]]] = None,
) -> Dict[str, Any]:
    """Aggregate routing info across all batches.

    Args:
        routing_infos: List of routing_info dicts from AdaptiveRouter.
        predictions: All predicted answer indices (optional, for accuracy breakdown).
        ground_truth: All ground truth indices (optional).
        skip_masks: Flattened skip masks for each sample (optional).

    Returns:
        Dict with:
            total_skip, total_adapt, skip_rate,
            avg_confidence_skip, avg_confidence_adapt,
            accuracy_skip_group, accuracy_adapt_group (if predictions provided)
    """
    total_skip = 0
    total_adapt = 0
    all_conf_skip = []
    all_conf_adapt = []

    for info in routing_infos:
        total_skip += info["skip_count"]
        total_adapt += info["adapt_count"]

        conf = info["confidences"]
        mask = info["skip_mask"]

        if mask.any():
            all_conf_skip.extend(conf[mask].tolist())
        if (~mask).any():
            all_conf_adapt.extend(conf[~mask].tolist())

    total = total_skip + total_adapt
    stats: Dict[str, Any] = {
        "total_skip": total_skip,
        "total_adapt": total_adapt,
        "skip_rate": total_skip / total if total > 0 else 0.0,
        "avg_confidence_skip": float(np.mean(all_conf_skip)) if all_conf_skip else 0.0,
        "avg_confidence_adapt": float(np.mean(all_conf_adapt)) if all_conf_adapt else 0.0,
    }

    # If predictions are provided, compute per-group accuracy
    if predictions is not None and ground_truth is not None and skip_masks is not None:
        flat_skip = []
        for sm in skip_masks:
            flat_skip.extend(sm)

        skip_correct = 0
        skip_total = 0
        adapt_correct = 0
        adapt_total = 0

        for pred, gt, is_skip in zip(predictions, ground_truth, flat_skip):
            if is_skip:
                skip_total += 1
                if pred == gt:
                    skip_correct += 1
            else:
                adapt_total += 1
                if pred == gt:
                    adapt_correct += 1

        stats["accuracy_skip_group"] = skip_correct / skip_total if skip_total > 0 else 0.0
        stats["accuracy_adapt_group"] = adapt_correct / adapt_total if adapt_total > 0 else 0.0

    return stats


def mcnemar_test(
    base_predictions: List[int],
    ttt_predictions: List[int],
    ground_truth: List[int],
) -> Dict[str, Any]:
    """McNemar's test comparing base vs TTT predictions.

    Tests whether the two models disagree symmetrically.

    Args:
        base_predictions: Predicted indices from base model.
        ttt_predictions: Predicted indices from TTT model.
        ground_truth: Ground truth indices.

    Returns:
        Dict with b, c counts, chi2 statistic, and p-value.
    """
    from scipy.stats import chi2 as chi2_dist

    # b = base correct, TTT wrong
    # c = base wrong, TTT correct
    b = 0
    c = 0
    for base, ttt, gt in zip(base_predictions, ttt_predictions, ground_truth):
        base_correct = (base == gt)
        ttt_correct = (ttt == gt)
        if base_correct and not ttt_correct:
            b += 1
        elif not base_correct and ttt_correct:
            c += 1

    # McNemar's chi-squared statistic (with continuity correction)
    if b + c == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - chi2_dist.cdf(chi2, df=1)

    return {
        "b_base_correct_ttt_wrong": b,
        "c_base_wrong_ttt_correct": c,
        "chi2": float(chi2),
        "p_value": float(p_value),
        "significant_at_005": p_value < 0.05,
    }
