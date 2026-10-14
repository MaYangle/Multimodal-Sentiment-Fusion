"""Strict NumPy-only metrics for fixed five-class JevAffect evaluation.

Class indices follow ``CLASS_NAMES``. Confusion matrices use true classes as
rows and predicted classes as columns. Macro-F1 always averages all five
classes; a class with no true or predicted examples contributes F1=0.

Temperature fitting is a bounded log-grid search over *validation* logits.
The caller is responsible for passing validation data only; test logits/labels
must never be used to select the temperature.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

__all__ = [
    "CLASS_NAMES",
    "NUM_CLASSES",
    "ECE_BINS",
    "MetricsError",
    "accuracy",
    "confusion_matrix",
    "per_class_metrics",
    "macro_f1",
    "classification_metrics",
    "negative_log_likelihood",
    "nll",
    "brier_score",
    "top_label_ece_15",
    "language_accuracy_summary",
    "temperature_scaled_probabilities",
    "fit_temperature_on_validation",
]

CLASS_NAMES = ("SNEG", "WNEG", "NEUT", "WPOS", "SPOS")
NUM_CLASSES = 5
ECE_BINS = 15
_PROBABILITY_SUM_TOLERANCE = 1e-6
_NLL_PROBABILITY_FLOOR = 1e-15


class MetricsError(ValueError):
    """Raised when data supplied to a metric violates its input contract."""


def _numeric_matrix(values: Any, name: str) -> np.ndarray:
    """Return finite real N-by-5 values; reject coercible strings and objects."""
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise MetricsError("{} must be a numeric N-by-5 matrix".format(name)) from exc
    if array.ndim != 2 or array.shape[1] != NUM_CLASSES or array.shape[0] == 0:
        raise MetricsError("{} must have shape (N, 5) with N > 0".format(name))
    if array.dtype.kind not in "iuf":
        raise MetricsError("{} must contain real numeric values".format(name))
    try:
        array = array.astype(np.float64, copy=False)
    except (OverflowError, TypeError, ValueError) as exc:
        raise MetricsError("{} must contain real numeric values".format(name)) from exc
    if not np.isfinite(array).all():
        raise MetricsError("{} must not contain NaN or Inf".format(name))
    return array


def _labels(values: Any, name: str = "labels") -> np.ndarray:
    """Validate a non-empty one-dimensional integer vector in [0, 4]."""
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise MetricsError("{} must be a one-dimensional integer vector".format(name)) from exc
    if array.ndim != 1 or array.size == 0:
        raise MetricsError("{} must be one-dimensional and non-empty".format(name))
    if array.dtype.kind not in "iu":
        raise MetricsError("{} must contain integer class indices".format(name))
    if np.any(array < 0) or np.any(array >= NUM_CLASSES):
        raise MetricsError("{} values must be in [0, 4]".format(name))
    return array.astype(np.int64, copy=False)


def _label_pair(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    true = _labels(y_true, "y_true")
    pred = _labels(y_pred, "y_pred")
    if true.shape != pred.shape:
        raise MetricsError("y_true and y_pred must have the same length")
    return true, pred


def _probability_inputs(
    y_true: Any, probabilities: Any
) -> tuple[np.ndarray, np.ndarray]:
    true = _labels(y_true, "y_true")
    probs = _numeric_matrix(probabilities, "probabilities")
    if probs.shape[0] != true.size:
        raise MetricsError("y_true and probabilities must have the same row count")
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise MetricsError("probabilities must be within [0, 1]")
    sums = probs.sum(axis=1)
    if np.any(sums <= 0.0) or np.any(np.abs(sums - 1.0) > _PROBABILITY_SUM_TOLERANCE):
        raise MetricsError("each probability row must sum to one within 1e-6")
    # Remove harmless float accumulation drift after enforcing the strict sum bound.
    probs = probs / sums[:, None]
    return true, probs


def _logits_inputs(y_true: Any, logits: Any) -> tuple[np.ndarray, np.ndarray]:
    true = _labels(y_true, "y_true")
    values = _numeric_matrix(logits, "logits")
    if values.shape[0] != true.size:
        raise MetricsError("y_true and logits must have the same row count")
    return true, values


def _finite_positive_scalar(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise MetricsError("{} must be a finite positive number".format(name))
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise MetricsError("{} must be a finite positive number".format(name)) from exc
    if not math.isfinite(result) or result <= 0.0:
        raise MetricsError("{} must be a finite positive number".format(name))
    return result


def accuracy(y_true: Any, y_pred: Any) -> float:
    """Return ordinary (micro/overall) five-class accuracy."""
    true, pred = _label_pair(y_true, y_pred)
    return float(np.mean(true == pred))


def confusion_matrix(y_true: Any, y_pred: Any) -> np.ndarray:
    """Return a fixed 5-by-5 confusion matrix (rows=true, columns=predicted)."""
    true, pred = _label_pair(y_true, y_pred)
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(matrix, (true, pred), 1)
    return matrix


def per_class_metrics(y_true: Any, y_pred: Any) -> tuple[dict[str, Any], ...]:
    """Return precision, recall, F1, and support for every fixed class index."""
    matrix = confusion_matrix(y_true, y_pred)
    true_support = matrix.sum(axis=1)
    predicted_support = matrix.sum(axis=0)
    true_positive = np.diag(matrix).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted_support,
        out=np.zeros(NUM_CLASSES, dtype=np.float64),
        where=predicted_support != 0,
    )
    recall = np.divide(
        true_positive,
        true_support,
        out=np.zeros(NUM_CLASSES, dtype=np.float64),
        where=true_support != 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(NUM_CLASSES, dtype=np.float64),
        where=(precision + recall) != 0,
    )
    return tuple(
        {
            "class_index": index,
            "class_name": CLASS_NAMES[index],
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(true_support[index]),
        }
        for index in range(NUM_CLASSES)
    )


def macro_f1(y_true: Any, y_pred: Any) -> float:
    """Return F1 averaged over all five classes, including absent classes."""
    class_rows = per_class_metrics(y_true, y_pred)
    return float(sum(row["f1"] for row in class_rows) / NUM_CLASSES)


def classification_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """Return accuracy, fixed-five macro-F1, confusion, and per-class metrics."""
    true, pred = _label_pair(y_true, y_pred)
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(matrix, (true, pred), 1)
    rows = per_class_metrics(true, pred)
    return {
        "accuracy": float(np.mean(true == pred)),
        "macro_f1": float(sum(row["f1"] for row in rows) / NUM_CLASSES),
        "confusion_matrix": matrix,
        "per_class": rows,
        "n_samples": int(true.size),
    }


def negative_log_likelihood(y_true: Any, probabilities: Any) -> float:
    """Return mean categorical NLL from five-class probabilities.

    A probability floor of 1e-15 keeps valid zero entries finite, matching
    common log-loss implementations. Inputs are otherwise strictly validated.
    """
    true, probs = _probability_inputs(y_true, probabilities)
    rows = np.arange(true.size)
    selected = np.maximum(probs[rows, true], _NLL_PROBABILITY_FLOOR)
    return float(-np.log(selected).mean())


# Short alias useful in metric tables and notebooks.
nll = negative_log_likelihood


def brier_score(y_true: Any, probabilities: Any) -> float:
    """Return mean per-example multiclass Brier score, summing all five classes."""
    true, probs = _probability_inputs(y_true, probabilities)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(true.size), true] = 1.0
    return float(np.square(probs - one_hot).sum(axis=1).mean())


def top_label_ece_15(y_true: Any, probabilities: Any) -> float:
    """Return top-label ECE with 15 fixed equal-width bins.

    Bins are [j/15, (j+1)/15), except that the final bin includes 1.0.
    """
    true, probs = _probability_inputs(y_true, probabilities)
    predicted = np.argmax(probs, axis=1)
    confidence = probs[np.arange(true.size), predicted]
    correct = (predicted == true).astype(np.float64)
    bin_index = np.minimum((confidence * ECE_BINS).astype(np.int64), ECE_BINS - 1)
    total = 0.0
    for index in range(ECE_BINS):
        selected = bin_index == index
        count = int(np.count_nonzero(selected))
        if count:
            total += (count / true.size) * abs(
                float(confidence[selected].mean()) - float(correct[selected].mean())
            )
    return float(total)


def language_accuracy_summary(
    y_true: Any,
    y_pred: Any,
    languages: Sequence[str],
    *,
    language_order: Sequence[str] = ("en", "zh"),
) -> dict[str, Any]:
    """Return per-language and overall accuracy plus equal-language accuracy.

    ``equal_language_accuracy`` is the unweighted mean of the accuracy for
    each requested language. By default both English (``en``) and Chinese
    (``zh``) are required; a missing or unexpected language is an error.
    """
    true, pred = _label_pair(y_true, y_pred)
    if isinstance(languages, (str, bytes)):
        raise MetricsError("languages must be a sequence with one value per sample")
    try:
        language_values = np.asarray(languages)
    except (TypeError, ValueError) as exc:
        raise MetricsError("languages must be a one-dimensional string sequence") from exc
    if language_values.ndim != 1 or language_values.size == 0:
        raise MetricsError("languages must be one-dimensional and non-empty")
    if language_values.size != true.size:
        raise MetricsError("languages and labels must have the same length")
    language_list = language_values.tolist()
    if any(not isinstance(value, str) or not value for value in language_list):
        raise MetricsError("each language must be a non-empty string")

    if isinstance(language_order, (str, bytes)):
        raise MetricsError("language_order must be a sequence of language names")
    try:
        order = tuple(language_order)
    except TypeError as exc:
        raise MetricsError("language_order must be a sequence of language names") from exc
    if len(order) < 2 or any(not isinstance(value, str) or not value for value in order):
        raise MetricsError("language_order must contain at least two non-empty names")
    if len(set(order)) != len(order):
        raise MetricsError("language_order must not contain duplicates")
    observed = set(language_list)
    expected = set(order)
    if observed != expected:
        raise MetricsError("observed languages must exactly match language_order")

    by_language: dict[str, dict[str, Any]] = {}
    values: list[float] = []
    for language in order:
        selected = np.fromiter(
            (value == language for value in language_list), dtype=np.bool_, count=true.size
        )
        subset_accuracy = float(np.mean(true[selected] == pred[selected]))
        count = int(np.count_nonzero(selected))
        by_language[language] = {"count": count, "accuracy": subset_accuracy}
        values.append(subset_accuracy)
    return {
        "n_samples": int(true.size),
        "overall_accuracy": float(np.mean(true == pred)),
        "equal_language_accuracy": float(sum(values) / len(values)),
        "per_language": by_language,
    }


def _logits_nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scaled = logits / temperature
        if not np.isfinite(scaled).all():
            raise MetricsError("scaled logits overflow; use a wider temperature bound")
        row_max = np.max(scaled, axis=1, keepdims=True)
        shifted = scaled - row_max
        if not np.isfinite(shifted).all():
            raise MetricsError("logit range is too large for finite NLL computation")
        log_partition = row_max[:, 0] + np.log(np.exp(shifted).sum(axis=1))
        selected = scaled[np.arange(labels.size), labels]
        losses = log_partition - selected
    if not np.isfinite(losses).all():
        raise MetricsError("logit NLL is not finite")
    return float(losses.mean())


def temperature_scaled_probabilities(logits: Any, temperature: float = 1.0) -> np.ndarray:
    """Convert N-by-5 logits to probabilities after positive temperature scaling."""
    values = _numeric_matrix(logits, "logits")
    temp = _finite_positive_scalar(temperature, "temperature")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scaled = values / temp
        if not np.isfinite(scaled).all():
            raise MetricsError("scaled logits overflow; use a wider temperature bound")
        shifted = scaled - np.max(scaled, axis=1, keepdims=True)
        if not np.isfinite(shifted).all():
            raise MetricsError("logit range is too large for finite softmax computation")
        exponentials = np.exp(shifted)
        probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all():
        raise MetricsError("softmax produced non-finite probabilities")
    return probabilities


def fit_temperature_on_validation(
    validation_logits: Any,
    validation_labels: Any,
    *,
    min_temperature: float = 0.05,
    max_temperature: float = 20.0,
    grid_size: int = 401,
) -> dict[str, float]:
    """Fit one scalar temperature with a bounded, log-spaced validation grid.

    Only the supplied validation logits/labels participate in selection. The
    grid is log-spaced and includes T=1 when it lies within the bounds, so the
    fitted validation NLL cannot be worse than the unscaled validation NLL.
    The returned mapping contains ``temperature``, ``before_nll``, and
    ``after_nll``. Do not pass test data to this function.
    """
    labels, logits = _logits_inputs(validation_labels, validation_logits)
    lower = _finite_positive_scalar(min_temperature, "min_temperature")
    upper = _finite_positive_scalar(max_temperature, "max_temperature")
    if lower >= upper:
        raise MetricsError("min_temperature must be less than max_temperature")
    if isinstance(grid_size, (bool, np.bool_)) or not isinstance(
        grid_size, (int, np.integer)
    ):
        raise MetricsError("grid_size must be an integer of at least 2")
    points = int(grid_size)
    if points < 2:
        raise MetricsError("grid_size must be an integer of at least 2")

    temperatures = np.geomspace(lower, upper, num=points, dtype=np.float64)
    if lower <= 1.0 <= upper:
        temperatures = np.unique(np.append(temperatures, 1.0))
    before = _logits_nll(logits, labels, 1.0)
    candidates = [
        (_logits_nll(logits, labels, float(temp)), float(temp))
        for temp in temperatures
    ]
    best_nll, best_temperature = min(
        candidates,
        key=lambda pair: (pair[0], abs(math.log(pair[1]))),
    )
    return {
        "temperature": best_temperature,
        "before_nll": before,
        "after_nll": best_nll,
    }
