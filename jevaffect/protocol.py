"""JevAffect's five-class label and prediction-file protocol.

The competition describes the MOSEI thresholds as "one possible mapping";
this project adopts those thresholds. The competition's mapping table and
sample output use ``NEUT`` while its prose also contains ``NEU`` once. Confirm
that spelling with the organizers before a competition submission. This module
follows the mapping table and sample output and emits ``NEUT``.

``choice_confidence`` is only the requested linear rescaling of the largest
class probability. It is not accuracy, probability of correctness, or a claim
that model probabilities have been calibrated.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any, Union

__all__ = [
    "CLASSES",
    "SNEG",
    "WNEG",
    "NEUT",
    "WPOS",
    "SPOS",
    "MOSEI_SCORE_RANGE",
    "SIMS_V2_SCORE_TO_CLASS",
    "DEFAULT_SIMSV2_LABEL_TOLERANCE",
    "DEFAULT_PROBABILITY_SUM_TOLERANCE",
    "ProtocolError",
    "map_mosei_sentiment",
    "map_simsv2_label",
    "validate_probability_vector",
    "make_prediction_record",
    "export_predictions",
]

SNEG = "SNEG"
WNEG = "WNEG"
NEUT = "NEUT"
WPOS = "WPOS"
SPOS = "SPOS"
CLASSES = (SNEG, WNEG, NEUT, WPOS, SPOS)

MOSEI_SCORE_RANGE = (-3.0, 3.0)
SIMS_V2_SCORE_TO_CLASS = {
    -1.0: SNEG,
    -0.8: SNEG,
    -0.6: WNEG,
    -0.4: WNEG,
    -0.2: WNEG,
    0.0: NEUT,
    0.2: WPOS,
    0.4: WPOS,
    0.6: WPOS,
    0.8: SPOS,
    1.0: SPOS,
}

DEFAULT_SIMSV2_LABEL_TOLERANCE = 1e-6
DEFAULT_PROBABILITY_SUM_TOLERANCE = 1e-6


class ProtocolError(ValueError):
    """Raised when input violates the label or prediction protocol."""


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ProtocolError("{} must be a finite real number".format(name))
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ProtocolError("{} must be a finite real number".format(name)) from exc
    if not math.isfinite(converted):
        raise ProtocolError("{} must be a finite real number".format(name))
    return converted


def map_mosei_sentiment(score: Real) -> str:
    """Apply this project's adopted possible mapping for a MOSEI score in [-3, 3]."""
    value = _finite_real(score, "MOSEI sentiment score")
    minimum, maximum = MOSEI_SCORE_RANGE
    if not minimum <= value <= maximum:
        raise ProtocolError("MOSEI sentiment score must be within [-3, 3]")

    if value <= -2.0:
        return SNEG
    if value <= -0.5:
        return WNEG
    if value < 0.5:
        return NEUT
    if value < 2.0:
        return WPOS
    return SPOS


def map_simsv2_label(
    label: Real,
    *,
    tolerance: float = DEFAULT_SIMSV2_LABEL_TOLERANCE,
) -> str:
    """Map an allowed CH-SIMS v2 label value to the contest's five classes.

    Only the eleven values listed in the competition mapping are accepted.
    ``tolerance`` handles CSV floating-point representation drift; it must be
    smaller than half the 0.2 spacing between adjacent source values.
    """
    value = _finite_real(label, "CH-SIMS v2 label")
    tol = _finite_real(tolerance, "label tolerance")
    if tol < 0.0 or tol >= 0.1:
        raise ProtocolError("label tolerance must be in [0, 0.1)")

    matches = [
        source_value
        for source_value in SIMS_V2_SCORE_TO_CLASS
        if abs(value - source_value) <= tol
    ]
    if len(matches) != 1:
        raise ProtocolError(
            "CH-SIMS v2 label must match one of the competition's allowed values"
        )
    return SIMS_V2_SCORE_TO_CLASS[matches[0]]


def validate_probability_vector(
    probabilities: Sequence[Real],
    *,
    sum_tolerance: float = DEFAULT_PROBABILITY_SUM_TOLERANCE,
) -> tuple[float, ...]:
    """Validate and normalize one probability per class, in ``CLASSES`` order.

    Entries must be finite and within [0, 1]. Their sum may differ from one by
    at most ``sum_tolerance`` to allow floating-point accumulation drift; the
    returned tuple is normalized to sum to one.
    """
    if isinstance(probabilities, (str, bytes)) or not isinstance(probabilities, Sequence):
        raise ProtocolError("probabilities must be a sequence of five numbers")
    if len(probabilities) != len(CLASSES):
        raise ProtocolError("probabilities must contain exactly five values")

    values = tuple(
        _finite_real(value, "probability for {}".format(class_name))
        for class_name, value in zip(CLASSES, probabilities)
    )
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ProtocolError("each class probability must be within [0, 1]")

    tol = _finite_real(sum_tolerance, "probability sum tolerance")
    if tol < 0.0 or tol >= 1.0:
        raise ProtocolError("probability sum tolerance must be in [0, 1)")
    total = math.fsum(values)
    if abs(total - 1.0) > tol:
        raise ProtocolError("class probabilities must sum to one")
    if total <= 0.0:
        raise ProtocolError("class probabilities must have a positive sum")

    return tuple(value / total for value in values)


def _validate_id(sample_id: Any) -> str:
    if not isinstance(sample_id, str) or not sample_id or sample_id.isspace():
        raise ProtocolError("ID must be a non-empty string")
    return sample_id  # Preserve the full filename exactly as supplied.


def make_prediction_record(
    sample_id: str,
    probabilities: Sequence[Real],
) -> dict[str, Any]:
    """Create a prediction record with all five scores and derived fields.

    Ties use the first class in ``CLASSES`` order. The prediction is always the
    argmax; low confidence is recorded separately and never becomes abstention.
    """
    identifier = _validate_id(sample_id)
    normalized = validate_probability_vector(probabilities)
    max_index = max(range(len(CLASSES)), key=normalized.__getitem__)
    predicted_class = CLASSES[max_index]
    max_probability = normalized[max_index]
    choice_confidence = (max_probability - 0.2) / 0.8

    return {
        "ID": identifier,
        "Label": predicted_class,
        "probabilities": dict(zip(CLASSES, normalized)),
        "argmax": predicted_class,
        "max_probability": max_probability,
        "choice_confidence": choice_confidence,
    }


def _validated_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ProtocolError("each prediction record must be a mapping")
    identifier = _validate_id(record.get("ID"))
    scores = record.get("probabilities")
    if not isinstance(scores, Mapping) or set(scores) != set(CLASSES):
        raise ProtocolError("record probabilities must contain exactly the five class names")

    rebuilt = make_prediction_record(identifier, [scores[name] for name in CLASSES])
    if record.get("Label") != rebuilt["Label"]:
        raise ProtocolError("record Label must equal the probability argmax")
    if record.get("argmax") != rebuilt["argmax"]:
        raise ProtocolError("record argmax must equal the probability argmax")

    max_probability = _finite_real(record.get("max_probability"), "max_probability")
    choice_confidence = _finite_real(
        record.get("choice_confidence"), "choice_confidence"
    )
    if not math.isclose(max_probability, rebuilt["max_probability"], abs_tol=1e-12):
        raise ProtocolError("record max_probability does not match its probabilities")
    if not math.isclose(choice_confidence, rebuilt["choice_confidence"], abs_tol=1e-12):
        raise ProtocolError("record choice_confidence does not match its probabilities")
    return rebuilt


def _materialize_ids(expected_ids: Iterable[str]) -> list[str]:
    if isinstance(expected_ids, (str, bytes)):
        raise ProtocolError("expected_ids must be an iterable of IDs, not one string")
    try:
        identifiers = [_validate_id(value) for value in expected_ids]
    except TypeError as exc:
        raise ProtocolError("expected_ids must be an iterable of IDs") from exc
    if len(set(identifiers)) != len(identifiers):
        raise ProtocolError("expected_ids contains duplicate IDs")
    return identifiers


def export_predictions(
    records: Iterable[Mapping[str, Any]],
    expected_ids: Iterable[str],
    csv_path: Union[str, Path],
    low_confidence_jsonl_path: Union[str, Path],
    *,
    low_confidence_threshold: Real,
) -> dict[str, int]:
    """Write complete ``ID,Label`` CSV and separate low-confidence JSONL.

    Rows follow ``expected_ids`` order. Any duplicate, missing, unexpected, or
    invalid record fails before either output file is opened. All predictions,
    including records below the confidence threshold, remain in the CSV.
    """
    identifiers = _materialize_ids(expected_ids)
    threshold = _finite_real(low_confidence_threshold, "low-confidence threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ProtocolError("low-confidence threshold must be within [0, 1]")

    csv_file = Path(csv_path)
    jsonl_file = Path(low_confidence_jsonl_path)
    if csv_file.resolve() == jsonl_file.resolve():
        raise ProtocolError("CSV and JSONL output paths must be different")

    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        validated = _validated_record(record)
        identifier = validated["ID"]
        if identifier in by_id:
            raise ProtocolError("prediction records contain duplicate ID: {}".format(identifier))
        by_id[identifier] = validated

    expected_set = set(identifiers)
    unexpected = set(by_id) - expected_set
    missing = expected_set - set(by_id)
    if unexpected:
        raise ProtocolError("prediction records contain unexpected IDs: {}".format(sorted(unexpected)))
    if missing:
        raise ProtocolError("prediction records are missing IDs: {}".format(sorted(missing)))

    ordered = [by_id[identifier] for identifier in identifiers]
    low_confidence = [
        record
        for record in ordered
        if record["choice_confidence"] < threshold
    ]

    with csv_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("ID", "Label"), lineterminator="\n")
        writer.writeheader()
        writer.writerows({"ID": record["ID"], "Label": record["Label"]} for record in ordered)

    with jsonl_file.open("w", encoding="utf-8", newline="\n") as handle:
        for record in low_confidence:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")

    return {
        "predictions_written": len(ordered),
        "low_confidence_written": len(low_confidence),
    }
