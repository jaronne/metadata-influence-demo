from typing import List


def compute_mrs(original_prediction: str, canonical_predictions: List[str]) -> float:
    if not canonical_predictions:
        return 0.0

    keep_count = sum(1 for prediction in canonical_predictions if prediction == original_prediction)
    canonical_keep_rate = keep_count / len(canonical_predictions)
    return 1.0 - canonical_keep_rate
