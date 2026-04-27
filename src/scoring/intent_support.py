from typing import List


def compute_iss(original_prediction: str, rewrite_predictions: List[str]) -> float:
    if not rewrite_predictions:
        return 0.0

    keep_count = sum(1 for prediction in rewrite_predictions if prediction == original_prediction)
    return keep_count / len(rewrite_predictions)
