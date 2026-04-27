import random
from typing import Dict, List


def shuffle_tools(tools: List[Dict], seed: int) -> List[Dict]:
    shuffled_tools = list(tools)
    rng = random.Random(seed)
    rng.shuffle(shuffled_tools)
    return shuffled_tools


def compute_ins(original_prediction: str, perturbation_predictions: List[str]) -> float:
    if not perturbation_predictions:
        return 0.0

    keep_count = sum(
        1 for prediction in perturbation_predictions if prediction == original_prediction
    )
    perturbation_keep_rate = keep_count / len(perturbation_predictions)
    return 1.0 - perturbation_keep_rate
