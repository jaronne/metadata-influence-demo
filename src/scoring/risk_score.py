def compute_risk_score(
    iss: float,
    mrs: float,
    ins: float,
    epsilon: float = 0.1,
) -> float:
    denominator = iss + epsilon
    risk_raw = (0.5 * mrs + 0.5 * ins) / denominator
    risk_score = min(1.0, risk_raw)
    return max(0.0, risk_score)
