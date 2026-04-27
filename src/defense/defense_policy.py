def decide_action(risk_score: float) -> str:
    if risk_score < 0.3:
        return "allow"
    if risk_score < 0.7:
        return "canonical_reselect"
    return "user_confirm"
