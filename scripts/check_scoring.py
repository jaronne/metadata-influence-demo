from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from defense.defense_policy import decide_action
from scoring.instability import compute_ins, shuffle_tools
from scoring.intent_support import compute_iss
from scoring.metadata_reliance import compute_mrs
from scoring.risk_score import compute_risk_score


def main() -> None:
    def approx_equal(left: float, right: float, tolerance: float = 1e-9) -> bool:
        return abs(left - right) <= tolerance

    assert compute_iss("weather_tool", []) == 0.0
    assert approx_equal(
        compute_iss("weather_tool", ["weather_tool", "weather_tool", "travel_tool"]),
        2 / 3,
    )

    assert compute_mrs("document_tool", []) == 0.0
    assert approx_equal(
        compute_mrs("document_tool", ["document_tool", "weather_tool", "weather_tool"]),
        1 - (1 / 3),
    )

    assert compute_ins("finance_tool", []) == 0.0
    assert approx_equal(
        compute_ins("finance_tool", ["finance_tool", "calendar_tool", "finance_tool"]),
        1 / 3,
    )

    tools = [
        {"tool_id": "weather_tool"},
        {"tool_id": "travel_tool"},
        {"tool_id": "document_tool"},
    ]
    shuffled_once = shuffle_tools(tools, seed=7)
    shuffled_twice = shuffle_tools(tools, seed=7)
    assert shuffled_once == shuffled_twice
    assert shuffled_once is not tools
    assert sorted(tool["tool_id"] for tool in shuffled_once) == sorted(
        tool["tool_id"] for tool in tools
    )

    assert compute_risk_score(iss=1.0, mrs=0.0, ins=0.0) == 0.0
    assert approx_equal(
        compute_risk_score(iss=0.5, mrs=0.6, ins=0.2),
        (0.5 * 0.6 + 0.5 * 0.2) / 0.6,
    )
    assert compute_risk_score(iss=0.0, mrs=1.0, ins=1.0) == 1.0

    assert decide_action(0.29) == "allow"
    assert decide_action(0.3) == "canonical_reselect"
    assert decide_action(0.69) == "canonical_reselect"
    assert decide_action(0.7) == "user_confirm"

    print("check_scoring.py: all checks passed")


if __name__ == "__main__":
    main()
