import re
from collections import Counter
from typing import Dict, List, Tuple


STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "be",
    "for",
    "from",
    "get",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "show",
    "the",
    "this",
    "to",
    "what",
    "will",
    "with",
}

PROMOTIONAL_TOKENS = {
    "all",
    "attractive",
    "broad",
    "broadly",
    "copilot",
    "everyday",
    "general",
    "generally",
    "helpful",
    "many",
    "powerful",
    "smart",
    "universal",
}


def normalize_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ing"):
        token = token[:-3]
    elif len(token) > 3 and token.endswith("ed"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("es"):
        token = token[:-2]
    elif len(token) > 2 and token.endswith("s"):
        token = token[:-1]
    return token


def tokenize(text: str) -> List[str]:
    return [normalize_token(token) for token in re.findall(r"[a-z0-9]+", text.lower())]


def meaningful_tokens(text: str) -> List[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS]


def score_tool(query: str, tool: Dict) -> Tuple[float, Dict[str, float]]:
    query_tokens = meaningful_tokens(query)
    query_counts = Counter(query_tokens)

    name_tokens = meaningful_tokens(tool["name"])
    desc_tokens = meaningful_tokens(tool["description"])
    schema_tokens = [normalize_token(token.lower()) for token in tool["schema"].get("keywords", [])]

    name_matches = sum(query_counts[token] for token in name_tokens)
    desc_matches = sum(query_counts[token] for token in desc_tokens)
    schema_matches = sum(query_counts[token] for token in schema_tokens)

    exact_schema_overlap = len(set(query_tokens) & set(schema_tokens))
    promotional_bonus = 0.6 * sum(
        1 for token in name_tokens + desc_tokens if token in PROMOTIONAL_TOKENS
    )

    score = (
        3.0 * name_matches
        + 2.0 * desc_matches
        + 4.0 * schema_matches
        + 1.5 * exact_schema_overlap
        + promotional_bonus
    )

    explanation = {
        "name_matches": float(name_matches),
        "description_matches": float(desc_matches),
        "schema_matches": float(schema_matches),
        "schema_overlap_bonus": float(exact_schema_overlap) * 1.5,
        "promotional_bonus": float(promotional_bonus),
        "total_score": float(score),
    }
    return score, explanation


class MockSelectorProvider:
    provider_name = "mock"

    def select_tool(self, query: str, tools: List[Dict]) -> Dict:
        scored = []
        for tool in tools:
            score, explanation = score_tool(query, tool)
            scored.append(
                {
                    "tool_id": tool["tool_id"],
                    "score": score,
                    "explanation": explanation,
                }
            )

        scored.sort(key=lambda item: (-item["score"], item["tool_id"]))
        return {
            "selected_tool": scored[0]["tool_id"],
            "ranking": scored,
            "provider_used": self.provider_name,
            "raw_response": None,
            "fallback_used": False,
            "fallback_reason": "",
            "reason": "Keyword-based mock selector result.",
            "parse_mode": "mock_scoring",
        }
