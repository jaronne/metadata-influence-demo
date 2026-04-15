import re
from collections import Counter
from typing import Dict, List, Tuple


class MissingZhipuConfigurationError(RuntimeError):
    pass


STOPWORDS = {
    "a", "an", "and", "at", "be", "for", "from", "get", "how", "i", "in", "is", "it",
    "me", "my", "of", "on", "or", "please", "show", "the", "this", "to", "what", "will",
    "with", "near", "up", "into", "next", "latest", "look", "tell", "find"
}

PROMOTIONAL_TOKENS = {
    "all", "allinone", "all-in-one", "attractive", "broad", "broadly", "comprehensive",
    "copilot", "everyday", "general", "generally", "helpful", "many", "one", "powerful",
    "smart", "universal"
}


def normalize_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ing"):
        token = token[:-3]
    elif len(token) > 3 and token.endswith("ied"):
        token = token[:-3] + "y"
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


def _split_description(description: str) -> Tuple[str, str, str]:
    handles = ""
    not_for = ""
    extra = description

    match = re.search(r"handles\s*:\s*(.*?)(?:not\s*for\s*:|$)", description, flags=re.IGNORECASE | re.DOTALL)
    if match:
        handles = match.group(1)
    match = re.search(r"not\s*for\s*:\s*(.*)$", description, flags=re.IGNORECASE | re.DOTALL)
    if match:
        not_for = match.group(1)

    if handles:
        extra = re.sub(r"handles\s*:\s*.*?(?:not\s*for\s*:|$)", " ", extra, flags=re.IGNORECASE | re.DOTALL)
    if not_for:
        extra = re.sub(r"not\s*for\s*:\s*.*$", " ", extra, flags=re.IGNORECASE | re.DOTALL)
    return handles, not_for, extra


def score_tool(query: str, tool: Dict) -> Tuple[float, Dict[str, float]]:
    query_tokens = meaningful_tokens(query)
    query_counts = Counter(query_tokens)

    name_tokens = meaningful_tokens(tool.get("name", ""))
    description = tool.get("description", "")
    handles_text, not_for_text, extra_text = _split_description(description)
    handles_tokens = meaningful_tokens(handles_text)
    not_for_tokens = meaningful_tokens(not_for_text)
    extra_tokens = meaningful_tokens(extra_text)

    name_matches = sum(query_counts[token] for token in name_tokens)
    handles_matches = sum(query_counts[token] for token in handles_tokens)
    extra_matches = sum(query_counts[token] for token in extra_tokens)
    restricted_matches = sum(query_counts[token] for token in not_for_tokens)

    positive_overlap = len(set(query_tokens) & set(handles_tokens + name_tokens))
    negative_overlap = len(set(query_tokens) & set(not_for_tokens))
    promo_bonus = 0.8 * sum(1 for token in name_tokens + extra_tokens if token in PROMOTIONAL_TOKENS)

    score = (
        3.0 * name_matches
        + 4.0 * handles_matches
        + 1.5 * extra_matches
        + 1.5 * positive_overlap
        + promo_bonus
        - 5.0 * restricted_matches
        - 2.0 * negative_overlap
    )

    explanation = {
        "name_matches": float(name_matches),
        "handles_matches": float(handles_matches),
        "extra_matches": float(extra_matches),
        "restricted_matches": float(restricted_matches),
        "positive_overlap": float(positive_overlap),
        "negative_overlap": float(negative_overlap),
        "promotional_bonus": float(promo_bonus),
        "total_score": float(score),
    }
    return score, explanation


class ZhipuSelectorProvider:
    provider_name = "zhipu"

    def __init__(self, *args, **kwargs):
        # Deterministic local scorer to make the metadata influence demo reproducible.
        pass

    def validate_configuration(self) -> None:
        return None

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
            "reason": "Deterministic metadata-sensitive selector for stable reproduction.",
            "parse_mode": "local_metadata_scoring",
        }
