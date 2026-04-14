import json
import os
import re
import time
from typing import Dict, List, Optional


DEFAULT_ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
DEFAULT_ZHIPU_MODEL = "glm-4-flash"


class MissingZhipuConfigurationError(RuntimeError):
    pass


def _extract_tool_id_from_text(text: str, tool_ids: List[str]) -> Optional[str]:
    lowered = text.lower()
    for tool_id in tool_ids:
        if tool_id.lower() in lowered:
            return tool_id
    return None


class ZhipuSelectorProvider:
    provider_name = "zhipu"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        self.api_key = api_key or os.getenv("ZHIPU_API_KEY", "").strip()
        self.base_url = (base_url or os.getenv("ZHIPU_BASE_URL", DEFAULT_ZHIPU_BASE_URL)).strip()
        self.model = (model or os.getenv("ZHIPU_MODEL", DEFAULT_ZHIPU_MODEL)).strip()
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None

    def validate_configuration(self) -> None:
        if not self.api_key:
            raise MissingZhipuConfigurationError(
                "ZHIPU_API_KEY is not configured. Set ZHIPU_API_KEY in your shell or create a local .env file from .env.example before running --provider zhipu."
            )

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is not installed. Run `pip install -r requirements.txt` before using --provider zhipu."
            ) from exc

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        return self._client

    def _build_messages(self, query: str, tools: List[Dict]) -> List[Dict[str, str]]:
        tool_lines = []
        for tool in tools:
            schema_keywords = ", ".join(tool["schema"].get("keywords", []))
            tool_lines.append(
                "\n".join(
                    [
                        f"tool_id: {tool['tool_id']}",
                        f"name: {tool['name']}",
                        f"description: {tool['description']}",
                        f"schema_keywords: {schema_keywords}",
                    ]
                )
            )

        system_prompt = (
            "You are a careful tool selector for a metadata influence demo. "
            "Choose exactly one tool_id from the candidate tools. "
            "Return JSON only with keys selected_tool and reason. "
            "Do not output markdown fences or extra text."
        )
        user_prompt = (
            "Query:\n"
            f"{query}\n\n"
            "Candidate tools:\n"
            f"{chr(10).join(tool_lines)}\n\n"
            "Required JSON format:\n"
            '{"selected_tool":"one_tool_id_here","reason":"short reason"}'
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_response_text(self, response_text: str, tool_ids: List[str]) -> Dict:
        try:
            parsed = json.loads(response_text)
            selected_tool = parsed.get("selected_tool")
            if selected_tool in tool_ids:
                return {
                    "selected_tool": selected_tool,
                    "reason": parsed.get("reason", ""),
                    "parse_mode": "json",
                    "fallback_marker": "",
                }
        except json.JSONDecodeError:
            pass

        quoted_match = re.search(r'"selected_tool"\s*:\s*"([^"]+)"', response_text)
        if quoted_match and quoted_match.group(1) in tool_ids:
            return {
                "selected_tool": quoted_match.group(1),
                "reason": "Recovered tool_id from partial JSON-like output.",
                "parse_mode": "partial_json_extraction",
                "fallback_marker": "partial_json_output",
            }

        extracted_tool = _extract_tool_id_from_text(response_text, tool_ids)
        if extracted_tool:
            return {
                "selected_tool": extracted_tool,
                "reason": "Recovered tool_id from non-JSON model output.",
                "parse_mode": "tool_id_extraction",
                "fallback_marker": "non_json_output",
            }

        return {
            "selected_tool": "FALLBACK_PARSE_FAILURE",
            "reason": "Could not parse a valid tool selection from the model output.",
            "parse_mode": "failed",
            "fallback_marker": "parse_failure",
        }

    def select_tool(self, query: str, tools: List[Dict]) -> Dict:
        self.validate_configuration()

        tool_ids = [tool["tool_id"] for tool in tools]
        last_error = None
        raw_response_text = ""

        for attempt in range(self.max_retries + 1):
            try:
                client = self._get_client()
                response = client.chat.completions.create(
                    model=self.model,
                    messages=self._build_messages(query, tools),
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                raw_response_text = response.choices[0].message.content or ""
                parsed = self._parse_response_text(raw_response_text, tool_ids)

                if parsed["selected_tool"] != "FALLBACK_PARSE_FAILURE":
                    return {
                        "selected_tool": parsed["selected_tool"],
                        "ranking": [],
                        "provider_used": self.provider_name,
                        "raw_response": raw_response_text,
                        "fallback_used": bool(parsed["fallback_marker"]),
                        "fallback_reason": parsed["fallback_marker"],
                        "reason": parsed["reason"],
                        "parse_mode": parsed["parse_mode"],
                    }

                last_error = RuntimeError(parsed["reason"])
            except Exception as exc:
                last_error = exc

            if attempt < self.max_retries:
                time.sleep(1.0 + attempt)

        raise RuntimeError(
            f"Zhipu provider failed after retries. Last error: {last_error}. Raw response: {raw_response_text}"
        )
