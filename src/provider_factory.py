from providers.mock_provider import MockSelectorProvider
from providers.zhipu_provider import ZhipuSelectorProvider


def create_provider(provider_name: str):
    normalized = provider_name.lower().strip()
    if normalized == "mock":
        return MockSelectorProvider()
    if normalized == "zhipu":
        return ZhipuSelectorProvider()
    raise ValueError(f"Unsupported provider: {provider_name}")
