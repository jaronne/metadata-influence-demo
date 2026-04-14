from provider_factory import create_provider


def select_tool(query, tools, provider_name="mock"):
    provider = create_provider(provider_name)
    return provider.select_tool(query, tools)
