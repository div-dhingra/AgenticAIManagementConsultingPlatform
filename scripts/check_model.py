"""Explicit one-request provider smoke test. Never invoked during startup/health."""

import asyncio

from consulting_platform.agents.llm import HostedLLMClient, RequestBudget, generate
from consulting_platform.config import Settings
from consulting_platform.contracts import (
    ContractModel,
    ServiceUnavailableError,
    UpstreamModelError,
    UpstreamTimeoutError,
)


class SmokeResult(ContractModel):
    ready: bool


def main():
    settings = Settings.from_env()
    try:
        result = asyncio.run(
            generate(
                HostedLLMClient(settings),
                RequestBudget(limit=2),
                SmokeResult,
                role="router",
                system="Return a JSON object with ready=true.",
                payload={},
            )
        )
    except (ServiceUnavailableError, UpstreamTimeoutError, UpstreamModelError) as exc:
        raise SystemExit(f"{type(exc).__name__}: {exc}") from None
    if not result.ready:
        raise SystemExit("Provider responded but did not satisfy the smoke check.")
    print(f"PASS: {settings.provider} model access and structured JSON response.")


if __name__ == "__main__":
    main()
