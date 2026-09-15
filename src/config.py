import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str | None = None
    temperature: float = 0.8

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    @classmethod
    def from_environment(cls) -> "AppConfig":
        return cls(
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=float(os.getenv("OPENROUTER_TEMPERATURE", "0.8")),
        )
