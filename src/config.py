import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    model: str = "gemini-3.6-flash"
    api_key: str | None = None
    temperature: float = 0.8

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    @classmethod
    def from_environment(cls) -> "AppConfig":
        return cls(
            model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.8")),
        )
