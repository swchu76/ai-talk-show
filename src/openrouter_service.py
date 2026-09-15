from collections.abc import Sequence
from dataclasses import dataclass
import time
from typing import Any

import requests

from src.config import AppConfig


FALLBACK_MODELS = [
    ("openrouter/free", "OpenRouter Free"),
    ("google/gemini-3.6-flash", "Google Gemini 3.6 Flash"),
    ("openai/gpt-5-mini", "OpenAI GPT-5 Mini"),
    ("anthropic/claude-sonnet", "Anthropic Claude Sonnet"),
]
RATE_LIMIT_MESSAGE = "OpenRouter 요청 제한에 도달했습니다. 잠시 후 다시 시도해주세요."


class OpenRouterError(Exception):
    def __init__(self, message: str, speaker: str = "", model: str = "", partial_text: str = "") -> None:
        super().__init__(message)
        self.speaker = speaker
        self.model = model
        self.partial_text = partial_text


class OpenRouterService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {config.api_key or ''}", "Content-Type": "application/json"})

    def list_models(self) -> list[tuple[str, str]]:
        if not self.config.has_api_key:
            return FALLBACK_MODELS
        try:
            response = self.session.get(f"{self.config.base_url}/models", timeout=10)
            response.raise_for_status()
            models = response.json().get("data", [])
            result = [(item["id"], item.get("name") or item["id"]) for item in models if item.get("id")]
            return result or FALLBACK_MODELS
        except requests.RequestException:
            return FALLBACK_MODELS

    def generate_debate(
        self,
        topic: str,
        host: dict[str, str],
        panelists: Sequence[dict[str, str]],
        messages: Sequence[dict[str, str]],
    ) -> str:
        if not self.config.has_api_key:
            return self._demo_debate(topic, host, panelists)
        transcript = self._format_messages(messages)
        sections: list[str] = []
        try:
            opening = self._call(host, self._host_opening_prompt(topic, host), transcript)
            sections.append(f"**{host['name']}:** {opening}")
            transcript = f"{transcript}\n\n사회자 {host['name']}: {opening}"

            for round_number in range(1, 4):
                sections.append(f"### {round_number}라운드")
                if round_number > 1:
                    transition = self._call(host, self._host_transition_prompt(round_number, host), transcript)
                    sections.append(f"**{host['name']}:** {transition}")
                    transcript = f"{transcript}\n\n사회자 {host['name']}: {transition}"

                for panel in panelists:
                    response = self._call(
                        panel,
                        self._panel_prompt(topic, round_number, panel, panelists, transcript),
                        transcript,
                    )
                    sections.append(f"**{panel['name']}:** {response}")
                    transcript = f"{transcript}\n\n패널 {panel['name']}: {response}"

            summary = self._call(host, self._summary_prompt(host, transcript), transcript)
            sections.append(f"### 전체 토론 요약\n**{host['name']}:** {summary}")
            return "\n\n".join(sections)
        except OpenRouterError as error:
            error.partial_text = "\n\n".join(sections)
            raise

    def _call(self, speaker: dict[str, str], prompt: str, context: str) -> str:
        model = speaker["model"]
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"지금까지의 토론 transcript:\n{context}"},
            ],
            "temperature": self.config.temperature,
            "max_tokens": 1800,
        }
        for attempt in range(2):
            try:
                response = self.session.post(f"{self.config.base_url}/chat/completions", json=payload, timeout=90)
            except requests.Timeout as error:
                raise OpenRouterError(f"{speaker['name']}의 모델 호출 시간이 초과되었습니다.", speaker['name'], model) from error
            except requests.ConnectionError as error:
                raise OpenRouterError(f"{speaker['name']}의 모델에 연결할 수 없습니다.", speaker['name'], model) from error
            except requests.RequestException as error:
                raise OpenRouterError(f"{speaker['name']}의 모델 호출에 실패했습니다.", speaker['name'], model) from error

            if response.status_code == 429:
                retry_after = self._retry_after(response)
                if attempt == 0 and retry_after is not None:
                    time.sleep(retry_after)
                    continue
                raise OpenRouterError(RATE_LIMIT_MESSAGE, speaker['name'], model)
            if response.status_code in (401, 403):
                raise OpenRouterError(f"{speaker['name']}의 모델 인증에 실패했습니다.", speaker['name'], model)
            if response.status_code >= 400:
                detail = self._error_detail(response)
                raise OpenRouterError(f"{speaker['name']}의 모델 호출에 실패했습니다: {detail}", speaker['name'], model)

            data = response.json()
            choice = (data.get("choices") or [{}])[0]
            finish_reason = choice.get("finish_reason")
            content = choice.get("message", {}).get("content") or ""
            if str(finish_reason).lower() in {"length", "max_tokens"}:
                raise OpenRouterError(f"{speaker['name']}의 응답이 토큰 제한으로 잘렸습니다.", speaker['name'], model, content)
            if not content.strip():
                raise OpenRouterError(f"{speaker['name']}의 모델이 빈 응답을 반환했습니다.", speaker['name'], model)
            return content.strip()
        raise OpenRouterError(RATE_LIMIT_MESSAGE, speaker["name"], model)

    @staticmethod
    def _host_opening_prompt(topic: str, host: dict[str, str]) -> str:
        return f"""당신은 {host['name']}인 토크쇼 진행자입니다.
페르소나: {host['persona']}
오늘의 주제는 '{topic}'입니다. 토론을 시작하는 중립적인 멘트를 2~3문장으로 작성하세요."""

    @staticmethod
    def _host_transition_prompt(round_number: int, host: dict[str, str]) -> str:
        return f"""당신은 {host['name']}인 토크쇼 진행자입니다.
페르소나: {host['persona']}
{round_number}라운드 시작 멘트를 1~2문장으로 작성하세요. 앞선 의견을 연결하고 이번 라운드의 질문을 제시하세요."""

    @staticmethod
    def _panel_prompt(topic: str, round_number: int, panel: dict[str, str], panelists: Sequence[dict[str, str]], transcript: str) -> str:
        other_names = ", ".join(item["name"] for item in panelists if item["name"] != panel["name"])
        return f"""당신은 '{panel['name']}'인 토크쇼 패널입니다.
페르소나: {panel['persona']}
주제: {topic}
라운드: {round_number}
다른 패널: {other_names}
이전 transcript의 발언을 참고해 동의, 반박, 질문 또는 보완을 포함한 발언을 작성하세요.
정확히 3~5문장으로 작성하고 이름이나 제목은 붙이지 마세요."""

    @staticmethod
    def _summary_prompt(host: dict[str, str], transcript: str) -> str:
        return f"""당신은 {host['name']}인 토크쇼 진행자입니다.
페르소나: {host['persona']}
전체 transcript를 읽고 핵심 쟁점, 합의점, 의견 차이와 실행 가능한 결론을 3~5문장으로 요약하세요."""

    @staticmethod
    def _format_messages(messages: Sequence[dict[str, str]]) -> str:
        return "\n\n".join(f"{item['role'].upper()}: {item['content']}" for item in messages)

    @staticmethod
    def _retry_after(response: requests.Response) -> float | None:
        try:
            value = float(response.headers.get("Retry-After", ""))
            return value if 0 < value <= 30 else None
        except ValueError:
            return None

    @staticmethod
    def _error_detail(response: requests.Response) -> str:
        try:
            return str(response.json().get("error", {}).get("message", response.text))[:240]
        except ValueError:
            return response.text[:240]

    @staticmethod
    def _demo_debate(topic: str, host: dict[str, str], panelists: Sequence[dict[str, str]]) -> str:
        sections = [f"**{host['name']}:** {topic}에 대한 토론을 시작하겠습니다."]
        for round_number in range(1, 4):
            sections.append(f"### {round_number}라운드")
            sections.extend(
                f"**{panel['name']}:** 데모 모드에서는 {panel['persona'].splitlines()[0]} 관점으로 의견을 제시합니다."
                for panel in panelists
            )
        sections.append(f"### 전체 토론 요약\n**{host['name']}:** 네 패널의 관점을 비교하고 다음 실행 과제를 정리할 수 있습니다.")
        return "\n\n".join(sections)


def model_label(model_id: str, models: Sequence[tuple[str, str]]) -> str:
    return next((label for item_id, label in models if item_id == model_id), model_id)