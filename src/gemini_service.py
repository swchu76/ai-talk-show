from collections.abc import Sequence
from dataclasses import dataclass
import re
import time

from google import genai
from google.genai.errors import ClientError

try:
    from google.genai.errors import RateLimitError
except ImportError:
    RateLimitError = ClientError

from src.config import AppConfig


RATE_LIMIT_MESSAGE = "Gemini 무료 API 요청 제한에 도달했습니다. 잠시 후 다시 시도해주세요."


class TalkShowService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = genai.Client(api_key=config.api_key) if config.has_api_key else None

    def generate_debate(
        self,
        system_prompt: str,
        messages: Sequence[dict[str, str]],
    ) -> str:
        if self.client is None:
            return self._demo_debate(messages)

        speakers = self._speakers_from_prompt(system_prompt)
        transcript = self._build_contents(messages)
        completed_rounds: list[str] = []

        try:
            for round_number in range(1, 4):
                round_prompt = self._build_round_prompt(
                    system_prompt=system_prompt,
                    round_number=round_number,
                    instruction=self._round_instruction(round_number),
                    speakers=speakers[1:],
                    transcript=transcript,
                )
                round_text = self._generate_text(
                    prompt=round_prompt,
                    context=transcript,
                    max_output_tokens=3600,
                )
                self._validate_panel_output(round_text, speakers[1:])
                formatted_round = (
                    f"### {round_number}라운드\n"
                    f"**{speakers[0]}:** {self._round_intro(round_number)}\n\n"
                    f"{round_text.strip()}"
                )
                completed_rounds.append(formatted_round)
                transcript = f"{transcript}\n\n{formatted_round}"

            summary_prompt = self._build_summary_prompt(
                system_prompt=system_prompt,
                host=speakers[0],
                transcript=transcript,
            )
            summary = self._generate_text(
                prompt=summary_prompt,
                context=transcript,
                max_output_tokens=1800,
            )
            completed_rounds.append(f"### 전체 토론 요약\n**{speakers[0]}:** {summary.strip()}")
            return "\n\n".join(completed_rounds)
        except TalkShowRateLimitError as error:
            raise TalkShowRateLimitError(
                RATE_LIMIT_MESSAGE,
                partial_text="\n\n".join(completed_rounds),
            ) from error
        except TalkShowGenerationError as error:
            if completed_rounds:
                error.partial_text = "\n\n".join(completed_rounds)
            raise

    def _generate_text(self, prompt: str, context: str, max_output_tokens: int) -> str:
        interaction = self._create_interaction(
            prompt=prompt,
            context=context,
            max_output_tokens=max_output_tokens,
        )
        finish_reason = self._find_finish_reason(interaction)
        output_text = getattr(interaction, "output_text", None) or self._extract_output_text(interaction)
        if self._is_token_limit(finish_reason):
            raise TalkShowGenerationError(
                f"Gemini 응답이 토큰 제한으로 잘렸습니다 (finish_reason: {finish_reason}).",
                partial_text=output_text,
            )
        if not output_text:
            raise TalkShowGenerationError("Gemini가 빈 응답을 반환했습니다.")
        return output_text

    def _create_interaction(
        self,
        prompt: str,
        context: str,
        max_output_tokens: int,
    ) -> object:
        attempts = 0
        while True:
            try:
                return self.client.interactions.create(
                    model=self.config.model,
                    input=f"{context}\n\n현재 요청:\n{prompt}",
                    system_instruction=prompt,
                    generation_config={
                        "temperature": self.config.temperature,
                        "max_output_tokens": max_output_tokens,
                    },
                )
            except (RateLimitError, ClientError) as error:
                if not self._is_rate_limit_error(error):
                    raise
                retry_after = self._retry_after_seconds(error)
                if attempts == 0 and retry_after is not None:
                    attempts += 1
                    time.sleep(retry_after)
                    continue
                raise TalkShowRateLimitError(RATE_LIMIT_MESSAGE) from error

    @staticmethod
    def _build_round_prompt(
        system_prompt: str,
        round_number: int,
        instruction: str,
        speakers: Sequence[str],
        transcript: str,
    ) -> str:
        panel_lines = "\n".join(f"- {speaker}" for speaker in speakers)
        return f"""{system_prompt}

이번 요청은 {round_number}라운드 패널 토론입니다.
{instruction}
이번 라운드 패널:
{panel_lines}

이전 라운드 전체 내용:
{transcript}

반드시 아래 형식을 지키고, 패널 3명의 발언을 모두 출력하세요.
{speakers[0]}: 3~5문장 발언
{speakers[1]}: 3~5문장 발언
{speakers[2]}: 3~5문장 발언
이름과 발언 외의 진행 멘트, 제목, 설명은 출력하지 마세요."""

    @staticmethod
    def _build_summary_prompt(system_prompt: str, host: str, transcript: str) -> str:
        return f"""{system_prompt}

당신은 사회자 {host}입니다. 아래 3라운드 전체 토론을 바탕으로 최종 요약만 작성하세요.
핵심 쟁점, 패널 간 합의와 반박, 실천 가능한 결론, 남은 질문을 포함해 3~5문장으로 작성하세요.
이름이나 제목 없이 요약문만 출력하세요.

전체 토론:
{transcript}"""

    @staticmethod
    def _round_instruction(round_number: int) -> str:
        instructions = {
            1: "각자의 기본 입장과 근거를 제시하세요.",
            2: "이전 라운드 전체 내용을 참고해 다른 패널의 의견에 동의, 반박 또는 보완하세요.",
            3: "이전 라운드 전체 내용을 참고해 최종 입장과 구체적인 제안을 정리하세요.",
        }
        return instructions[round_number]

    @staticmethod
    def _round_intro(round_number: int) -> str:
        intros = {
            1: "1라운드를 시작합니다.",
            2: "2라운드에서는 앞선 의견을 반박하거나 보완해주세요.",
            3: "3라운드에서는 최종 입장을 정리해주세요.",
        }
        return intros[round_number]

    @staticmethod
    def _validate_panel_output(output: str, speakers: Sequence[str]) -> None:
        missing = [
            speaker
            for speaker in speakers
            if not re.search(rf"(?m)^\s*{re.escape(speaker)}\s*:", output)
        ]
        if missing:
            raise TalkShowGenerationError(
                f"Gemini 응답에 패널 발언이 누락되었습니다: {', '.join(missing)}"
            )

    @staticmethod
    def _speakers_from_prompt(system_prompt: str) -> list[str]:
        panel_line = next(
            (line for line in system_prompt.splitlines() if line.startswith("패널:")),
            "패널: AI 연구자 알렉스, 사회학자 수진, 창업가 도윤",
        )
        host_line = next(
            (line for line in system_prompt.splitlines() if line.startswith("사회자:")),
            "사회자: 민지",
        )
        return [
            host_line.split(":", 1)[1].strip(),
            *[name.strip() for name in panel_line.split(":", 1)[1].split(",")],
        ]

    @staticmethod
    def _build_contents(messages: Sequence[dict[str, str]]) -> str:
        return "\n\n".join(
            f"{message['role'].upper()}: {message['content']}" for message in messages
        )

    @staticmethod
    def _extract_output_text(interaction: object) -> str:
        outputs = getattr(interaction, "outputs", None) or []
        return "\n".join(
            content.text
            for content in outputs
            if getattr(content, "text", None)
        )

    @staticmethod
    def _find_finish_reason(value: object) -> str | None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if key in {"finish_reason", "finishReason", "stop_reason", "stopReason"}:
                    return str(nested)
                found = TalkShowService._find_finish_reason(nested)
                if found:
                    return found
        elif isinstance(value, (list, tuple)):
            for nested in value:
                found = TalkShowService._find_finish_reason(nested)
                if found:
                    return found
        else:
            for key in ("finish_reason", "finishReason", "stop_reason", "stopReason"):
                nested = getattr(value, key, None)
                if nested is not None:
                    return str(nested)
            model_dump = getattr(value, "model_dump", None)
            if callable(model_dump):
                return TalkShowService._find_finish_reason(model_dump())
        return None

    @staticmethod
    def _is_token_limit(finish_reason: str | None) -> bool:
        normalized = (finish_reason or "").upper()
        return "MAX_TOKENS" in normalized or "MAX_OUTPUT" in normalized

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        code = getattr(error, "code", None)
        status = str(getattr(error, "status", "")).upper()
        message = str(error).upper()
        return code == 429 or "RESOURCE_EXHAUSTED" in status or "RATE LIMIT" in message

    @staticmethod
    def _retry_after_seconds(error: Exception) -> float | None:
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        value = headers.get("retry-after") if headers else None
        if value is None:
            details = getattr(error, "details", {})
            if isinstance(details, dict):
                error_details = details.get("error", details)
                if isinstance(error_details, dict):
                    value = error_details.get("retryAfter") or error_details.get("retry_after")
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            return None
        return seconds if 0 < seconds <= 30 else None

    @staticmethod
    def _demo_debate(messages: Sequence[dict[str, str]]) -> str:
        latest = messages[-1]["content"] if messages else "오늘의 주제를 소개해 주세요."
        rounds = []
        for round_number in range(1, 4):
            rounds.append(
                f"### {round_number}라운드\n"
                f"알렉스: {latest}에 대해 기술의 가능성을 중심으로 보겠습니다. 검증 가능한 실험이 중요합니다.\n"
                "수진: 그 가능성이 사회에 미치는 영향도 함께 살펴야 합니다. 접근성과 공정성을 기준에 넣어야 합니다.\n"
                "도윤: 두 관점을 바탕으로 작은 범위에서 시작해 결과를 측정하는 접근이 현실적입니다."
            )
        rounds.append("### 전체 토론 요약\n**민지:** 가능성을 실험하되 검증과 공정성을 함께 고려해야 한다는 결론입니다.")
        return "\n\n".join(rounds)


@dataclass
class TalkShowGenerationError(Exception):
    message: str
    partial_text: str = ""

    def __str__(self) -> str:
        return self.message


@dataclass
class TalkShowRateLimitError(Exception):
    message: str
    partial_text: str = ""

    def __str__(self) -> str:
        return self.message