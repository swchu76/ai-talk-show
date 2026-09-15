from collections.abc import Sequence
from dataclasses import dataclass

from google import genai

from src.config import AppConfig


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

        transcript = [self._build_contents(messages)]
        debate: list[str] = []
        speakers = self._speakers_from_prompt(system_prompt)

        for round_number in range(1, 4):
            host_prompt = self._turn_prompt(
                system_prompt,
                round_number,
                speakers[0],
                "사회자",
                transcript,
            )
            host_turn = self._generate_turn(host_prompt, transcript)
            debate.append(f"### {round_number}라운드\n**{speakers[0]}:** {host_turn}")
            transcript.append(f"{speakers[0]} (사회자): {host_turn}")

            for panel in speakers[1:]:
                panel_prompt = self._turn_prompt(
                    system_prompt,
                    round_number,
                    panel,
                    "패널",
                    transcript,
                )
                panel_turn = self._generate_turn(panel_prompt, transcript)
                debate.append(f"**{panel}:** {panel_turn}")
                transcript.append(f"{panel} (패널): {panel_turn}")

        summary_prompt = self._turn_prompt(
            system_prompt,
            3,
            speakers[0],
            "사회자 요약",
            transcript,
        )
        summary = self._generate_turn(summary_prompt, transcript)
        debate.append(f"\n### 전체 토론 요약\n**{speakers[0]}:** {summary}")
        return "\n\n".join(debate)

    def _generate_turn(self, prompt: str, transcript: Sequence[str]) -> str:
        interaction = self.client.interactions.create(
            model=self.config.model,
            input="\n\n".join(transcript) + f"\n\n현재 요청:\n{prompt}",
            system_instruction=prompt,
            generation_config={
                "temperature": self.config.temperature,
                "max_output_tokens": 2048,
            },
        )
        finish_reason = self._find_finish_reason(interaction)
        output_text = getattr(interaction, "output_text", None) or self._extract_output_text(interaction)
        if self._is_token_limit(finish_reason):
            raise TalkShowGenerationError(
                f"Gemini 응답이 토큰 제한으로 잘렸습니다 (finish_reason: {finish_reason}). "
                "다시 시도하거나 주제를 짧게 입력해 주세요.",
                partial_text=output_text,
            )
        return output_text.strip() if output_text else "응답을 받지 못했습니다. 다시 시도해 주세요."

    @staticmethod
    def _turn_prompt(
        base_prompt: str,
        round_number: int,
        speaker: str,
        role: str,
        transcript: Sequence[str],
    ) -> str:
        previous = "\n".join(transcript[-8:])
        if role == "사회자 요약":
            instruction = "전체 토론의 핵심 쟁점, 합의점, 의견 차이와 남은 질문을 3~5문장으로 요약하세요."
        elif role == "사회자":
            instruction = "앞선 발언을 짧게 짚고 다음 논점을 제시하는 3~5문장의 진행 발언을 하세요."
        else:
            instruction = (
                "앞선 발언을 반드시 참고해 동의하거나 반박하고, 자신의 전문 관점과 구체적 사례를 포함한 "
                "3~5문장 발언을 하세요."
            )
        return f"""{base_prompt}

현재 라운드: {round_number}
발언자: {speaker}
역할: {role}
지금까지의 토론 내용:
{previous}

이번 발언 지침: {instruction}
발언자 이름이나 마크다운 제목을 붙이지 말고 발언 내용만 출력하세요."""

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
        return [host_line.split(":", 1)[1].strip(), *[name.strip() for name in panel_line.split(":", 1)[1].split(",")]]

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
    def _demo_debate(messages: Sequence[dict[str, str]]) -> str:
        latest = messages[-1]["content"] if messages else "오늘의 주제를 소개해 주세요."
        return f"""**민지:** 오늘의 질문은 '{latest}'입니다. 세 분의 의견을 차례로 들어보겠습니다.

### 1라운드
**알렉스:** 기술의 가능성을 먼저 봐야 합니다. 다만 실제 활용에서는 검증이 중요합니다.
**수진:** 알렉스의 의견에 동의하면서도, 사회적 영향과 접근성도 함께 살펴야 합니다.
**도윤:** 두 분의 관점을 바탕으로 보면 작은 실험부터 시작하는 것이 현실적입니다.

### 2라운드
**민지:** 앞선 발언을 바탕으로 구체적인 사례를 들어보겠습니다.
**알렉스:** 앞서 말한 검증을 위해 사용자가 결과를 확인할 수 있는 장치가 필요합니다.
**수진:** 그 장치가 모든 사람에게 공평하게 제공되는지도 확인해야 합니다.
**도윤:** 두 조건을 만족하는 서비스를 작게 출시하고 반응을 측정할 수 있습니다.

### 3라운드
**민지:** 마지막으로 각자의 제안을 한 문장으로 정리해 주세요.
**알렉스:** 검증 가능한 기술 활용이 출발점입니다.
**수진:** 사람과 사회에 미치는 영향을 함께 평가해야 합니다.
**도윤:** 작게 시작해 빠르게 배우되 책임 있게 확장해야 합니다.

**민지:** 요약하면, 가능성을 실험하되 검증과 공정성을 놓치지 않는 접근이 세 분의 공통된 결론입니다."""


@dataclass
class TalkShowGenerationError(Exception):
    message: str
    partial_text: str = ""

    def __str__(self) -> str:
        return self.message