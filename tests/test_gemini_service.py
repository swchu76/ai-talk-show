import unittest
from types import SimpleNamespace
from unittest.mock import patch

from google.genai.errors import ClientError

from src.config import AppConfig
from src.gemini_service import TalkShowRateLimitError, TalkShowService


def panel_reply(round_number: int) -> str:
    return (
        f"알렉스: 라운드 {round_number}의 첫 발언입니다. 앞선 의견을 참고합니다. 근거를 보완합니다.\n"
        f"수진: 라운드 {round_number}의 두 번째 발언입니다. 동의하거나 반박합니다. 새로운 관점을 제시합니다.\n"
        f"도윤: 라운드 {round_number}의 세 번째 발언입니다. 실행 방안을 보탭니다. 결론을 구체화합니다."
    )


class FakeInteractions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        call_number = len(self.calls)
        output = panel_reply(call_number) if call_number <= 3 else "전체 토론을 요약합니다. 합의와 차이를 정리합니다."
        return SimpleNamespace(output_text=output)


class TalkShowServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = TalkShowService(AppConfig(api_key="test", model="gemini-3.6-flash"))

    def test_generates_three_rounds_and_summary_in_four_calls(self) -> None:
        interactions = FakeInteractions()
        self.service.client = SimpleNamespace(interactions=interactions)

        result = self.service.generate_debate(
            "사회자: 민지\n패널: 알렉스, 수진, 도윤",
            [{"role": "user", "content": "주제"}],
        )

        self.assertEqual(len(interactions.calls), 4)
        self.assertEqual(result.count("### 1라운드"), 1)
        self.assertEqual(result.count("### 2라운드"), 1)
        self.assertEqual(result.count("### 3라운드"), 1)
        for speaker in ("알렉스", "수진", "도윤"):
            self.assertEqual(result.count(f"{speaker}:"), 3)
        self.assertIn("라운드 1의 첫 발언", str(interactions.calls[1]["input"]))
        self.assertIn("라운드 2의 첫 발언", str(interactions.calls[2]["input"]))
        self.assertEqual(interactions.calls[0]["generation_config"]["max_output_tokens"], 3600)
        self.assertEqual(interactions.calls[3]["generation_config"]["max_output_tokens"], 1800)

    def test_rate_limit_preserves_completed_round_and_stops(self) -> None:
        interactions = FakeInteractions()

        def create(**kwargs: object) -> object:
            interactions.calls.append(kwargs)
            if len(interactions.calls) == 2:
                raise ClientError(429, {"error": {"status": "RESOURCE_EXHAUSTED"}})
            return SimpleNamespace(output_text=panel_reply(1))

        interactions.create = create  # type: ignore[method-assign]
        self.service.client = SimpleNamespace(interactions=interactions)

        with self.assertRaises(TalkShowRateLimitError) as raised:
            self.service.generate_debate(
                "사회자: 민지\n패널: 알렉스, 수진, 도윤",
                [{"role": "user", "content": "주제"}],
            )

        self.assertEqual(len(interactions.calls), 2)
        self.assertIn("### 1라운드", raised.exception.partial_text)

    def test_retry_after_is_used_once(self) -> None:
        interactions = FakeInteractions()
        failures = [ClientError(429, {"error": {"status": "RESOURCE_EXHAUSTED", "retryAfter": 0.01}})]

        def create(**kwargs: object) -> object:
            interactions.calls.append(kwargs)
            if failures:
                raise failures.pop()
            return SimpleNamespace(output_text=panel_reply(1))

        interactions.create = create  # type: ignore[method-assign]
        self.service.client = SimpleNamespace(interactions=interactions)

        with patch("src.gemini_service.time.sleep") as sleep:
            result = self.service.generate_debate(
                "사회자: 민지\n패널: 알렉스, 수진, 도윤",
                [{"role": "user", "content": "주제"}],
            )

        sleep.assert_called_once_with(0.01)
        self.assertEqual(len(interactions.calls), 5)
        self.assertIn("전체 토론 요약", result)


if __name__ == "__main__":
    unittest.main()