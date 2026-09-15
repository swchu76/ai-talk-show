import unittest
from types import SimpleNamespace

from src.config import AppConfig
from src.openrouter_service import OpenRouterService


class FakeResponse:
    status_code = 200
    headers = {}

    def json(self):
        return {"choices": [{"message": {"content": "응답입니다."}, "finish_reason": "stop"}]}

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self):
        self.headers = {}
        self.calls = []

    def post(self, url, json, timeout):
        self.calls.append((url, json))
        return FakeResponse()

    def get(self, url, timeout):
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"data": [{"id": "test/model", "name": "Test Model"}]},
        )


class OpenRouterServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = OpenRouterService(AppConfig(api_key="test"))
        self.session = FakeSession()
        self.service.session = self.session

    def test_lists_models_and_uses_selected_model_per_speaker(self):
        models = self.service.list_models()
        self.assertEqual(models, [("test/model", "Test Model")])
        host = {"name": "민지", "persona": "중립적 진행자", "model": "host/model"}
        panels = [
            {"name": "지훈", "persona": "기술 전문가", "model": "panel/one"},
            {"name": "수진", "persona": "교사", "model": "panel/two"},
            {"name": "도윤", "persona": "정책 전문가", "model": "panel/three"},
            {"name": "서연", "persona": "윤리 전문가", "model": "panel/four"},
        ]
        result = self.service.generate_debate("주제", host, panels, [{"role": "user", "content": "시작"}])
        self.assertIn("전체 토론 요약", result)
        self.assertEqual(len(self.session.calls), 16)
        selected_models = [call[1]["model"] for call in self.session.calls]
        self.assertEqual(selected_models[0], "host/model")
        self.assertEqual(selected_models[1:5], ["panel/one", "panel/two", "panel/three", "panel/four"])

    def test_each_panel_context_contains_previous_panel_output(self):
        host = {"name": "민지", "persona": "진행자", "model": "host/model"}
        panels = [{"name": name, "persona": "전문가", "model": f"model/{name}"} for name in ("지훈", "수진", "도윤", "서연")]
        self.service.generate_debate("주제", host, panels, [{"role": "user", "content": "시작"}])
        second_panel_input = self.session.calls[2][1]["messages"][1]["content"]
        self.assertIn("응답입니다.", second_panel_input)


if __name__ == "__main__":
    unittest.main()