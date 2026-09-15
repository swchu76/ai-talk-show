from src.prompts import build_system_prompt


def test_build_system_prompt_contains_show_details() -> None:
    prompt = build_system_prompt("우주 여행", "지우", ["샘", "나래", "준"])

    assert "우주 여행" in prompt
    assert "지우" in prompt
    assert "샘" in prompt
    assert "나래" in prompt
    assert "3라운드" in prompt
    assert "한국어" in prompt
