import os

import streamlit as st
from dotenv import load_dotenv

from src.config import AppConfig
from src.openrouter_service import OpenRouterError, OpenRouterService, model_label


load_dotenv()

DEFAULT_TOPIC = "AX전환과 교육의 미래(도구를 넘어 시스템이 되다)"
DEFAULT_HOST = {
    "name": "민지",
    "persona": "교육·기술 분야의 전문 진행자.\n토론을 중립적으로 진행하고,\n각 패널의 의견을 연결하며,\n논쟁이 한쪽으로 치우치지 않도록 질문을 던진다.",
    "model": "openrouter/free",
}
DEFAULT_PANELS = [
    {"name": "지훈", "persona": "AX 전환을 적극적으로 추진해야 한다고 보는 교육혁신 및 AI 기술 전문가.\n기술 발전과 생산성 향상을 중요하게 생각한다.", "model": "google/gemini-3.6-flash"},
    {"name": "수진", "persona": "학교 현장의 현실을 중요하게 보는 교사.\nAI 도입의 필요성에는 동의하지만 교사의 업무 부담과 실제 교육 효과를 신중하게 검토한다.", "model": "openai/gpt-5-mini"},
    {"name": "도윤", "persona": "교육 정책과 행정 시스템 관점에서 바라보는 전문가.\n개별 AI 도구보다 조직과 제도 전체의 AX 전환을 중요하게 본다.", "model": "anthropic/claude-sonnet"},
    {"name": "서연", "persona": "AI 윤리와 인간 중심 교육을 강조하는 전문가.\nAI 의존, 개인정보, 평가 공정성, 인간 교사의 역할에 대해 비판적인 시각을 제시한다.", "model": "openrouter/free"},
]


def _secret_or_env(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name)
    except (FileNotFoundError, KeyError, AttributeError):
        return None


def initialize_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "config" not in st.session_state:
        st.session_state.config = AppConfig(
            base_url=_secret_or_env("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1",
            api_key=_secret_or_env("OPENROUTER_API_KEY"),
            temperature=float(_secret_or_env("OPENROUTER_TEMPERATURE") or "0.8"),
        )
    if "show_settings" not in st.session_state:
        st.session_state.show_settings = {
            "topic": DEFAULT_TOPIC,
            "host": DEFAULT_HOST.copy(),
            "panelists": [panel.copy() for panel in DEFAULT_PANELS],
        }
    if "model_options" not in st.session_state:
        st.session_state.model_options = None


def render_persona_fields(label: str, person: dict[str, str], models: list[tuple[str, str]], key_prefix: str) -> dict[str, str]:
    st.markdown(f"**{label}**")
    name = st.text_input("이름", value=person["name"], key=f"{key_prefix}_name")
    model_ids = [item[0] for item in models]
    current_model = person["model"] if person["model"] in model_ids else model_ids[0]
    model = st.selectbox(
        "AI 모델",
        options=model_ids,
        index=model_ids.index(current_model),
        format_func=lambda item_id: model_label(item_id, models),
        key=f"{key_prefix}_model",
    )
    persona = st.text_area("페르소나", value=person["persona"], height=120, key=f"{key_prefix}_persona")
    return {"name": name.strip() or person["name"], "model": model, "persona": persona.strip()}


def render_sidebar(service: OpenRouterService) -> tuple[str, dict[str, str], list[dict[str, str]]]:
    settings = st.session_state.show_settings
    if st.session_state.model_options is None:
        st.session_state.model_options = service.list_models()
    models = st.session_state.model_options
    with st.sidebar:
        st.header("쇼 설정")
        topic = st.text_area("오늘의 주제", value=settings["topic"], height=80, key="show_topic")
        st.divider()
        host = render_persona_fields("진행자", settings["host"], models, "host")
        st.divider()
        panels = []
        for index, panel in enumerate(settings["panelists"], start=1):
            panels.append(render_persona_fields(f"패널 {index}", panel, models, f"panel_{index}"))
            if index < 4:
                st.divider()
        settings["topic"] = topic
        settings["host"] = host
        settings["panelists"] = panels
        st.divider()
        if st.session_state.config.has_api_key:
            st.success("OpenRouter 연결됨")
        else:
            st.info("API 키 없음 · 데모 모드")
        if st.button("대화 기록 초기화", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    return topic, host, panels


def main() -> None:
    initialize_state()
    service = OpenRouterService(st.session_state.config)
    topic, host, panels = render_sidebar(service)
    st.title("🎙️ AI Talk Show")
    st.caption("사회자와 네 명의 패널이 3라운드로 함께 만드는 토크쇼")
    if not st.session_state.messages:
        st.info("왼쪽에서 출연자와 모델을 설정한 뒤 토론을 시작하세요.")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("토론 주제나 시청자 질문을 입력하세요")
    if not prompt and st.button("토크쇼 시작", type="primary"):
        prompt = topic
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("출연자별 AI 모델이 토론을 준비하고 있습니다..."):
                try:
                    answer = service.generate_debate(topic, host, panels, st.session_state.messages)
                except OpenRouterError as error:
                    if error.partial_text:
                        st.markdown(error.partial_text)
                        st.session_state.messages.append({"role": "assistant", "content": error.partial_text, "partial": True})
                    st.error(f"{error.speaker or '토론'}의 모델 {error.model or ''} 호출에 실패했습니다.\n\n{error}")
                    return
            st.markdown(answer)
        st.session_state.messages = [item for item in st.session_state.messages if not item.get("partial")]
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
