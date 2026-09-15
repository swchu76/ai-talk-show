import streamlit as st
from dotenv import load_dotenv

from src.config import AppConfig
from src.gemini_service import TalkShowGenerationError, TalkShowService
from src.prompts import build_system_prompt


load_dotenv()

st.set_page_config(
    page_title="AI Talk Show",
    page_icon="🎙️",
    layout="wide",
)


def initialize_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "config" not in st.session_state:
        st.session_state.config = AppConfig.from_environment()


def render_sidebar() -> tuple[str, str, list[str]]:
    with st.sidebar:
        st.header("쇼 설정")
        topic = st.text_input("오늘의 주제", value="생성형 AI가 바꾸는 우리의 일상")
        host = st.text_input("진행자", value="민지")
        st.subheader("패널 3명")
        panels = [
            st.text_input("패널 1", value="AI 연구자 알렉스"),
            st.text_input("패널 2", value="사회학자 수진"),
            st.text_input("패널 3", value="창업가 도윤"),
        ]

        st.divider()
        if st.session_state.config.has_api_key:
            st.success(f"Gemini 연결됨 · {st.session_state.config.model}")
        else:
            st.info("API 키 없음 · 데모 모드")

        if st.button("대화 기록 초기화", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return topic, host, panels


def main() -> None:
    initialize_state()
    topic, host, panels = render_sidebar()
    config = st.session_state.config
    service = TalkShowService(config)

    st.title("🎙️ AI Talk Show")
    st.caption("사회자와 세 패널이 3라운드로 함께 만드는 토크쇼")

    if not st.session_state.messages:
        st.info("왼쪽에서 쇼를 설정한 뒤 질문을 입력하거나, 아래 버튼으로 첫 질문을 만들어 보세요.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("토론 주제나 시청자 질문을 입력하세요")
    if not prompt:
        if st.button("첫 질문 자동 생성", type="primary"):
            prompt = f"{topic}에 대해 시청자가 가장 궁금해할 질문을 시작해 주세요."

    if prompt:
        system_prompt = build_system_prompt(topic=topic, host=host, panels=panels)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("대화를 준비하고 있습니다..."):
                try:
                    answer = service.generate_debate(
                        system_prompt=system_prompt,
                        messages=st.session_state.messages,
                    )
                except TalkShowGenerationError as error:
                    if error.partial_text:
                        st.markdown(error.partial_text)
                    st.error(str(error))
                    return
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
