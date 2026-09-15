# AI Talk Show

Streamlit과 Google Gemini API로 만드는 한국어 AI 토크쇼 프로토타입입니다.

## 실행

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.venv\Scripts\streamlit.exe run app.py
```

`.env`에 `GEMINI_API_KEY`를 넣으면 Gemini API를 사용합니다. 키가 없으면 별도 설정 없이 데모 모드로 화면을 확인할 수 있습니다.

사회자 1명과 패널 3명이 총 3라운드 토론을 진행하며, 각 패널은 앞선 발언을 참고합니다. 마지막에는 사회자가 전체 토론을 요약합니다.

## 프로젝트 구조

```text
ai-talk-show/
├── app.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── gemini_service.py
│   └── prompts.py
├── .env.example
├── .gitignore
├── .streamlit/config.toml
├── requirements.txt
└── README.md
```
