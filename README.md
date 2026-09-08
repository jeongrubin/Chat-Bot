# 전주대학교 비교과 프로그램 추천 챗봇

[웹에서 실행하기](https://jeongrubin-jju-chatbot.streamlit.app/)

학교 비교과 프로그램을 검색하고 관련 프로그램을 보여주는 Streamlit 앱입니다. 처음에는 키워드 검색으로 만들었고, 이후 한국어 문자 단위 TF-IDF와 선택형 Gemini 답변 기능을 추가했습니다.

## 구현한 기능

- 프로그램명, 설명, 신청 대상, 기간, 장소와 혜택 검색
- API 키 없이 사용할 수 있는 TF-IDF 로컬 검색
- 검색된 프로그램만 참고해 답변하는 Gemini 모드
- 새 채팅 생성과 브라우저 세션 내 대화 목록 저장
- 검색에 사용된 프로그램과 유사도 점수 표시

질문을 받으면 먼저 TF-IDF로 관련 프로그램 세 개를 찾습니다. 로컬 모드에서는 검색 결과를 그대로 정리하고, Gemini 모드에서는 같은 검색 결과를 문맥으로 전달해 답변을 만듭니다. API 호출이 실패하면 로컬 결과를 보여줍니다.

## 실행 방법

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Gemini 모드를 사용하려면 환경 변수나 Streamlit Secrets에 다음 값을 설정합니다.

```toml
GEMINI_API_KEY = "발급받은_API_키"
GEMINI_MODEL = "gemini-2.5-flash-lite"
```

API 키는 저장소에 올리지 않습니다. 키가 없어도 로컬 검색은 사용할 수 있습니다.

테스트는 아래 명령으로 실행합니다.

```bash
python -m unittest discover -s tests -v
```

## 사용한 기술

Python, Streamlit, scikit-learn, Gemini API

현재 데이터는 프로젝트를 만들 때 정리한 2025년 프로그램 정보입니다. 실제 모집 여부와 신청 기간은 학교 공지에서 다시 확인해야 합니다.

현재는 준비된 JSON 데이터를 검색합니다. 다음에는 공지 데이터를 갱신하는 방법과 학년·모집 기간 필터를 추가해 보고 싶습니다.
