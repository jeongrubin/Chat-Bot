# 전주대학교 비교과 프로그램 추천 챗봇

[![웹 챗봇 실행](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://jeongrubin-jju-chatbot.streamlit.app/)

**[웹에서 챗봇 바로 실행하기](https://jeongrubin-jju-chatbot.streamlit.app/)**

학교 비교과 프로그램 데이터를 검색하고, 사용자의 관심 분야·대상 학년·원하는 혜택과 관련성이 높은 프로그램을 안내하는 Streamlit 애플리케이션입니다.

## 주요 기능

- 한국어 문자 단위 TF-IDF를 이용한 프로그램 검색
- 프로그램명, 설명, 신청 대상, 기간, 장소 및 혜택 통합 검색
- 외부 API 없이 동작하는 무료 로컬 검색 모드
- Gemini API 키가 있으면 검색 결과만 근거로 답변하는 선택적 AI 답변 모드
- 사이드바에서 Gemini AI와 무료 로컬 검색 모드 전환
- 본문 모드 배너와 답변별 라벨을 통한 현재 응답 방식 구분
- 처음 사용하는 사람을 위한 예시 질문 버튼 제공
- 새 채팅 생성과 현재 브라우저 세션의 대화 목록 전환
- 검색에 사용된 프로그램과 유사도 점수 확인
- API 오류 발생 시 로컬 검색 결과로 자동 전환

## 동작 구조

```text
사용자 질문
    ↓
로컬 TF-IDF 검색
    ↓
관련 프로그램 상위 3개 추출
    ├─ 무료 로컬 검색 → 구조화된 검색 결과 출력
    └─ Gemini AI → 검색 결과를 근거로 자연어 답변 생성
```

외부 API 호출 전에 관련 데이터를 먼저 검색하므로, 모델이 등록되지 않은 프로그램을 임의로 만들어 안내하는 문제를 줄였습니다.

## 프로젝트 구조

```text
.
├── app.py                 # Streamlit 화면과 대화 흐름
├── recommender.py         # 데이터 로드, 검색, 답변 생성
├── data/
│   └── programs.json      # 비교과 프로그램 데모 데이터
├── tests/
│   └── test_recommender.py
├── .env.example
└── requirements.txt
```

## 실행 방법

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Gemini 기반 자연어 답변을 사용하려면 `.env.example`을 `.env`로 복사한 뒤 본인의 API 키를 설정합니다.

```env
GEMINI_API_KEY=your_new_api_key_here
GEMINI_MODEL=gemini-2.5-flash-lite
```

API 키는 코드나 GitHub 저장소에 올리지 않습니다. 키가 없어도 검색·추천 기능은 정상적으로 실행됩니다.

### Streamlit Cloud 설정

배포 앱의 `Manage app → Settings → Secrets`에 새로 발급한 키를 다음과 같이 등록합니다.

```toml
GEMINI_API_KEY = "your_new_api_key_here"
GEMINI_MODEL = "gemini-2.5-flash-lite"
```

저장 후 앱을 재부팅하면 사이드바에서 `Gemini AI 모드`와 `무료 로컬 검색 모드`를 선택할 수 있습니다. Gemini API의 무료·유료 사용 여부와 한도는 키 자체가 아니라 해당 Google 프로젝트의 결제 및 할당량 설정에 따라 결정됩니다.

## 테스트

```bash
python -m unittest discover -s tests -v
```

테스트에서는 데이터 로드, NCS 프로그램 검색, 검색 결과 기반 응답 및 빈 질문 처리를 확인합니다.

## 기술 스택

- Python
- Streamlit
- scikit-learn
- Gemini API(선택 사항)

## 데이터 안내

현재 데이터는 프로젝트 구현을 위해 정리한 2025년 비교과 프로그램의 정적 스냅샷입니다. 실시간 학교 시스템과 연동된 서비스가 아니므로 실제 신청 기간과 운영 여부는 공식 공지를 통해 다시 확인해야 합니다.

## 개선 방향

- 학교 공지 데이터의 정기 갱신 파이프라인 구축
- 학년·모집 상태·신청 기간을 이용한 구조화 필터 추가
- 사용자 평가를 활용한 추천 품질 검증
- 배포 환경에서의 비밀키 관리와 사용량 제한
