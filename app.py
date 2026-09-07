"""Streamlit entry point for the extracurricular-program recommender."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from recommender import (
    ProgramRetriever,
    build_fallback_answer,
    generate_gemini_answer,
    load_programs,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "programs.json"
load_dotenv(BASE_DIR / ".env")

st.set_page_config(
    page_title="전주대학교 비교과 프로그램 추천",
    page_icon="🎓",
    layout="centered",
)


@st.cache_resource
def get_retriever(data_path: str) -> ProgramRetriever:
    return ProgramRetriever(load_programs(data_path))


retriever = get_retriever(str(DATA_PATH))
api_key = os.getenv("GEMINI_API_KEY", "").strip()
model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").strip()

st.title("🎓 비교과 프로그램 추천 챗봇")
st.write(
    "관심 분야, 학년 또는 원하는 혜택을 입력하면 등록된 프로그램 중 "
    "관련성이 높은 항목을 찾아드립니다."
)
st.caption(
    "포트폴리오용 데모입니다. 데이터는 2025년 수집본이므로 실제 신청 전 "
    "학교 공식 공지를 확인하세요."
)

with st.sidebar:
    st.subheader("답변 방식")
    if api_key:
        answer_mode = st.radio(
            "모드 선택",
            ("Gemini AI 모드", "무료 로컬 검색 모드"),
            help="Gemini 모드는 API 할당량을 사용하고, 로컬 모드는 외부 API를 호출하지 않습니다.",
        )
        if answer_mode == "Gemini AI 모드":
            st.success(f"Gemini 기반 RAG · {model}")
        else:
            st.info("TF-IDF 검색 · API 사용 없음")
    else:
        answer_mode = "무료 로컬 검색 모드"
        st.info("무료 로컬 검색 모드 · API 키 없이 실행 중")
        st.caption("운영자가 Gemini API 키를 설정하면 AI 답변 모드가 활성화됩니다.")
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("예: 3학년 취업 준비에 도움이 되는 프로그램을 알려줘")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    results = retriever.search(query, top_k=3)
    if answer_mode == "Gemini AI 모드" and api_key and results:
        try:
            answer = generate_gemini_answer(
                query,
                results,
                api_key=api_key,
                model=model,
            )
        except Exception as error:
            answer = (
                "Gemini 답변 생성 중 오류가 발생해 로컬 검색 결과로 안내합니다.\n\n"
                + build_fallback_answer(results)
            )
            st.toast(f"Gemini API 오류: {error}", icon="⚠️")
    else:
        answer = build_fallback_answer(results)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    if results:
        with st.expander("검색 근거 보기"):
            for index, result in enumerate(results, start=1):
                st.write(
                    f"{index}. {result['제목']} "
                    f"(유사도 {result['검색점수']:.3f})"
                )
