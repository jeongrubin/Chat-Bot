"""Streamlit entry point for the extracurricular-program recommender."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from recommender import (
    ProgramRetriever,
    build_fallback_answer,
    generate_openai_answer,
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
api_key = os.getenv("OPENAI_API_KEY", "").strip()
model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

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
    st.subheader("실행 상태")
    if api_key:
        st.success(f"AI 답변 모드 · {model}")
    else:
        st.info("로컬 검색 모드 · API 키 없이 실행 중")
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
    if api_key and results:
        try:
            answer = generate_openai_answer(
                query,
                results,
                api_key=api_key,
                model=model,
            )
        except Exception as error:
            answer = (
                "AI 답변 생성 중 오류가 발생해 로컬 검색 결과로 안내합니다.\n\n"
                + build_fallback_answer(results)
            )
            st.toast(f"OpenAI API 오류: {error}", icon="⚠️")
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
