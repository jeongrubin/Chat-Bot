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
    page_title="비교과 프로그램 AI 검색·추천",
    page_icon="🎓",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 900px;
        padding-top: 2.5rem;
        padding-bottom: 5rem;
    }
    [data-testid="stChatMessage"] {
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 16px;
        padding: 0.35rem 0.65rem;
        margin-bottom: 0.8rem;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.16);
    }
    div[data-testid="stRadio"] > div {
        gap: 0.75rem;
    }
    .hero-copy {
        color: #64748b;
        font-size: 1.02rem;
        line-height: 1.7;
        margin-bottom: 1.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_retriever(data_path: str) -> ProgramRetriever:
    return ProgramRetriever(load_programs(data_path))


def make_chat_title(query: str, max_length: int = 24) -> str:
    """Create a compact conversation title from the first user question."""
    normalized = " ".join(query.split())
    return normalized[:max_length] + ("…" if len(normalized) > max_length else "")


retriever = get_retriever(str(DATA_PATH))
api_key = os.getenv("GEMINI_API_KEY", "").strip()
model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").strip()

st.title("🎓 비교과 프로그램 AI 검색·추천")
st.markdown(
    '<p class="hero-copy">관심 분야와 학년, 준비 중인 진로를 입력하면 '
    "등록된 비교과 프로그램을 검색하고 참여 목적에 맞게 안내합니다.</p>",
    unsafe_allow_html=True,
)

st.subheader("답변 모드", divider="gray")
mode_options = ["🔎 무료 로컬 검색"]
if api_key:
    mode_options.insert(0, "✨ Gemini AI 추천")

answer_mode = st.radio(
    "답변 모드 선택",
    mode_options,
    horizontal=True,
    label_visibility="collapsed",
)
is_gemini_mode = answer_mode.startswith("✨")

if "chat_counter" not in st.session_state:
    st.session_state.chat_counter = 1
if "conversations" not in st.session_state:
    previous_messages = st.session_state.pop("messages", [])
    st.session_state.conversations = {
        "chat-1": {"title": "새 대화", "messages": previous_messages}
    }
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = "chat-1"

for conversation in st.session_state.conversations.values():
    if conversation["title"] != "새 대화":
        continue
    first_question = next(
        (
            message["content"]
            for message in conversation["messages"]
            if message.get("role") == "user" and message.get("content")
        ),
        None,
    )
    if first_question:
        conversation["title"] = make_chat_title(first_question)

active_chat = st.session_state.conversations[st.session_state.active_chat_id]
messages = active_chat["messages"]

if is_gemini_mode:
    st.success(
        f"현재 **Gemini AI 추천 모드**입니다. 검색된 프로그램을 근거로 "
        f"`{model}`이 추천 이유를 작성합니다."
    )
else:
    st.info(
        "현재 **무료 로컬 검색 모드**입니다. 외부 AI를 호출하지 않고 "
        "TF-IDF 유사도로 관련 프로그램을 찾습니다."
    )

with st.sidebar:
    st.header("대화")
    if st.button("＋ 새 채팅", type="primary", use_container_width=True):
        st.session_state.chat_counter += 1
        new_chat_id = f"chat-{st.session_state.chat_counter}"
        st.session_state.conversations[new_chat_id] = {
            "title": "새 대화",
            "messages": [],
        }
        st.session_state.active_chat_id = new_chat_id
        st.rerun()

    st.caption("대화 목록")
    for chat_id, conversation in reversed(st.session_state.conversations.items()):
        is_active = chat_id == st.session_state.active_chat_id
        if st.button(
            conversation["title"],
            key=f"open-{chat_id}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state.active_chat_id = chat_id
            st.rerun()

    st.caption("대화 목록은 현재 브라우저 접속 동안 유지됩니다.")
    st.divider()
    st.subheader("서비스 안내")
    st.metric("등록 프로그램", f"{len(retriever.programs)}개")
    st.caption("데이터 기준: 2025년 정적 수집본")
    st.divider()
    st.subheader("현재 연결 상태")
    if api_key:
        st.success("Gemini API 연결됨")
    else:
        st.warning("Gemini API 연결 안 됨")
        st.caption("현재는 무료 로컬 검색만 사용할 수 있습니다.")
    st.divider()
    st.caption(
        "이 서비스는 포트폴리오용 데모입니다. 실제 모집 여부와 신청 기간은 "
        "학교 공식 공지를 확인하세요."
    )
    if st.button("현재 대화 비우기", use_container_width=True):
        active_chat["messages"] = []
        active_chat["title"] = "새 대화"
        st.rerun()

for message in messages:
    mode = message.get("mode", "local")
    avatar = "✨" if mode == "gemini" else "🔎"
    with st.chat_message(message["role"], avatar=avatar if message["role"] == "assistant" else None):
        if message["role"] == "assistant":
            label = "Gemini AI 답변" if mode == "gemini" else "로컬 검색 결과"
            st.caption(label)
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("results"):
            with st.expander("추천 근거와 검색 점수 확인"):
                for index, result in enumerate(message["results"], start=1):
                    st.write(
                        f"{index}. {result['제목']} "
                        f"(유사도 {result['검색점수']:.3f})"
                    )

quick_query = None
if not messages:
    st.subheader("이렇게 질문해보세요", divider="gray")
    examples = (
        "3학년 취업 준비 프로그램을 추천해줘",
        "공기업 NCS 준비에 도움이 되는 프로그램이 있어?",
        "면접 역량을 높일 수 있는 프로그램을 알려줘",
    )
    columns = st.columns(3)
    for index, example in enumerate(examples):
        if columns[index].button(example, use_container_width=True):
            quick_query = example

typed_query = st.chat_input("예: 공기업 취업을 준비하는 3학년에게 추천해줘")
query = typed_query or quick_query

if query:
    if active_chat["title"] == "새 대화":
        active_chat["title"] = make_chat_title(query)
    messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    results = retriever.search(query, top_k=3)
    response_mode = "gemini" if is_gemini_mode else "local"

    if is_gemini_mode and api_key and results:
        try:
            answer = generate_gemini_answer(
                query,
                results,
                api_key=api_key,
                model=model,
            )
        except Exception as error:
            response_mode = "local"
            answer = (
                "Gemini 답변 생성 중 오류가 발생해 로컬 검색 결과로 안내합니다.\n\n"
                + build_fallback_answer(results)
            )
            st.toast(f"Gemini API 오류: {error}", icon="⚠️")
    else:
        answer = build_fallback_answer(results)

    messages.append(
        {
            "role": "assistant",
            "content": answer,
            "mode": response_mode,
            "results": results,
        }
    )
    st.rerun()
