import os
import json
import numpy as np
import streamlit as st
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from streamlit_chat import message

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
chat_model = ChatOpenAI(model_name="gpt-4o", temperature=0.1)

# Streamlit 페이지 설정
st.set_page_config(page_title="전주대학교 비교과 챗봇", page_icon="🎓", layout="centered")
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .chat-container { max-width: 600px; margin: auto; }
    </style>
""", unsafe_allow_html=True)

st.title("🎓 전주대학교 비교과 챗봇")
st.write("비교과 프로그램에 대해 질문하세요!")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# JSON 데이터 로드
@st.cache_data
def load_program_data():
    file_path = "programs.json"
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        possible_keys = ["프로그램_정보", "비교과_프로그램", "프로그램"]
        for key in possible_keys:
            if key in data:
                return data[key]
        return []
    except Exception as e:
        st.error(f"❌ JSON 로드 오류: {str(e)}")
        return []

program_data = load_program_data()
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

@st.cache_data
def create_embeddings(data):
    if not data:
        return np.array([])
    texts = [f"{item['제목']} {item['설명']} {item.get('신청대상', '')} {item.get('혜택', '')}" for item in data]
    embeddings = embeddings_model.embed_documents(texts)
    return np.array(embeddings)

program_embeddings = create_embeddings(program_data)

def search_similar_programs(query, top_k=3):
    if program_embeddings.size == 0:
        return []
    query_embedding = np.array(embeddings_model.embed_query(query)).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, program_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.1:
            results.append(program_data[idx])
    return results

def generate_rag_response(query):
    results = search_similar_programs(query)
    if results:
        gpt_prompt = f"""
        사용자 질문: "{query}"
        검색된 비교과 프로그램 목록:
        {json.dumps(results, indent=2, ensure_ascii=False)}
        위 정보를 바탕으로 사용자에게 적절한 답변을 만들어줘.
        """
    else:
        gpt_prompt = f"""
        사용자 질문: "{query}"
        관련된 비교과 프로그램이 데이터에 명확히 없습니다. 하지만 유사한 정보를 제공할 수 있도록 최선을 다할게요.
        """
    response = chat_model.invoke(gpt_prompt)
    return response.content

# 채팅 UI 개선
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for i, msg in enumerate(st.session_state["messages"]):
    if isinstance(msg, HumanMessage):
        message(msg.content, is_user=True, key=f"user_{i}")
    elif isinstance(msg, AIMessage):
        message(msg.content, is_user=False, key=f"ai_{i}")

st.markdown("</div>", unsafe_allow_html=True)

# 사용자 입력창
user_input = st.text_input("메시지 입력:", key="user_input")
if st.button("보내기", key="send_btn") and user_input:
    st.session_state["messages"].append(HumanMessage(content=user_input))
    response_content = generate_rag_response(user_input)
    st.session_state["messages"].append(AIMessage(content=response_content))
    st.experimental_rerun()
