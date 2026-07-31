# -*- coding: utf-8 -*-
"""
make_chain 시뮬레이터 - Streamlit 웹앱
=====================================

기능
1. API Key를 웹페이지에서 직접 입력 (session_state에만 저장, 서버/파일에 저장 안 함)
2. 에이전트(Agent) / 프롬프트(Prompt) 단계를 웹페이지에서 직접 추가·수정·삭제·순서 편집
3. 파일을 업로드하면 텍스트를 청크로 나눠 임베딩 -> 질문과 유사한 청크를 검색해 RAG로 활용
4. 최종 상호작용은 챗봇(chat) 형태
5. 웹 검색(duckduckgo-search)을 옵션으로 켜서 답변에 실시간 웹 정보를 반영

실행 전 설치가 필요한 패키지:
    pip install streamlit openai numpy pypdf python-docx duckduckgo-search

실행:
    streamlit run chain_simulator_app.py
"""

import io
import uuid

import numpy as np
import streamlit as st
from openai import OpenAI

# ---- 선택적 의존성 (없어도 앱은 뜨지만 해당 기능만 비활성화) ----
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None


# =========================================================
# 세션 상태 초기화
# =========================================================
def init_session_state():
    defaults = {
        "api_key": "",
        "model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "pipeline": [
            {
                "id": str(uuid.uuid4()),
                "type": "prompt",
                "name": "지침 추가",
                "content": "다음 질문에 대해 정확하고 간결하게 한국어로 답변하세요.",
            },
            {
                "id": str(uuid.uuid4()),
                "type": "agent",
                "name": "기본 에이전트",
                "content": "You are a helpful, precise assistant.",
            },
        ],
        "rag_chunks": [],  # [{"text":..., "embedding": np.array, "source":...}]
        "chat_history": [],  # [{"role":"user"/"assistant", "content":...}]
        "use_rag": True,
        "use_web_search": False,
        "top_k": 3,
        "show_steps": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_client():
    if not st.session_state.api_key:
        return None
    return OpenAI(api_key=st.session_state.api_key)


# =========================================================
# 파일 -> 텍스트 추출
# =========================================================
def extract_text_from_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()

    if name.endswith(".pdf"):
        if PdfReader is None:
            st.error("pypdf가 설치되어 있지 않습니다: pip install pypdf")
            return ""
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if name.endswith(".docx"):
        if DocxDocument is None:
            st.error("python-docx가 설치되어 있지 않습니다: pip install python-docx")
            return ""
        doc = DocxDocument(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)

    # txt, md, csv 등 텍스트 계열은 그냥 디코딩
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150):
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# =========================================================
# 임베딩 / RAG
# =========================================================
def get_embedding(client: OpenAI, text: str) -> np.ndarray:
    resp = client.embeddings.create(
        model=st.session_state.embedding_model,
        input=text[:8000],
    )
    return np.array(resp.data[0].embedding, dtype=np.float32)


def build_rag_index(client: OpenAI, uploaded_files):
    new_chunks = []
    for f in uploaded_files:
        raw_text = extract_text_from_file(f)
        pieces = chunk_text(raw_text)
        for piece in pieces:
            new_chunks.append({"text": piece, "source": f.name})

    progress = st.progress(0.0, text="임베딩 생성 중...")
    total = max(len(new_chunks), 1)
    for i, chunk in enumerate(new_chunks):
        chunk["embedding"] = get_embedding(client, chunk["text"])
        progress.progress((i + 1) / total, text=f"임베딩 생성 중... ({i + 1}/{total})")
    progress.empty()

    st.session_state.rag_chunks.extend(new_chunks)


def retrieve_context(client: OpenAI, query: str, top_k: int = 3) -> str:
    if not st.session_state.rag_chunks:
        return ""
    q_emb = get_embedding(client, query)
    scored = []
    for chunk in st.session_state.rag_chunks:
        emb = chunk["embedding"]
        sim = float(
            np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb) + 1e-8)
        )
        scored.append((sim, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    parts = []
    for sim, chunk in top:
        parts.append(f"[출처: {chunk['source']} | 유사도 {sim:.2f}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


# =========================================================
# 웹 검색
# =========================================================
def web_search(query: str, max_results: int = 5) -> str:
    if DDGS is None:
        return "(웹 검색 사용 불가: pip install duckduckgo-search 를 설치하세요)"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"(웹 검색 중 오류 발생: {e})"

    if not results:
        return "(검색 결과 없음)"

    parts = []
    for r in results:
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        parts.append(f"- {title}\n  {body}\n  ({href})")
    return "\n".join(parts)


# =========================================================
# LLM 호출 (agent 실행) - 원본 response() 함수를 스트리밍 지원 형태로 재구성
# =========================================================
def call_agent(client: OpenAI, model: str, system: str, user: str, placeholder=None) -> str:
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        stream=True,
    )
    full_response = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        full_response += delta
        if placeholder is not None:
            placeholder.markdown(full_response + "▌")
    if placeholder is not None:
        placeholder.markdown(full_response)
    return full_response


# =========================================================
# make_chain 로직 재구현: 파이프라인(steps)을 순서대로 실행
# =========================================================
def run_pipeline(client: OpenAI, model: str, pipeline, user_input: str, final_placeholder=None):
    """
    pipeline: [{"type": "agent"/"prompt", "content": str, "name": str}, ...]
    원본 make_chain과 동일한 규칙:
      - agent 단계: 이전 결과를 입력으로 LLM 호출 -> 결과로 치환
      - prompt 단계: 문자열을 앞에 이어붙임 (prepend)
    """
    current = user_input
    steps_log = []
    last_agent_index = max(
        (i for i, s in enumerate(pipeline) if s["type"] == "agent"), default=-1
    )

    for i, step in enumerate(pipeline):
        if step["type"] == "agent":
            is_last_agent = i == last_agent_index
            placeholder = final_placeholder if is_last_agent else None
            result = call_agent(client, model, step["content"], current, placeholder)
            steps_log.append({"step": step["name"], "type": "agent", "output": result})
            current = result
        elif step["type"] == "prompt":
            current = step["content"] + "\n" + current
            steps_log.append({"step": step["name"], "type": "prompt", "output": current})

    return current, steps_log


# =========================================================
# 사이드바 UI
# =========================================================
def render_sidebar():
    st.sidebar.header("⚙️ 설정")

    st.session_state.api_key = st.sidebar.text_input(
        "OpenAI API Key", value=st.session_state.api_key, type="password",
        help="이 키는 브라우저 세션에만 보관되며 서버에 저장되지 않습니다."
    )
    st.session_state.model = st.sidebar.text_input("모델명 (chat)", value=st.session_state.model)
    st.session_state.embedding_model = st.sidebar.text_input(
        "임베딩 모델명", value=st.session_state.embedding_model
    )

    st.sidebar.divider()
    st.sidebar.subheader("🔎 RAG (파일 참조)")
    st.session_state.use_rag = st.sidebar.checkbox("RAG 사용", value=st.session_state.use_rag)
    st.session_state.top_k = st.sidebar.slider("검색할 청크 수 (top_k)", 1, 10, st.session_state.top_k)

    uploaded_files = st.sidebar.file_uploader(
        "참조 파일 업로드 (txt, md, pdf, docx, csv)",
        type=["txt", "md", "pdf", "docx", "csv"],
        accept_multiple_files=True,
    )
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("📥 인덱스에 추가", use_container_width=True):
            client = get_client()
            if client is None:
                st.sidebar.error("먼저 API Key를 입력하세요.")
            elif not uploaded_files:
                st.sidebar.warning("업로드된 파일이 없습니다.")
            else:
                build_rag_index(client, uploaded_files)
                st.sidebar.success(f"{len(uploaded_files)}개 파일 인덱싱 완료")
    with col2:
        if st.button("🗑️ 인덱스 초기화", use_container_width=True):
            st.session_state.rag_chunks = []
            st.sidebar.info("RAG 인덱스를 초기화했습니다.")

    st.sidebar.caption(f"현재 저장된 청크 수: {len(st.session_state.rag_chunks)}")

    st.sidebar.divider()
    st.sidebar.subheader("🌐 웹 검색")
    st.session_state.use_web_search = st.sidebar.checkbox(
        "답변 생성 시 웹 검색 결과 반영", value=st.session_state.use_web_search
    )
    if DDGS is None:
        st.sidebar.caption("⚠️ duckduckgo-search 미설치: pip install duckduckgo-search")

    st.sidebar.divider()
    st.session_state.show_steps = st.sidebar.checkbox(
        "중간 단계(체인 실행 로그) 보기", value=st.session_state.show_steps
    )
    if st.sidebar.button("🧹 대화 기록 초기화"):
        st.session_state.chat_history = []


# =========================================================
# 파이프라인(에이전트/프롬프트) 편집 UI
# =========================================================
def render_pipeline_editor():
    st.subheader("🧩 체인(Chain) 구성 - make_chain(*args) 편집")
    st.caption("위에서부터 순서대로 실행됩니다. Agent 단계는 LLM을 호출하고, Prompt 단계는 텍스트를 이전 내용 앞에 붙입니다.")

    col_add1, col_add2 = st.columns(2)
    with col_add1:
        if st.button("➕ Agent 단계 추가", use_container_width=True):
            st.session_state.pipeline.append({
                "id": str(uuid.uuid4()),
                "type": "agent",
                "name": f"에이전트 {len(st.session_state.pipeline) + 1}",
                "content": "You are a helpful assistant.",
            })
    with col_add2:
        if st.button("➕ Prompt 단계 추가", use_container_width=True):
            st.session_state.pipeline.append({
                "id": str(uuid.uuid4()),
                "type": "prompt",
                "name": f"프롬프트 {len(st.session_state.pipeline) + 1}",
                "content": "추가 지침을 여기에 작성하세요.",
            })

    for idx, step in enumerate(st.session_state.pipeline):
        icon = "🤖" if step["type"] == "agent" else "✏️"
        with st.expander(f"{icon} [{idx + 1}] {step['name']} ({step['type']})", expanded=False):
            step["name"] = st.text_input("이름", value=step["name"], key=f"name_{step['id']}")
            label = "시스템 프롬프트 (agent)" if step["type"] == "agent" else "붙일 텍스트 (prompt)"
            step["content"] = st.text_area(label, value=step["content"], key=f"content_{step['id']}", height=120)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if idx > 0 and st.button("⬆️ 위로", key=f"up_{step['id']}"):
                    st.session_state.pipeline[idx - 1], st.session_state.pipeline[idx] = (
                        st.session_state.pipeline[idx], st.session_state.pipeline[idx - 1]
                    )
                    st.rerun()
            with c2:
                if idx < len(st.session_state.pipeline) - 1 and st.button("⬇️ 아래로", key=f"down_{step['id']}"):
                    st.session_state.pipeline[idx + 1], st.session_state.pipeline[idx] = (
                        st.session_state.pipeline[idx], st.session_state.pipeline[idx + 1]
                    )
                    st.rerun()
            with c3:
                pass
            with c4:
                if st.button("🗑️ 삭제", key=f"del_{step['id']}"):
                    st.session_state.pipeline.pop(idx)
                    st.rerun()


# =========================================================
# 챗봇 UI
# =========================================================
def render_chat():
    st.subheader("💬 챗봇")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("steps") and st.session_state.show_steps:
                with st.expander("체인 실행 로그 보기"):
                    for s in msg["steps"]:
                        st.markdown(f"**[{s['type']}] {s['step']}**")
                        st.code(s["output"][:2000])

    user_msg = st.chat_input("메시지를 입력하세요...")
    if not user_msg:
        return

    client = get_client()
    if client is None:
        st.error("먼저 사이드바에서 API Key를 입력하세요.")
        return
    if not st.session_state.pipeline:
        st.error("최소 하나 이상의 체인 단계(Agent 또는 Prompt)를 추가하세요.")
        return

    st.session_state.chat_history.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    # ---- RAG 컨텍스트 수집 ----
    extra_context = ""
    with st.spinner("컨텍스트 준비 중..."):
        if st.session_state.use_rag and st.session_state.rag_chunks:
            rag_ctx = retrieve_context(client, user_msg, st.session_state.top_k)
            if rag_ctx:
                extra_context += f"[참고 문서 발췌]\n{rag_ctx}\n\n"
        if st.session_state.use_web_search:
            web_ctx = web_search(user_msg)
            extra_context += f"[웹 검색 결과]\n{web_ctx}\n\n"

    final_user_input = user_msg
    if extra_context:
        final_user_input = f"{extra_context}[사용자 질문]\n{user_msg}"

    # ---- 체인 실행 ----
    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("체인 실행 중..."):
            final_output, steps_log = run_pipeline(
                client, st.session_state.model, st.session_state.pipeline,
                final_user_input, final_placeholder=placeholder
            )
        placeholder.markdown(final_output)
        if st.session_state.show_steps:
            with st.expander("체인 실행 로그 보기"):
                for s in steps_log:
                    st.markdown(f"**[{s['type']}] {s['step']}**")
                    st.code(s["output"][:2000])

    st.session_state.chat_history.append(
        {"role": "assistant", "content": final_output, "steps": steps_log}
    )


# =========================================================
# 메인
# =========================================================
def main():
    st.set_page_config(page_title="make_chain 시뮬레이터", page_icon="🔗", layout="wide")
    init_session_state()

    st.title("🔗 make_chain 시뮬레이터")
    st.caption("Agent/Prompt 체인 + RAG + 웹 검색을 웹페이지에서 직접 구성하고 테스트하는 챗봇")

    render_sidebar()

    tab1, tab2 = st.tabs(["🧩 체인 편집", "💬 챗봇"])
    with tab1:
        render_pipeline_editor()
    with tab2:
        render_chat()


if __name__ == "__main__":
    main()
