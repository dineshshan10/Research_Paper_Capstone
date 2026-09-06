"""Streamlit research workspace with chat, corpus, evaluation, and guide tabs."""
from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import DEFAULT_RETRIEVAL_STRATEGY, RETRIEVAL_STRATEGIES
from core.graph import get_agent
from entrypoints.ui_components import render_conversation_guide, render_corpus, render_evaluation, render_inspector, render_sources

SAMPLE_QUESTIONS = (
    "How does Mistral improve inference efficiency?",
    "What is the goal of RLHF in InstructGPT?",
    "How does Gemini handle text and images?",
    "Compare Mistral attention with the original Transformer.",
    "What is DeepSeek-R1's training recipe?",
    "What is the temperature on Venus?",
)

def start_new_conversation() -> None:
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid4())

def clear_current_conversation() -> None:
    get_agent().clear_thread(st.session_state.thread_id)
    st.session_state.messages = []

st.set_page_config(page_title="Research Paper Answer Bot", page_icon=":material/article:", layout="wide")
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())

st.title("Research Paper Answer Bot")
st.caption("Grounded answers across Attention, GPT-4, InstructGPT, Mistral 7B, and Gemini 1.0.")
with st.sidebar:
    st.subheader("Conversation")
    st.button("New conversation", icon=":material/add_comment:", on_click=start_new_conversation, width="stretch")
    st.button("Clear current conversation", icon=":material/delete:", on_click=clear_current_conversation, width="stretch")
    st.caption(f"Thread: `{st.session_state.thread_id[:8]}`")
    strategy = st.selectbox("Retrieval strategy", RETRIEVAL_STRATEGIES, index=RETRIEVAL_STRATEGIES.index(DEFAULT_RETRIEVAL_STRATEGY))
    with st.expander("Optional live services"):
        tavily_api_key = st.text_input(
            "Tavily API key",
            type="password",
            key="tavily_api_key",
            help="Used only for this browser session when a question is outside the five-paper corpus.",
        )
        st.caption("For a persistent local setup, put `TAVILY_API_KEY=...` in `.env`. Do not commit API keys.")

chat_tab, corpus_tab, evaluation_tab, guide_tab = st.tabs([":material/forum: Research chat", ":material/library_books: Corpus", ":material/analytics: Evaluation", ":material/help: How it works"])
with chat_tab:
    with st.container(border=True):
        st.caption("Ask a question or continue the active research thread")
        with st.form("research-question-form", border=False, clear_on_submit=True):
            prompt = st.text_input(
                "Research question",
                placeholder="Ask anything about any AI research papers…",
                key="research_question",
            )
            submitted = st.form_submit_button("Ask", type="primary")
        st.caption("Sample questions")
        sample_prompt = None
        with st.container(horizontal=True, wrap=True, gap="xsmall"):
            for index, question in enumerate(SAMPLE_QUESTIONS):
                if st.button(question, key=f"sample-question-{index}", width="content"):
                    sample_prompt = question

    chat_history = st.container(height=520, border=True, key="chat-history", autoscroll=True)
    incoming_prompt = sample_prompt or (prompt.strip() if submitted else "")
    if incoming_prompt:
        prompt = incoming_prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_history:
            with st.status(":shimmer[Retrieving cited evidence]", type="compact") as status:
                result = get_agent().ask(prompt, st.session_state.thread_id, strategy, tavily_api_key or None)
                status.update(label="Evidence retrieved", state="complete")
        st.session_state.messages.append({"role": "assistant", "content": result.answer, "result": result})
    with chat_history:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message.get("result"):
                    render_sources(message["result"])
                    render_inspector(message["result"])
with corpus_tab:
    render_corpus()
with evaluation_tab:
    render_evaluation()
with guide_tab:
    render_conversation_guide(st.session_state.thread_id, len(st.session_state.messages))
