from __future__ import annotations
import pandas as pd
import streamlit as st
from config.settings import CORPUS, RESULTS_DIR
from core.schema import Answer

def render_sources(result: Answer) -> None:
    if not result.sources: return
    with st.expander("Sources", expanded=False):
        for source in result.sources:
            label = f"[{source.number}] {source.title} — p. {source.page}" if source.page else f"[{source.number}] {source.title}"
            st.markdown(f"**{label}**")
            st.caption(source.excerpt)
            if source.url: st.link_button("Open source", source.url)

def render_inspector(result: Answer) -> None:
    with st.expander("Retrieval inspector", expanded=False):
        st.caption(f"Strategy: {result.retrieval_strategy} · {result.latency_ms:.0f} ms · provenance: {result.provenance}")
        st.dataframe([{"rank": i, "paper": d.title, "page": d.page, "score": round(d.score, 4), "section": d.section} for i, d in enumerate(result.inspector, 1)], hide_index=True)

def render_corpus() -> None:
    st.subheader("Papers in the corpus")
    st.caption("The assistant searches these five primary sources. Page-level metadata is retained for every citation.")
    rows = [{"Paper": paper.title, "Year": paper.year, "Topics": ", ".join(paper.topics), "PDF": paper.filename, "Size (MB)": round(paper.path.stat().st_size / 1_000_000, 1)} for paper in CORPUS]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    selected_title = st.selectbox("Preview a paper", [paper.title for paper in CORPUS])
    paper = next(item for item in CORPUS if item.title == selected_title)
    with st.container(border=True):
        st.markdown(f"**{paper.title}** ({paper.year})")
        st.write("Topics: " + ", ".join(paper.topics))
        with open(paper.path, "rb") as file:
            st.download_button("Download PDF", file, file_name=paper.filename, mime="application/pdf", icon=":material/download:")

def render_evaluation() -> None:
    st.subheader("Evaluation results")
    st.caption("Offline fallback runs are labelled in the report. Re-run the bake-off with cached neural models and API credentials for final model-selection evidence.")
    for path in [RESULTS_DIR / "embedding_comparison.md", RESULTS_DIR / "retrieval_comparison.md"]:
        with st.container(border=True):
            st.markdown(f"#### {path.stem.replace('_', ' ').title()}")
            if path.exists():
                st.markdown(path.read_text(encoding="utf-8"))
                st.download_button("Download report", path.read_bytes(), file_name=path.name, mime="text/markdown", key=f"download-{path.name}", icon=":material/download:")
            else:
                st.info("No report yet. Run `python -m evaluation.run_benchmark` from the project root.")

def render_conversation_guide(thread_id: str, message_count: int) -> None:
    st.subheader("Conversation controls")
    st.table({
        "New conversation": "Creates a fresh thread with no prior context.",
        "Continue conversation": "Ask a follow-up such as “What about its context window?”; it uses this thread's previous turn.",
        "Clear conversation": "Removes visible messages and server-side memory for the current thread.",
        "Current thread": f"`{thread_id[:8]}` · {message_count // 2} completed turn(s)",
    }, border="horizontal", width="stretch")
    st.markdown("**Try a multi-turn example:** Ask “How does Mistral improve inference efficiency?” then ask “What about its context window?”")
