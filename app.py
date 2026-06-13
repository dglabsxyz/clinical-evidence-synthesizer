"""Streamlit UI that visualizes the CRAG routing live."""
import streamlit as st
import graph as cg

st.set_page_config(page_title="Clinical Evidence Synthesizer",
                   page_icon="\U0001FA7A", layout="wide")

st.title("\U0001FA7A Clinical Evidence Synthesizer")
st.caption("Corrective RAG (CRAG) · Qwen + pgvector + Tavily — retrieves, "
           "self-grades its evidence, and falls back to live web sources when local "
           "evidence is insufficient.")

if "app" not in st.session_state:
    st.session_state.app = cg.build_graph()

examples = {
    "Baseline (stays local)": "What are the symptoms of Type 2 Diabetes?",
    "Time-sensitive (forces web fallback)":
        "What are the latest 2025 AHA guidelines for GLP-1 agonists in heart "
        "failure patients?",
}
c1, c2 = st.columns(2)
if c1.button(list(examples)[0]):
    st.session_state.q = examples[list(examples)[0]]
if c2.button(list(examples)[1]):
    st.session_state.q = examples[list(examples)[1]]

q = st.text_input("Ask a clinical question:",
                  value=st.session_state.get("q", ""),
                  placeholder="e.g., What are the symptoms of Type 2 Diabetes?")

if st.button("Synthesize", type="primary") and q:
    with st.spinner("retrieve → grade → route → generate ..."):
        result = st.session_state.app.invoke({"question": q})

    if result.get("relevant"):
        st.success("✅ Grader verdict: local evidence **RELEVANT** → "
                   "answered from the **Local DB (MedQuAD)**")
    else:
        st.warning("⚠️ Grader verdict: local evidence **INSUFFICIENT** → "
                   "routed to **Web fallback (CDC / WHO / PubMed via Tavily)**")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Synthesized answer")
        st.markdown(result.get("answer", "_no answer_"))
    with right:
        st.subheader("Retrieved local evidence")
        for d in result.get("documents", []):
            st.markdown(f"**{d['question']}**  \n"
                        f"score `{d['score']:.3f}` · _{d['source']}_  \n"
                        f"{d['answer'][:240]}…")
        if not result.get("relevant"):
            st.subheader("Live web sources used")
            for w in result.get("web_results", []):
                st.markdown(f"• [{w['title']}]({w['url']})")
