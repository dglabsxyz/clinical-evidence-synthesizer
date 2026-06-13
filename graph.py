"""CRAG (Corrective RAG) state machine with LangGraph.

retrieve -> grade -> (relevant?) -> generate
                         |no
                         v
                     web_search -> generate
"""
import os
import json
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from openai import OpenAI
from tavily import TavilyClient
import db
import embeddings

BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)
GRADER_MODEL = "qwen-plus"   # cheap/fast: graded on every retrieval
GEN_MODEL = "qwen-max"       # top quality: final synthesis

_llm = None
def llm():
    global _llm
    if _llm is None:
        _llm = OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=BASE_URL)
    return _llm

def tavily():
    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

class State(TypedDict, total=False):
    question: str
    documents: List[dict]
    relevant: bool
    route: str
    web_results: List[dict]
    answer: str

# ---- nodes ----------------------------------------------------------------
def retrieve_node(state):
    conn = db.get_conn()
    qvec = embeddings.embed_query(state["question"])
    rows = db.search(conn, qvec, k=3)
    docs = [{"question": r[0], "answer": r[1], "source": r[2], "score": float(r[3])}
            for r in rows]
    return {"documents": docs}

def grade_node(state):
    docs = state.get("documents", [])
    context = "\n\n".join(
        f"[Doc {i+1}] Q: {d['question']}\nA: {d['answer']}" for i, d in enumerate(docs)
    )
    prompt = (
        "You are a strict medical evidence reviewer. Decide whether the retrieved "
        "documents contain RELEVANT and SUFFICIENT, current clinical evidence to "
        "answer the question accurately. If the question asks for the latest/most "
        "recent guidelines and the documents are generic or undated, treat them as "
        'INSUFFICIENT. Respond with JSON only: {"relevant": true} or '
        '{"relevant": false}.\n\n'
        f"Question: {state['question']}\n\nDocuments:\n{context or '(none)'}"
    )
    resp = llm().chat.completions.create(
        model=GRADER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    relevant = _parse_relevant(resp.choices[0].message.content or "")
    return {"relevant": relevant, "route": "generate" if relevant else "web_search"}

def _parse_relevant(text):
    t = text.strip().lower()
    try:
        s, e = t.find("{"), t.rfind("}")
        if s != -1 and e != -1:
            return bool(json.loads(t[s:e + 1]).get("relevant"))
    except Exception:
        pass
    return ("true" in t) and ("false" not in t)

def web_search_node(state):
    res = tavily().search(
        query=state["question"],
        max_results=4,
        search_depth="advanced",
        include_domains=["cdc.gov", "who.int", "ncbi.nlm.nih.gov",
                         "pubmed.ncbi.nlm.nih.gov", "ahajournals.org", "nih.gov"],
    )
    web = [{"title": r.get("title", ""), "content": r.get("content", ""),
            "url": r.get("url", "")} for r in res.get("results", [])]
    return {"web_results": web}

def generate_node(state):
    if state.get("relevant"):
        ctx = "\n\n".join(
            f"[{i+1}] ({d['source']}) Q: {d['question']}\nA: {d['answer']}"
            for i, d in enumerate(state.get("documents", []))
        )
        src_label = "the local clinical knowledge base (MedQuAD)"
    else:
        ctx = "\n\n".join(
            f"[{i+1}] {w['url']}\n{w['content']}"
            for i, w in enumerate(state.get("web_results", []))
        )
        src_label = "live web sources (CDC / WHO / PubMed)"
    prompt = (
        "You are a clinical evidence synthesizer. Answer using ONLY the provided "
        "context. Be accurate and concise. Use inline citations like [1], [2] and "
        "list the sources at the end. If the context is insufficient, say so "
        "explicitly rather than guessing.\n\n"
        f"Question: {state['question']}\n\nContext from {src_label}:\n{ctx or '(none)'}"
    )
    resp = llm().chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return {"answer": (resp.choices[0].message.content or "").strip()}

def _route(state):
    return state["route"]

def build_graph():
    g = StateGraph(State)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("web_search", web_search_node)
    g.add_node("generate", generate_node)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", _route,
                            {"generate": "generate", "web_search": "web_search"})
    g.add_edge("web_search", "generate")
    g.add_edge("generate", END)
    return g.compile()
