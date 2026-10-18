import json, os
import chromadb
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()
API_KEY = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv("LLM_BASE_URL", "")
MODEL = os.getenv("LLM_MODEL", "qwen-plus")

st.set_page_config(page_title="Chat with your metadata", layout="wide")
st.title("Chat with your metadata (PoC)")
st.caption("Demo on synthetic Axon + EDC metadata")

if not API_KEY or API_KEY.startswith("PASTE") or not BASE_URL:
    st.error("API key or base URL is missing. Open the .env file, fill LLM_API_KEY / LLM_BASE_URL / LLM_MODEL, save, then restart run.bat.")
    st.stop()

llm = OpenAI(api_key=API_KEY, base_url=BASE_URL)

SYSTEM = """You are a metadata assistant for a bank's data catalog (Informatica Axon + EDC).
Answer ONLY from the CONTEXT. If the answer is not in the context, say clearly that it is not available in the catalog.
Put the direct answer first, then short supporting details. Cite the object IDs or names you used.
Reply in the user's language (Arabic or English) and keep technical names in English.
Finish with 3 short suggested follow-up questions."""

@st.cache_resource(show_spinner="Loading embedding model and building the index (first run only, a few minutes)...")
def load():
    emb = SentenceTransformer("intfloat/multilingual-e5-small")
    col = chromadb.PersistentClient("chroma_db").get_or_create_collection("meta", metadata={"hnsw:space": "cosine"})
    if col.count() == 0:
        docs = [json.loads(l) for l in open("vector_docs.jsonl", encoding="utf-8")]
        for i in range(0, len(docs), 100):
            b = docs[i:i + 100]
            col.add(ids=[d["id"] for d in b], documents=[d["text"] for d in b],
                    metadatas=[{"type": d["type"]} for d in b],
                    embeddings=emb.encode(["passage: " + d["text"] for d in b]).tolist())
    return emb, col

emb, col = load()

SAMPLES = [
    "Who owns the Customer 360 Report and which KPIs does it use?",
    "Where does ECL_AMOUNT come from? Show the lineage step by step.",
    "If I change DOB in T24 CUSTOMER, what is impacted?",
    "Which columns are Restricted and tagged PII?",
    "Which systems are not onboarded to EDC yet?",
    "ايه الـ KPIs اللي بتستخدم Days Past Due؟",
    "What is the quality score of the Trade Finance tables?",
]
st.sidebar.header("Sample questions")
for s in SAMPLES:
    if st.sidebar.button(s):
        st.session_state["pending"] = s
if st.sidebar.button("Clear chat"):
    st.session_state.hist = []

if "hist" not in st.session_state:
    st.session_state.hist = []
for m in st.session_state.hist:
    st.chat_message(m["role"]).write(m["content"])

q = st.chat_input("Ask about glossary, KPIs, reports, lineage, quality, classification...") or st.session_state.pop("pending", None)
if q:
    st.chat_message("user").write(q)
    r = col.query(query_embeddings=emb.encode(["query: " + q]).tolist(), n_results=8)
    ctx = "\n\n".join(f"[{i}] {d}" for i, d in zip(r["ids"][0], r["documents"][0]))
    msgs = [{"role": "system", "content": SYSTEM}] + st.session_state.hist[-6:] + \
           [{"role": "user", "content": f"CONTEXT:\n{ctx}\n\nQUESTION: {q}"}]
    try:
        ans = llm.chat.completions.create(model=MODEL, messages=msgs, temperature=0.1).choices[0].message.content
    except Exception as e:
        st.error(f"The model API returned an error: {e}")
        st.stop()
    st.chat_message("assistant").write(ans)
    with st.expander("Sources used (retrieved from the vector database)"):
        for i, d in zip(r["ids"][0], r["documents"][0]):
            st.markdown(f"**{i}**: {d[:220]}...")
    st.session_state.hist += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
