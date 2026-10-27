import os
import streamlit as st
try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass
from dpdp_rag import config, query

st.set_page_config(
    page_title="DPDP Act AI Assistant",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ DPDP Act AI Assistant")
st.caption(
    "A retrieval-augmented assistant for the Digital Personal Data "
    "Protection Act, 2023 and DPDP Rules, 2025."
)

with st.sidebar:
    st.header("Settings")
    model = st.text_input("Groq chat model", value=config.CHAT_MODEL)
    st.info("Answers use Groq with local sentence-transformer embeddings.")
    top_k = st.slider("Retrieved passages", min_value=1, max_value=20, value=config.TOP_K)
    st.info("Answers are generated using your local DPDP RAG pipeline.")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(f"**{source['source']}** — {source['unit']}")

question = st.chat_input("Ask a question about the DPDP Act or Rules")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
        if m["role"] in ("user", "assistant")
    ][-6:]

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching legal documents and generating an answer..."):
                answer, hits = query.answer(
                    question,
                    k=top_k,
                    model=model,
                    history=history,
                )

            st.markdown(answer)

            sources = []
            seen = set()
            for hit in hits:
                meta = hit.get("meta", {})
                source = {
                    "source": meta.get("source", "Unknown source"),
                    "unit": meta.get("unit", "Unspecified unit"),
                }
                key = (source["source"], source["unit"])
                if key not in seen:
                    seen.add(key)
                    sources.append(source)

            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.write(f"**{source['source']}** — {source['unit']}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        except Exception as exc:
            st.error(f"Could not generate an answer: {exc}")
            st.caption(
                "Check that Ollama is running, the configured models are "
                "available, and the ChromaDB index exists."
            )

