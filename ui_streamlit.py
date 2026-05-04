from __future__ import annotations

import os

import requests
import streamlit as st


API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="IPCC AR6 RAG", page_icon="IPCC", layout="wide")
st.title("IPCC AR6 RAG")

question = st.text_input(
    "Question",
    placeholder="What are the main observed impacts of climate change?",
)
k = st.slider("Retrieved sources", min_value=1, max_value=10, value=4)

if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Retrieving IPCC context and asking Ollama..."):
        try:
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question, "k": k},
                timeout=180,
            )
        except requests.RequestException as exc:
            st.error(f"Could not reach backend at {API_URL}: {exc}")
        else:
            if not response.ok:
                st.error(f"API error {response.status_code}: {response.text}")
            else:
                data = response.json()
                st.subheader("Answer")
                st.write(data["answer"])

                st.subheader("Sources")
                for source in data.get("sources", []):
                    label = f"{source.get('source')} - page {source.get('page')} - chunk {source.get('chunk')}"
                    with st.expander(label):
                        st.write(source.get("preview", ""))
