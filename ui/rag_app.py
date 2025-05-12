import os
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

import streamlit as st
st.set_page_config(page_title="RAG Cyberbot")

import sys
sys.path.append(os.path.dirname(__file__))

from rag_ui import get_rag_response

st.title("RAG Cyberbot Interface")

query = st.text_input("Enter your question:")

if query:
    with st.spinner("Generating response..."):
        try:
            response = get_rag_response(query)
            st.markdown("### Response")
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")