"""Streamlit chat UI for the conversational text-to-SQL agent.

Phase 0: placeholder page that confirms the Streamlit entry point starts.
The real chat experience (message history, SQL cards, result tables,
database picker) is built in Phase 8.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Text-to-SQL Agent", page_icon=":speech_balloon:")
st.title("Text-to-SQL Agent")
st.caption("Phase 0 placeholder. Chat UI arrives in Phase 8.")
st.info(
    "Services scaffolded. Next: wire the text-to-SQL FastAPI endpoints "
    "and the orchestrator agent, then build the chat interface here."
)
