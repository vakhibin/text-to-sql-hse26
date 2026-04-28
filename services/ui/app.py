"""Streamlit chat UI for the conversational text-to-SQL agent."""

from __future__ import annotations

import os
import uuid

import streamlit as st

from services.ui.client import (
    DEFAULT_ORCHESTRATOR_URL,
    OrchestratorUIClient,
    OrchestratorUIError,
    format_history_label,
    normalize_base_url,
    sql_history_from_session,
    visible_messages,
)


def _init_state() -> None:
    st.session_state.setdefault("session_id", f"ui-{uuid.uuid4().hex[:8]}")
    st.session_state.setdefault("user_id", "streamlit-user")
    st.session_state.setdefault(
        "orchestrator_url",
        os.getenv("ORCHESTRATOR_API_URL", DEFAULT_ORCHESTRATOR_URL),
    )
    st.session_state.setdefault("active_db_id", "")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_session", None)
    st.session_state.setdefault("last_error", None)


def _client() -> OrchestratorUIClient:
    return OrchestratorUIClient(normalize_base_url(st.session_state.orchestrator_url))


def _load_session() -> None:
    client = _client()
    try:
        session = client.get_session(st.session_state.session_id)
    except OrchestratorUIError as exc:
        st.session_state.last_error = str(exc)
        return
    finally:
        client.close()

    if session is None:
        st.session_state.messages = []
        st.session_state.last_session = None
        return
    st.session_state.last_session = session
    st.session_state.messages = visible_messages(session.get("messages") or [])
    active_db = session.get("active_db_id")
    if active_db:
        st.session_state.active_db_id = active_db


def _send_message(prompt: str) -> None:
    st.session_state.messages.append({"role": "human", "content": prompt})
    client = _client()
    try:
        response = client.chat(
            session_id=st.session_state.session_id,
            user_id=st.session_state.user_id,
            message=prompt,
            active_db_id=st.session_state.active_db_id.strip() or None,
        )
    except OrchestratorUIError as exc:
        st.session_state.last_error = str(exc)
        st.session_state.messages.append(
            {"role": "ai", "content": f"Request failed: {exc}"}
        )
        return
    finally:
        client.close()

    st.session_state.last_error = None
    st.session_state.active_db_id = response.get("active_db_id") or st.session_state.active_db_id
    st.session_state.messages.extend(visible_messages(response.get("messages_delta") or [])[1:])
    _load_session()


def _reset_session() -> None:
    client = _client()
    try:
        client.reset_session(st.session_state.session_id)
    except OrchestratorUIError as exc:
        st.session_state.last_error = str(exc)
    finally:
        client.close()
    st.session_state.messages = []
    st.session_state.last_session = None


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Connection")
        st.text_input("Orchestrator API URL", key="orchestrator_url")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Check health", use_container_width=True):
                client = _client()
                try:
                    health = client.health()
                    st.success(f"{health.get('service')} {health.get('version')}")
                except OrchestratorUIError as exc:
                    st.error(str(exc))
                finally:
                    client.close()
        with col_b:
            if st.button("Reload", use_container_width=True):
                _load_session()

        st.divider()
        st.header("Session")
        st.text_input("Session ID", key="session_id")
        st.text_input("User ID", key="user_id")
        st.text_input(
            "Active DB ID",
            key="active_db_id",
            help="Optional. The agent can also choose/switch DBs using tools.",
        )
        if st.button("Reset session", type="secondary", use_container_width=True):
            _reset_session()
            st.rerun()

        session = st.session_state.last_session
        if session:
            st.divider()
            st.caption(f"Active DB: `{session.get('active_db_id') or '-'}`")
            if session.get("last_sql"):
                st.caption("Last SQL available")
            history = sql_history_from_session(session)
            if history:
                st.subheader("SQL History")
                for i, entry in enumerate(history[:5], start=1):
                    st.caption(format_history_label(i, entry))


def _render_artifacts() -> None:
    session = st.session_state.last_session
    if not session:
        return
    extra = session.get("extra") or {}
    last_sql = session.get("last_sql")
    if last_sql:
        with st.expander("Last SQL", expanded=True):
            st.code(last_sql, language="sql")

    if extra.get("last_result_export"):
        with st.expander("Last Result Export", expanded=False):
            st.code(str(extra["last_result_export"]))

    pending = extra.get("pending_confirmation")
    if pending:
        st.warning("Write SQL is waiting for explicit confirmation.")
        with st.expander("Pending Write SQL", expanded=True):
            st.caption(f"Database: `{pending.get('db_id') or '-'}`")
            st.code(str(pending.get("sql") or ""), language="sql")
            if pending.get("rationale"):
                st.caption(str(pending["rationale"]))

    row_count = extra.get("last_row_count")
    columns = extra.get("last_rows_columns")
    if row_count is not None or columns:
        st.caption(
            f"Latest result: {row_count if row_count is not None else '?'} rows"
            + (f" | columns: {', '.join(columns)}" if columns else "")
        )


def main() -> None:
    st.set_page_config(page_title="Text-to-SQL Agent", page_icon=":speech_balloon:")
    _init_state()

    st.title("Text-to-SQL Agent")
    st.caption("Conversational orchestrator over the text-to-SQL pipeline")

    _render_sidebar()

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    _render_artifacts()

    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "human" else "assistant"
        with st.chat_message(role):
            st.markdown(msg["content"])

    prompt = st.chat_input(
        "Ask a question, modify the last SQL, export results, or inspect a database..."
    )
    if prompt:
        with st.spinner("Agent is thinking..."):
            _send_message(prompt)
        st.rerun()


if __name__ == "__main__":
    main()
