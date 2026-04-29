"""Streamlit DBeaver-light UI for the conversational text-to-SQL agent."""

from __future__ import annotations

import os
import uuid

import streamlit as st

from services.ui.client import (
    DEFAULT_ORCHESTRATOR_URL,
    DEFAULT_UI_TIMEOUT_S,
    OrchestratorUIClient,
    OrchestratorUIError,
    format_history_label,
    latest_result_from_session,
    normalize_base_url,
    session_extra,
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
    st.session_state.setdefault("active_db_id_input", st.session_state.active_db_id)
    st.session_state.setdefault(
        "orchestrator_timeout_s",
        float(os.getenv("ORCHESTRATOR_UI_TIMEOUT_S", DEFAULT_UI_TIMEOUT_S)),
    )
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("question_prompt", "")
    st.session_state.setdefault("followup_prompt", "")
    st.session_state.setdefault("last_session", None)
    st.session_state.setdefault("last_error", None)


def _client() -> OrchestratorUIClient:
    return OrchestratorUIClient(
        normalize_base_url(st.session_state.orchestrator_url),
        timeout_s=float(st.session_state.orchestrator_timeout_s),
    )


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


def _active_db_param() -> str | None:
    return (
        st.session_state.active_db_id_input.strip()
        or st.session_state.active_db_id.strip()
        or None
    )


def _send_message(prompt: str) -> bool:
    st.session_state.messages.append({"role": "human", "content": prompt})
    client = _client()
    try:
        response = client.chat(
            session_id=st.session_state.session_id,
            user_id=st.session_state.user_id,
            message=prompt,
            active_db_id=_active_db_param(),
        )
    except OrchestratorUIError as exc:
        st.session_state.last_error = str(exc)
        st.session_state.messages.append(
            {"role": "ai", "content": f"Request failed: {exc}"}
        )
        return False
    finally:
        client.close()

    st.session_state.last_error = None
    if response.get("active_db_id"):
        st.session_state.active_db_id = response["active_db_id"]
    st.session_state.messages.extend(
        visible_messages(response.get("messages_delta") or [])[1:]
    )
    _load_session()
    return True


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
        st.number_input(
            "Request timeout (s)",
            key="orchestrator_timeout_s",
            min_value=30.0,
            max_value=600.0,
            step=30.0,
            help="Full text-to-SQL runs may take 1-3 minutes on first request.",
        )

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
            "Active DB ID override",
            key="active_db_id_input",
            help="Optional. The agent can also choose/switch DBs using tools.",
        )
        if st.session_state.active_db_id_input.strip():
            st.session_state.active_db_id = st.session_state.active_db_id_input.strip()
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


def _run_prompt(prompt: str, *, label: str | None = None) -> None:
    prompt = prompt.strip()
    if not prompt:
        st.warning("Enter a request first.")
        return
    with st.spinner(label or "Running agent pipeline..."):
        _send_message(prompt)
    st.rerun()


def _render_query_workspace() -> None:
    session = st.session_state.last_session
    active_db = (
        st.session_state.active_db_id_input.strip()
        or st.session_state.active_db_id.strip()
        or (session or {}).get("active_db_id")
        or "-"
    )
    st.subheader("Ask Database")
    st.caption(f"Active DB: `{active_db}`")

    with st.form("question_form"):
        question = st.text_area(
            "Natural-language question",
            key="question_prompt",
            height=110,
            placeholder="Example: Show the top 5 singers by number of concerts.",
        )
        submitted = st.form_submit_button("Run Text-to-SQL", type="primary")
    if submitted:
        _run_prompt(
            question,
            label="Running text-to-SQL pipeline... first request can take 1-3 minutes.",
        )

    with st.form("followup_form"):
        followup = st.text_input(
            "Follow-up command",
            key="followup_prompt",
            placeholder="Example: add a filter for 2020, explain SQL, export as csv",
        )
        followup_submitted = st.form_submit_button("Send Follow-up")
    if followup_submitted:
        _run_prompt(followup, label="Sending follow-up command...")


def _render_sql_panel() -> None:
    session = st.session_state.last_session
    last_sql = (session or {}).get("last_sql")
    st.subheader("Generated SQL")
    if last_sql:
        st.code(last_sql, language="sql")
    else:
        st.info("Run a question to generate SQL.")


def _render_result_panel() -> None:
    session = st.session_state.last_session
    extra = session_extra(session)
    result = latest_result_from_session(session)
    row_count = result["row_count"]
    columns = result["columns"]
    rows = result["rows"]

    st.subheader("Latest Result")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Rows", row_count if row_count is not None else "-")
    metric_cols[1].metric("Columns", len(columns) if columns else 0)
    metric_cols[2].metric("Preview Rows", len(rows))

    if columns:
        st.caption("Columns: " + ", ".join(columns))

    with st.expander("Preview rows", expanded=False):
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.caption("No row preview is available yet.")

    export_cols = st.columns(3)
    for fmt, col in zip(("markdown", "csv", "json"), export_cols):
        if col.button(f"Export {fmt.upper()}", use_container_width=True):
            _run_prompt(f"export latest results as {fmt}", label=f"Exporting {fmt}...")

    if extra.get("last_result_export"):
        with st.expander("Last export", expanded=False):
            st.code(str(extra["last_result_export"]))


def _render_guardrails_panel() -> None:
    pending = session_extra(st.session_state.last_session).get("pending_confirmation")
    if pending:
        st.warning("Write SQL is waiting for explicit confirmation.")
        with st.expander("Pending Write SQL", expanded=True):
            st.caption(f"Database: `{pending.get('db_id') or '-'}`")
            st.code(str(pending.get("sql") or ""), language="sql")
            if pending.get("rationale"):
                st.caption(str(pending["rationale"]))


def _render_conversation_log() -> None:
    with st.expander("Conversation log", expanded=False):
        if not st.session_state.messages:
            st.caption("No messages yet.")
            return
        for msg in st.session_state.messages:
            role = "user" if msg["role"] == "human" else "assistant"
            with st.chat_message(role):
                st.markdown(msg["content"])


def main() -> None:
    st.set_page_config(
        page_title="Text-to-SQL Workbench",
        page_icon=":mag:",
        layout="wide",
    )
    _init_state()

    st.title("Text-to-SQL Workbench")
    st.caption("DBeaver-light UI over the conversational text-to-SQL orchestrator")

    _render_sidebar()

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    _render_guardrails_panel()

    top_left, top_right = st.columns([1.05, 0.95], gap="large")
    with top_left:
        _render_query_workspace()
    with top_right:
        _render_sql_panel()

    st.divider()
    _render_result_panel()
    _render_conversation_log()


if __name__ == "__main__":
    main()
