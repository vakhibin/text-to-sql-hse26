"""Streamlit DBeaver-light UI for the conversational text-to-SQL agent."""

from __future__ import annotations

import os
import time
import uuid

import streamlit as st

from services.ui.client import (
    DEFAULT_ORCHESTRATOR_URL,
    DEFAULT_TEXT_TO_SQL_URL,
    DEFAULT_UI_TIMEOUT_S,
    OrchestratorUIClient,
    OrchestratorUIError,
    TextToSQLUIClient,
    TextToSQLUIError,
    database_options,
    format_history_label,
    latest_result_from_session,
    latest_run_meta_from_session,
    normalize_base_url,
    normalize_text_to_sql_url,
    schema_tables,
    session_extra,
    sql_history_from_session,
    visible_messages,
)


def _query_param_value(name: str) -> str | None:
    raw = st.query_params.get(name)
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    value = str(raw or "").strip()
    return value or None


def _sync_session_query_param() -> None:
    session_id = str(st.session_state.get("session_id") or "").strip()
    if session_id and _query_param_value("session_id") != session_id:
        st.query_params["session_id"] = session_id


def _init_state() -> None:
    initial_session_id = _query_param_value("session_id") or f"ui-{uuid.uuid4().hex[:8]}"
    st.session_state.setdefault("session_id", initial_session_id)
    st.session_state.setdefault("user_id", "streamlit-user")
    st.session_state.setdefault(
        "orchestrator_url",
        os.getenv("ORCHESTRATOR_API_URL", DEFAULT_ORCHESTRATOR_URL),
    )
    st.session_state.setdefault(
        "text_to_sql_url",
        os.getenv("TEXT_TO_SQL_API_URL", DEFAULT_TEXT_TO_SQL_URL),
    )
    st.session_state.setdefault("active_db_id", "")
    st.session_state.setdefault("active_db_id_input", st.session_state.active_db_id)
    st.session_state.setdefault(
        "orchestrator_timeout_s",
        float(os.getenv("ORCHESTRATOR_UI_TIMEOUT_S", DEFAULT_UI_TIMEOUT_S)),
    )
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("sql_editor", "")
    st.session_state.setdefault("sql_editor_source_sql", "")
    st.session_state.setdefault("reload_sql_editor", False)
    st.session_state.setdefault("sidebar_chat_prompt", "")
    st.session_state.setdefault("last_session", None)
    st.session_state.setdefault("last_error", None)
    st.session_state.setdefault("catalog", None)
    st.session_state.setdefault("catalog_error", None)
    st.session_state.setdefault("schema_cache", {})
    st.session_state.setdefault("browse_db_id", "")
    # Latest /chat turn's Langfuse trace URL (orchestrator span). Surfaced as
    # a button next to "Open Langfuse trace" so the user can jump straight to
    # the conversational trace; per-pipeline-run links live in the pipeline
    # panel via meta["langfuse_trace_url"].
    st.session_state.setdefault("last_chat_trace_url", None)
    _sync_session_query_param()


def _inject_styles() -> None:
    """Small visual polish layer for the Streamlit workbench."""
    st.markdown(
        """
        <style>
        div[data-testid="stSidebar"] section[data-testid="stSidebarContent"] {
            padding-top: 1.25rem;
        }
        div[data-testid="stTextArea"] textarea {
            font-family: "SFMono-Regular", Menlo, Monaco, Consolas, monospace;
            font-size: 0.9rem;
            line-height: 1.45;
            border-radius: 10px;
        }
        div[data-testid="stDataFrame"] {
            border-radius: 10px;
            overflow: hidden;
        }
        button[kind="primary"] {
            border-radius: 9px;
            font-weight: 650;
        }
        .stStatus {
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _client() -> OrchestratorUIClient:
    return OrchestratorUIClient(
        normalize_base_url(st.session_state.orchestrator_url),
        timeout_s=float(st.session_state.orchestrator_timeout_s),
    )


def _text_to_sql_client() -> TextToSQLUIClient:
    return TextToSQLUIClient(
        normalize_text_to_sql_url(st.session_state.text_to_sql_url)
    )


def _refresh_catalog() -> None:
    client = _text_to_sql_client()
    try:
        st.session_state.catalog = client.list_databases()
        st.session_state.catalog_error = None
        st.session_state.schema_cache = {}
    except TextToSQLUIError as exc:
        st.session_state.catalog_error = str(exc)
        st.session_state.catalog = None
    finally:
        client.close()


def _load_schema(db_id: str) -> dict | None:
    if not db_id:
        return None
    cache = st.session_state.schema_cache
    if db_id in cache:
        return cache[db_id]
    client = _text_to_sql_client()
    try:
        schema = client.get_schema(db_id)
    except TextToSQLUIError as exc:
        st.session_state.catalog_error = str(exc)
        return None
    finally:
        client.close()
    cache[db_id] = schema
    st.session_state.schema_cache = cache
    return schema


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


def _progress_steps(prompt: str) -> list[tuple[str, str]]:
    normalized = prompt.lower()
    if "execute this sql" in normalized:
        return [
            ("Checking read-only guardrails", "Validating that the query is safe to run."),
            ("Executing SQL", "Running the query against the selected database."),
            ("Preparing result table", "Normalizing rows and storing execution history."),
        ]
    if "explain this sql" in normalized or "explain sql" in normalized:
        return [
            ("Reading SQL", "Parsing the query and identifying the referenced tables."),
            ("Preparing explanation", "Writing a concise explanation for the user."),
        ]
    if "modify this sql" in normalized or "refine" in normalized:
        return [
            ("Reading current SQL", "Understanding the existing query shape."),
            ("Applying requested change", "Asking the agent to update the query."),
            ("Syncing editor", "Saving the updated SQL back into the workbench."),
        ]
    return [
        ("Selecting relevant tables", "Retrieving and reranking schema context."),
        ("Building query plan", "Preparing a schema-grounded sketch."),
        ("Generating SQL query", "Creating and validating SQL candidates."),
        ("Preparing answer", "Formatting SQL and preview rows for the chat."),
    ]


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
    st.session_state.last_chat_trace_url = response.get("langfuse_trace_url")
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
    st.session_state.sql_editor = ""
    st.session_state.sql_editor_source_sql = ""


def _render_database_browser() -> None:
    st.header("Databases")
    st.text_input("Text-to-SQL API URL", key="text_to_sql_url")
    if st.button("Refresh databases", use_container_width=True):
        _refresh_catalog()

    if st.session_state.catalog is None and st.session_state.catalog_error is None:
        st.caption("Click Refresh to load the catalog from text_to_sql_api.")
        return

    if st.session_state.catalog_error and st.session_state.catalog is None:
        st.error(st.session_state.catalog_error)
        return

    options = database_options(st.session_state.catalog)
    if not options:
        st.warning("No databases reported by text_to_sql_api.")
        return

    labels = [item["label"] for item in options]
    db_ids = [item["db_id"] for item in options]

    current = st.session_state.browse_db_id or st.session_state.active_db_id
    default_idx = db_ids.index(current) if current in db_ids else 0

    selected_label = st.selectbox(
        "Browse",
        labels,
        index=default_idx,
        key="browse_select_label",
    )
    selected_db_id = db_ids[labels.index(selected_label)]
    st.session_state.browse_db_id = selected_db_id

    if st.button(
        f"Use '{selected_db_id}' in chat",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.active_db_id_input = selected_db_id
        st.session_state.active_db_id = selected_db_id
        with st.spinner("Switching active database..."):
            _send_message(f"Use {selected_db_id} database.")
        st.rerun()

    schema = _load_schema(selected_db_id)
    if schema is None:
        st.caption("Schema not available for this database.")
        return

    tables = schema_tables(schema)
    if not tables:
        st.caption("Database reports no tables.")
        return

    st.caption(f"Tables in `{selected_db_id}` ({len(tables)})")
    for table in tables:
        with st.expander(table["name"], expanded=False):
            pks = set(table["primary_keys"])
            for col in table["columns"]:
                marker = "PK" if col["name"] in pks else ""
                col_type = col["type"] or "?"
                st.markdown(
                    f"- **{col['name']}** `{col_type}`"
                    + (f"  _(PK)_" if marker else "")
                )
            if table["foreign_keys"]:
                st.caption("Foreign keys:")
                for fk in table["foreign_keys"]:
                    st.markdown(
                        f"- `{fk['column']}` → "
                        f"`{fk['ref_table']}.{fk['ref_column']}`"
                    )


def _render_control_panel() -> None:
    _render_database_browser()

    st.divider()
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
    _sync_session_query_param()
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


def _run_prompt(prompt: str, *, label: str | None = None) -> None:
    prompt = prompt.strip()
    if not prompt:
        st.warning("Enter a request first.")
        return
    with st.status(label or "Working on your request...", expanded=True) as status:
        for step_title, step_detail in _progress_steps(prompt):
            st.write(f"**{step_title}**")
            st.caption(step_detail)
            time.sleep(0.12)
        status.update(label="Waiting for agent response...", state="running", expanded=True)
        ok = _send_message(prompt)
        if ok:
            status.update(label="Done", state="complete", expanded=False)
        else:
            status.update(label="Request failed", state="error", expanded=True)
    st.rerun()


def _sync_sql_editor_from_session() -> None:
    """Load newly generated SQL into the editor without clobbering manual edits."""
    session = st.session_state.last_session or {}
    last_sql = str(session.get("last_sql") or "").strip()
    if not last_sql:
        return
    if st.session_state.reload_sql_editor or last_sql != st.session_state.sql_editor_source_sql:
        st.session_state.sql_editor = last_sql
        st.session_state.sql_editor_source_sql = last_sql
        st.session_state.reload_sql_editor = False


def _active_db_label() -> str:
    session = st.session_state.last_session or {}
    return (
        st.session_state.active_db_id_input.strip()
        or st.session_state.active_db_id.strip()
        or str(session.get("active_db_id") or "").strip()
        or "-"
    )


def _render_chat_sidebar() -> None:
    st.header("Agent Chat")
    st.caption(f"Active DB: `{_active_db_label()}`")

    chat_box = st.container(height=500, border=True)
    with chat_box:
        if not st.session_state.messages:
            st.info(
                "Ask a question, request SQL changes, or ask for an explanation. "
                "Generated SQL appears in the main editor."
            )
        for msg in st.session_state.messages:
            role = "user" if msg["role"] == "human" else "assistant"
            with st.chat_message(role):
                st.markdown(msg["content"])

    with st.form("sidebar_chat_form", clear_on_submit=True):
        prompt = st.text_area(
            "Message",
            key="sidebar_chat_prompt",
            height=90,
            label_visibility="collapsed",
            placeholder="Ask for data, refine SQL, explain the query...",
        )
        submitted = st.form_submit_button("Send", type="primary", use_container_width=True)
    if submitted:
        _run_prompt(prompt, label="Running agent...")


def _execute_sql_from_editor() -> None:
    sql = st.session_state.sql_editor.strip()
    if not sql:
        st.warning("SQL editor is empty.")
        return
    prompt = "Execute this SQL and show the result:\n\n```sql\n" + sql + "\n```"
    _run_prompt(prompt, label="Executing SQL through orchestrator guardrails...")


def _render_result_table() -> None:
    result = latest_result_from_session(st.session_state.last_session)
    row_count = result["row_count"]
    columns = result["columns"]
    rows = result["rows"]

    result_header = st.columns([0.5, 0.5, 1.4])
    result_header[0].metric("Rows", row_count if row_count is not None else "-")
    result_header[1].metric("Columns", len(columns) if columns else 0)
    result_header[2].caption(
        "Latest execution preview"
        if rows
        else "Execute SQL to populate the result table."
    )

    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True, height=360)
    else:
        st.info("No result preview yet.")


def _render_sql_workbench() -> None:
    _sync_sql_editor_from_session()

    st.subheader("SQL Workbench")
    st.caption("Generated SQL is editable. Use the left chat for follow-up changes.")

    st.text_area(
        "SQL editor",
        key="sql_editor",
        height=220,
        label_visibility="collapsed",
        placeholder="SELECT ...",
    )

    action_cols = st.columns([0.9, 0.9, 0.9, 2.4])
    if action_cols[0].button("Execute SQL", type="primary", use_container_width=True):
        _execute_sql_from_editor()
    if action_cols[1].button("Explain", use_container_width=True):
        sql = st.session_state.sql_editor.strip()
        if sql:
            _run_prompt(
                "Explain this SQL in plain language:\n\n```sql\n" + sql + "\n```",
                label="Asking agent to explain SQL...",
            )
        else:
            st.warning("SQL editor is empty.")
    if action_cols[2].button("Reload Last SQL", use_container_width=True):
        st.session_state.reload_sql_editor = True
        st.rerun()

    st.divider()
    _render_result_table()


def _render_right_rail() -> None:
    st.subheader("Run Inspector")
    rail = st.container(height=640, border=False)
    with rail:
        meta = latest_run_meta_from_session(st.session_state.last_session)
        if meta["elapsed_s"] or meta["cost_usd"]:
            metric_cols = st.columns(2)
            metric_cols[0].metric("Latency", f"{meta['elapsed_s']:.2f}s")
            metric_cols[1].metric("Cost", f"${meta['cost_usd']:.4f}")
        else:
            st.caption("Run a query to see traces and metadata.")

        chat_trace_url = st.session_state.get("last_chat_trace_url")
        if chat_trace_url:
            st.link_button(
                "Open chat trace",
                chat_trace_url,
                use_container_width=True,
                help="Inspect the orchestrator turn in Langfuse.",
            )
        if meta.get("langfuse_trace_url"):
            st.link_button(
                "Open pipeline trace",
                meta["langfuse_trace_url"],
                use_container_width=True,
                help="Inspect the text-to-SQL pipeline run in Langfuse.",
            )

        with st.expander("Pipeline stages", expanded=True):
            _render_pipeline_panel()

        with st.expander("Query grounding", expanded=False):
            if meta["selected_tables"]:
                st.caption("Selected tables")
                st.markdown(", ".join(f"`{table}`" for table in meta["selected_tables"]))
            if meta["query_sketch_text"]:
                st.caption("Query sketch")
                st.markdown(str(meta["query_sketch_text"]))
            if not meta["selected_tables"] and not meta["query_sketch_text"]:
                st.caption("No grounding metadata yet.")

        with st.expander("SQL History", expanded=False):
            history = sql_history_from_session(st.session_state.last_session)
            if history:
                for i, entry in enumerate(history[:10], start=1):
                    st.caption(format_history_label(i, entry))
            else:
                st.caption("No SQL history yet.")

        with st.expander("Settings", expanded=False):
            settings_box = st.container(height=420, border=False)
            with settings_box:
                _render_control_panel()

        _render_conversation_log()


def _stage_icon(status: str | None) -> str:
    normalized = (status or "").lower()
    if normalized in {"success", "skipped", "ok", "done"}:
        return "OK"
    if normalized in {"failed", "error"}:
        return "ERR"
    if normalized in {"running", "in_progress"}:
        return "RUN"
    return "PENDING"


def _render_pipeline_panel() -> None:
    meta = latest_run_meta_from_session(st.session_state.last_session)
    stage_status = meta["stage_status"]

    if not stage_status and not meta["trace_id"]:
        st.caption("No pipeline metadata yet.")
        return

    default_stages = [
        "selector",
        "value_linker",
        "query_sketcher",
        "generator",
        "execution_filter",
        "voting",
        "judge",
        "refiner",
    ]
    stages = [stage for stage in default_stages if stage in stage_status]
    stages.extend(stage for stage in stage_status if stage not in stages)

    if stages:
        for stage in stages:
            status = stage_status.get(stage)
            st.markdown(f"- `{_stage_icon(status)}` **{stage}**: `{status or 'pending'}`")

    if meta["trace_id"]:
        st.caption(f"Trace ID: `{meta['trace_id']}`")
    if meta["error"]:
        st.error(str(meta["error"]))
    if meta["warnings"]:
        st.caption(f"Warnings: {len(meta['warnings'])}")
        for warning in meta["warnings"]:
            st.warning(warning)


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

    _inject_styles()

    with st.sidebar:
        _render_chat_sidebar()

    header_left, header_right = st.columns([1.0, 0.32], gap="large")
    with header_left:
        st.title("Text-to-SQL Workbench")
        st.caption("Chat-driven SQL workspace over the conversational text-to-SQL orchestrator")
    with header_right:
        st.write("")
        st.write("")
        with st.popover("Run Inspector", use_container_width=True):
            _render_right_rail()

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    _render_guardrails_panel()
    _render_sql_workbench()


if __name__ == "__main__":
    main()
