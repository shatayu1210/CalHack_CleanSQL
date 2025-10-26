import html
import os
import tempfile
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from llm_integration import DataAssistant
from profiler import profile_csv

st.set_page_config(
    page_title="CleanSQL Assistant",
    page_icon="🧼",
    layout="wide",
)

CUSTOM_PAGE_STYLE = """
<style>
:root {
    --bg-950: #020617;
    --bg-900: #0b1220;
    --bg-800: #111b2e;
    --bg-700: #1a2840;
    --text-100: #f8fafc;
    --text-200: #e2e8f0;
    --text-400: #cbd5f5;
    --text-muted: rgba(226, 232, 240, 0.7);
    --accent-sky: #38bdf8;
    --accent-indigo: #6366f1;
    --accent-indigo-dark: #4338ca;
    --pill-bg: rgba(99, 102, 241, 0.22);
    --pill-border: rgba(129, 140, 248, 0.3);
}

body {
    color: var(--text-200);
    background: var(--bg-900);
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 15% 20%, rgba(79, 70, 229, 0.18), transparent 55%),
        radial-gradient(circle at 85% 25%, rgba(14, 165, 233, 0.14), transparent 60%),
        var(--bg-900);
    color: var(--text-200);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(17, 24, 39, 0.95));
    color: var(--text-200);
}

[data-testid="stAppViewContainer"] * {
    color: inherit;
}

[data-testid="stHeader"] {
    background: transparent;
}

.hero {
    padding: 2.75rem 3rem;
    border-radius: 1.4rem;
    background: linear-gradient(135deg, rgba(14, 23, 42, 0.95), rgba(79, 70, 229, 0.8));
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 28px 60px rgba(2, 6, 23, 0.45);
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    border-radius: 999px;
    padding: 0.35rem 1.1rem;
    background: rgba(148, 163, 184, 0.45);
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    margin-bottom: 0.75rem;
    color: rgba(255, 255, 255, 0.95);
}

.hero h1 {
    font-size: 2.6rem;
    font-weight: 700;
    margin-bottom: 0.65rem;
    color: var(--text-100);
}

.hero p {
    font-size: 1.06rem;
    max-width: 42rem;
    line-height: 1.65;
    color: rgba(241, 245, 249, 0.88);
}

.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 0.85rem;
    color: var(--text-100);
    letter-spacing: 0.01em;
}

.helper-text {
    font-size: 0.9rem;
    color: var(--text-muted);
    margin-top: 0.5rem;
}

.assistant-answer {
    font-size: 1rem;
    line-height: 1.7;
    color: var(--text-200);
    margin-bottom: 0.65rem;
}

.assistant-notes {
    font-size: 0.87rem;
    color: var(--text-muted);
}

.dataset-meta {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 0.75rem;
}

.dataset-meta .pill {
    background: var(--pill-bg);
    border: 1px solid var(--pill-border);
    border-radius: 999px;
    padding: 0.45rem 1rem;
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-100);
}

div[data-testid="stAlert"] {
    border-radius: 1rem;
    box-shadow: 0 16px 35px rgba(2, 6, 23, 0.4);
    background: rgba(15, 23, 42, 0.85);
    color: var(--text-200);
}

.follow-up-list {
    padding-left: 1.2rem;
    margin-bottom: 0;
}

.follow-up-list li {
    line-height: 1.65;
    color: var(--text-200);
}

[data-testid="stTextArea"] textarea {
    border-radius: 1rem !important;
    border: 1px solid rgba(99, 102, 241, 0.45) !important;
    box-shadow: none !important;
    font-size: 1rem;
    color: var(--text-100) !important;
    background: rgba(17, 24, 39, 0.85) !important;
}

[data-testid="stTextArea"] textarea:focus {
    border-color: rgba(99, 102, 241, 0.85) !important;
    box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.35) !important;
}

.stButton>button {
    border-radius: 999px;
    padding: 0.7rem 1.9rem;
    background: linear-gradient(135deg, var(--accent-indigo), var(--accent-sky));
    border: none;
    color: white;
    font-weight: 600;
    letter-spacing: 0.01em;
    box-shadow: 0 18px 32px rgba(14, 23, 42, 0.55);
}

.stButton>button:hover {
    filter: brightness(1.1);
}

.stDataFrame {
    border-radius: 1rem;
    overflow: hidden;
    border: 1px solid rgba(148, 163, 184, 0.2);
    background: rgba(15, 23, 42, 0.7);
}
</style>
"""

st.markdown(CUSTOM_PAGE_STYLE, unsafe_allow_html=True)

HAS_STREAMLIT_MODAL = hasattr(st, "modal")


def format_bytes(num_bytes: int) -> str:
    """Return a human readable string for a byte count."""
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:,.1f} {unit}"
        value /= 1024
    return f"{value:,.1f} TB"


def render_dataset_summary(dataframe: pd.DataFrame, filename: Optional[str]) -> None:
    """Show dataset headline stats."""
    rows, cols = dataframe.shape
    memory = format_bytes(dataframe.memory_usage(deep=True).sum())
    with st.container():
        st.markdown(
            '<div class="section-title">Dataset overview</div>', unsafe_allow_html=True
        )
        st.markdown(
            f"<p><strong>{filename or 'Uploaded data'}</strong></p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="dataset-meta">
                <span class="pill">{rows:,} rows</span>
                <span class="pill">{cols:,} columns</span>
                <span class="pill">{memory}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def parse_question_answer(raw_answer: Any) -> Tuple[Optional[str], str]:
    """Extract question and answer segments from assistant output, if present."""
    if raw_answer is None:
        return None, ""

    text = str(raw_answer).strip()
    lower_text = text.lower()
    question_marker = "question:"
    answer_marker = "answer:"

    question: Optional[str] = None
    answer = text

    if answer_marker in lower_text:
        answer_idx = lower_text.find(answer_marker)
        answer_start = answer_idx + len(answer_marker)
        answer = text[answer_start:].strip()

        prefix_lower = lower_text[:answer_idx]
        if question_marker in prefix_lower:
            question_idx = prefix_lower.find(question_marker)
            question_start = question_idx + len(question_marker)
            question = text[question_start:answer_idx].strip()

        if not answer:
            answer = text
    elif question_marker in lower_text:
        question_idx = lower_text.find(question_marker)
        question_start = question_idx + len(question_marker)
        question = text[question_start:].strip()
        answer = ""

    return question, answer if answer else text


def to_html_block(text: str, css_class: str) -> str:
    """Return escaped HTML block with preserved line breaks."""
    safe_text = html.escape(text).replace("\n", "<br>")
    return f'<div class="{css_class}">{safe_text}</div>'


def load_dataframe(uploaded_file) -> pd.DataFrame:
    """Return a DataFrame for CSV or Excel uploads."""
    uploaded_file.seek(0)
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Upload a CSV or Excel file.")


def reset_assistant_state():
    """Clean up assistant resources and cached outputs."""
    assistant = st.session_state.get("assistant")
    if assistant is not None:
        try:
            assistant.close()
        except Exception:
            pass
    st.session_state.assistant = None
    st.session_state.profile = None
    st.session_state.query_ready = False
    st.session_state.last_answer = None
    st.session_state.last_sql = None
    st.session_state.last_raw_sql = None
    st.session_state.last_robust_sql = None
    st.session_state.result_df = None
    st.session_state.result_truncated = False
    st.session_state.last_error = None
    st.session_state.last_notes = ""
    st.session_state.last_question = None
    st.session_state.follow_up_questions = []
    st.session_state.show_raw_modal = False
    st.session_state.show_robust_modal = False
    st.session_state.is_running_query = False


def _render_query_popup(
    title: str, sql_text: Optional[str], state_key: str, placeholder
) -> None:
    """Render a modal-like popup for showing SQL snippets."""
    if not sql_text:
        st.session_state[state_key] = False
        placeholder.empty()
        return

    if not st.session_state.get(state_key):
        placeholder.empty()
        return

    if HAS_STREAMLIT_MODAL:
        with st.modal(title):
            st.code(sql_text, language="sql")
            if st.button("Close", key=f"{state_key}_close_modal", type="primary"):
                st.session_state[state_key] = False
    else:
        with placeholder.container():
            st.markdown(f"**{title}**")
            st.code(sql_text, language="sql")
            if st.button(
                "Close",
                key=f"{state_key}_close_fallback",
                type="primary",
                help="Hide this query preview",
            ):
                st.session_state[state_key] = False


def initialise_assistant(dataframe: pd.DataFrame, filename: str) -> Optional[str]:
    """Prepare the DataAssistant and return an error message on failure."""
    reset_assistant_state()

    fd, temp_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    dataframe.to_csv(temp_path, index=False)

    try:
        profile = profile_csv(temp_path, None, 10, True, False)
        profile.setdefault("dataset", {})
        profile["dataset"]["filename"] = filename

        assistant = DataAssistant()
        assistant.setup_database(profile, temp_path)

        st.session_state.assistant = assistant
        st.session_state.profile = profile
        st.session_state.query_ready = True
        st.session_state.last_error = None
        st.session_state.prompt_input = ""
        return None
    except Exception as exc:
        reset_assistant_state()
        return str(exc)
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


# Initialise Streamlit session state
DEFAULT_STATE: Dict[str, Any] = {
    "dataframe": None,
    "assistant": None,
    "profile": None,
    "filename": None,
    "file_signature": None,
    "query_ready": False,
    "prompt_input": "",
    "is_running_query": False,
    "last_question": None,
    "last_answer": None,
    "last_sql": None,
    "last_raw_sql": None,
    "last_robust_sql": None,
    "result_df": None,
    "result_truncated": False,
    "last_error": None,
    "last_notes": "",
    "follow_up_questions": [],
    "show_raw_modal": False,
    "show_robust_modal": False,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        if isinstance(value, (list, dict)):
            st.session_state[key] = value.copy()
        else:
            st.session_state[key] = value


st.markdown(
    """
    <section class="hero">
        <span class="hero-badge">CleanSQL Assistant</span>
        <h1>Query your data at the speed of thought</h1>
        <p>Upload your tabular data, generate clean SQL, and explore insights instantly — no manual querying required.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

upload_col, tips_col = st.columns([3, 2])

with upload_col:
    st.markdown("<h3>1. Upload your dataset</h3>", unsafe_allow_html=True)
    st.markdown(
        "<p>Select a CSV or Excel file to profile and explore.</p>",
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )
    st.markdown(
        '<div class="helper-text">Supports CSV, XLSX, or XLS files up to 50MB.</div>',
        unsafe_allow_html=True,
    )

with tips_col:
    st.markdown("<h3>Tips for best results</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <ul class="follow-up-list">
            <li>Keep column headers descriptive — they improve the SQL context.</li>
            <li>Ask detailed questions to receive better SQL and analysis.</li>
            <li>Use the robust SQL if you plan to run the query elsewhere.</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )

if uploaded_file is None:
    if st.session_state.file_signature is not None:
        reset_assistant_state()
        st.session_state.dataframe = None
        st.session_state.filename = None
        st.session_state.file_signature = None
        st.session_state.prompt_input = ""
        st.session_state.last_notes = ""
        st.session_state.follow_up_questions = []
else:
    signature = (uploaded_file.name, getattr(uploaded_file, "size", None))
    if st.session_state.file_signature != signature:
        try:
            dataframe = load_dataframe(uploaded_file)
            st.session_state.dataframe = dataframe
            st.session_state.filename = uploaded_file.name
            st.session_state.file_signature = signature
            st.session_state.result_df = None
            st.session_state.last_answer = None
            st.session_state.last_sql = None
            st.session_state.last_raw_sql = None
            st.session_state.last_robust_sql = None
            st.session_state.prompt_input = ""
            st.session_state.last_question = None
            st.session_state.last_notes = ""
            st.session_state.follow_up_questions = []
            st.session_state.show_raw_modal = False
            st.session_state.show_robust_modal = False
            st.session_state.is_running_query = False

            error_message = initialise_assistant(dataframe, uploaded_file.name)
            if error_message:
                st.session_state.last_error = error_message
            else:
                st.success("Data loaded. Ask a question about your dataset.")
        except Exception as exc:
            reset_assistant_state()
            st.session_state.dataframe = None
            st.session_state.filename = None
            st.session_state.file_signature = None
            st.session_state.last_error = str(exc)
            st.session_state.last_notes = ""
            st.session_state.follow_up_questions = []


if st.session_state.dataframe is None:
    st.markdown(
        '<div class="section-title">Awaiting data</div>'
        "<p>Upload a CSV or Excel file to profile your dataset and start asking questions.</p>",
        unsafe_allow_html=True,
    )
    if st.session_state.last_error:
        st.error(st.session_state.last_error)
    st.stop()

render_dataset_summary(st.session_state.dataframe, st.session_state.filename)

with st.container():
    st.markdown(
        '<div class="section-title">Uploaded data preview</div>', unsafe_allow_html=True
    )
    st.dataframe(st.session_state.dataframe, use_container_width=True)

st.markdown(
    '<div class="section-title">2. Ask anything about this dataset</div>',
    unsafe_allow_html=True,
)

query_col, button_col = st.columns([3, 1])

with query_col:
    prompt = st.text_area(
        "Natural language prompt",
        key="prompt_input",
        placeholder='Example: "Show me the top 10 rows" or "What is the average salary?"',
        disabled=(
            not st.session_state.query_ready or st.session_state.is_running_query
        ),
        height=160,
        label_visibility="collapsed",
    )

with button_col:
    st.markdown("<h3>Run it</h3>", unsafe_allow_html=True)
    ask_button = st.button(
        "Run Query",
        type="primary",
        disabled=(
            not st.session_state.query_ready or st.session_state.is_running_query
        ),
        use_container_width=True,
    )

prompt = st.session_state.prompt_input

if ask_button:
    question = (prompt or "").strip()
    if not question:
        st.session_state.is_running_query = False
        st.session_state.last_error = "Please enter a question to run."
        st.session_state.last_question = None
        st.session_state.last_answer = None
        st.session_state.last_sql = None
        st.session_state.last_raw_sql = None
        st.session_state.last_robust_sql = None
        st.session_state.result_df = None
        st.session_state.result_truncated = False
        st.session_state.last_notes = ""
        st.session_state.follow_up_questions = []
        st.session_state.show_raw_modal = False
        st.session_state.show_robust_modal = False
    else:
        st.session_state.last_question = question
        assistant = st.session_state.assistant
        profile = st.session_state.profile
        if assistant is None or profile is None:
            st.session_state.is_running_query = False
            st.session_state.last_error = (
                "Assistant not initialised. Please re-upload your dataset."
            )
            st.session_state.last_answer = None
            st.session_state.last_sql = None
            st.session_state.last_raw_sql = None
            st.session_state.last_robust_sql = None
            st.session_state.result_df = None
            st.session_state.result_truncated = False
            st.session_state.last_notes = ""
            st.session_state.follow_up_questions = []
            st.session_state.show_raw_modal = False
            st.session_state.show_robust_modal = False
        else:
            st.session_state.is_running_query = True
            try:
                with st.spinner("Running query..."):
                    response = assistant.ask_question(question, profile)
                if not isinstance(response, dict):
                    st.session_state.last_answer = str(response)
                    st.session_state.last_sql = None
                    st.session_state.last_raw_sql = None
                    st.session_state.last_robust_sql = None
                    st.session_state.result_df = None
                    st.session_state.result_truncated = False
                    st.session_state.last_error = None
                    st.session_state.last_notes = ""
                    st.session_state.follow_up_questions = []
                    st.session_state.show_raw_modal = False
                    st.session_state.show_robust_modal = False
                elif response.get("error"):
                    st.session_state.last_error = response["error"]
                    st.session_state.last_answer = None
                    robust_sql = response.get("robust_sql") or response.get("sql")
                    st.session_state.last_sql = robust_sql
                    st.session_state.last_raw_sql = response.get("raw_sql")
                    st.session_state.last_robust_sql = robust_sql
                    st.session_state.result_df = None
                    st.session_state.result_truncated = False
                    st.session_state.last_notes = response.get("notes", "")
                    st.session_state.follow_up_questions = (
                        response.get("follow_up_questions") or []
                    )
                    st.session_state.show_raw_modal = False
                    st.session_state.show_robust_modal = False
                else:
                    st.session_state.last_answer = response.get("answer")
                    robust_sql = response.get("robust_sql") or response.get("sql")
                    st.session_state.last_sql = robust_sql
                    st.session_state.last_raw_sql = response.get("raw_sql")
                    st.session_state.last_robust_sql = robust_sql
                    rows = response.get("rows") or []
                    columns = response.get("columns") or []
                    if rows:
                        st.session_state.result_df = pd.DataFrame(rows, columns=columns)
                    else:
                        st.session_state.result_df = pd.DataFrame(columns=columns)
                    st.session_state.result_truncated = bool(
                        response.get("truncated", False)
                    )
                    st.session_state.last_error = None
                    st.session_state.last_notes = response.get("notes", "")
                    st.session_state.follow_up_questions = (
                        response.get("follow_up_questions") or []
                    )
                    st.session_state.show_raw_modal = False
                    st.session_state.show_robust_modal = False
            except Exception as exc:
                st.session_state.last_error = str(exc)
                st.session_state.last_answer = None
                st.session_state.last_sql = None
                st.session_state.last_raw_sql = None
                st.session_state.last_robust_sql = None
                st.session_state.result_df = None
                st.session_state.result_truncated = False
                st.session_state.last_notes = ""
                st.session_state.follow_up_questions = []
                st.session_state.show_raw_modal = False
                st.session_state.show_robust_modal = False
            finally:
                st.session_state.is_running_query = False

if st.session_state.last_error:
    st.error(st.session_state.last_error)

parsed_question, parsed_answer = parse_question_answer(st.session_state.last_answer)
if st.session_state.last_question is None and parsed_question:
    st.session_state.last_question = parsed_question

display_answer = parsed_answer.strip()

if display_answer or st.session_state.last_notes:
    st.markdown(
        '<div class="section-title">Assistant response</div>', unsafe_allow_html=True
    )
    if display_answer:
        st.markdown(
            to_html_block(display_answer, "assistant-answer"),
            unsafe_allow_html=True,
        )
    if st.session_state.last_notes:
        st.markdown(
            to_html_block(st.session_state.last_notes, "assistant-notes"),
            unsafe_allow_html=True,
        )

if st.session_state.last_raw_sql is None:
    st.session_state.show_raw_modal = False
if st.session_state.last_robust_sql is None:
    st.session_state.show_robust_modal = False

if st.session_state.last_raw_sql or st.session_state.last_robust_sql:
    st.markdown('<div class="section-title">SQL previews</div>', unsafe_allow_html=True)
    raw_disabled = not bool(st.session_state.last_raw_sql)
    robust_disabled = not bool(st.session_state.last_robust_sql)
    col_raw, col_robust = st.columns(2)
    with col_raw:
        if st.button(
            "Raw query",
            key="open_raw_query_btn",
            type="secondary",
            disabled=raw_disabled,
            use_container_width=True,
        ):
            st.session_state.show_raw_modal = True
    with col_robust:
        if st.button(
            "Robust query",
            key="open_robust_query_btn",
            type="secondary",
            disabled=robust_disabled,
            use_container_width=True,
        ):
            st.session_state.show_robust_modal = True

raw_modal_placeholder = st.container()
robust_modal_placeholder = st.container()

_render_query_popup(
    "Raw query", st.session_state.last_raw_sql, "show_raw_modal", raw_modal_placeholder
)
_render_query_popup(
    "Robust query",
    st.session_state.last_robust_sql,
    "show_robust_modal",
    robust_modal_placeholder,
)

if st.session_state.result_df is not None:
    st.markdown(
        '<div class="section-title">Query results</div>', unsafe_allow_html=True
    )
    if st.session_state.result_df.empty:
        st.markdown(
            "<p class='helper-text'>Query executed successfully but returned no rows.</p>",
            unsafe_allow_html=True,
        )
    else:
        st.dataframe(st.session_state.result_df, use_container_width=True)
        if st.session_state.result_truncated:
            st.caption("Showing the first 200 rows of the result.")

follow_ups = st.session_state.follow_up_questions or []
if follow_ups:
    st.markdown(
        '<div class="section-title">Suggested follow-up questions</div>',
        unsafe_allow_html=True,
    )
    items = "".join(f"<li>{html.escape(item)}</li>" for item in follow_ups)
    st.markdown(f"<ul class='follow-up-list'>{items}</ul>", unsafe_allow_html=True)
