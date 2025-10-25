import os
import tempfile
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from llm_integration import DataAssistant
from profiler import profile_csv


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
    st.session_state.result_df = None
    st.session_state.result_truncated = False
    st.session_state.last_error = None
    st.session_state.last_notes = ""
    st.session_state.follow_up_questions = []


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
    "last_answer": None,
    "last_sql": None,
    "result_df": None,
    "result_truncated": False,
    "last_error": None,
    "last_notes": "",
    "follow_up_questions": [],
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        if isinstance(value, (list, dict)):
            st.session_state[key] = value.copy()
        else:
            st.session_state[key] = value


uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file", type=["csv", "xlsx", "xls"]
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
            st.session_state.prompt_input = ""
            st.session_state.last_notes = ""
            st.session_state.follow_up_questions = []

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
    st.info("Upload a CSV or Excel file to explore your data.")
    if st.session_state.last_error:
        st.error(st.session_state.last_error)
    st.stop()

st.subheader("Uploaded Data Preview")
st.dataframe(st.session_state.dataframe)

st.write("---")
st.subheader("Ask a Question")

prompt = st.text_area(
    "Natural language prompt",
    key="prompt_input",
    placeholder="Example: \"Show me the top 10 rows\" or \"What is the average salary?\"",
    disabled=not st.session_state.query_ready,
    height=120,
)

ask_button = st.button(
    "Run Query", type="primary", disabled=not st.session_state.query_ready
)

if ask_button:
    question = (prompt or "").strip()
    if not question:
        st.session_state.last_error = "Please enter a question to run."
        st.session_state.last_answer = None
        st.session_state.last_sql = None
        st.session_state.result_df = None
        st.session_state.result_truncated = False
        st.session_state.last_notes = ""
        st.session_state.follow_up_questions = []
    else:
        assistant = st.session_state.assistant
        profile = st.session_state.profile
        if assistant is None or profile is None:
            st.session_state.last_error = (
                "Assistant not initialised. Please re-upload your dataset."
            )
            st.session_state.last_notes = ""
            st.session_state.follow_up_questions = []
        else:
            try:
                response = assistant.ask_question(question, profile)
                if not isinstance(response, dict):
                    st.session_state.last_answer = str(response)
                    st.session_state.last_sql = None
                    st.session_state.result_df = None
                    st.session_state.result_truncated = False
                    st.session_state.last_error = None
                    st.session_state.last_notes = ""
                    st.session_state.follow_up_questions = []
                elif response.get("error"):
                    st.session_state.last_error = response["error"]
                    st.session_state.last_answer = None
                    st.session_state.last_sql = response.get("sql")
                    st.session_state.result_df = None
                    st.session_state.result_truncated = False
                    st.session_state.last_notes = response.get("notes", "")
                    st.session_state.follow_up_questions = (
                        response.get("follow_up_questions") or []
                    )
                else:
                    st.session_state.last_answer = response.get("answer")
                    st.session_state.last_sql = response.get("sql")
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
            except Exception as exc:
                st.session_state.last_error = str(exc)
                st.session_state.last_answer = None
                st.session_state.last_sql = None
                st.session_state.result_df = None
                st.session_state.result_truncated = False
                st.session_state.last_notes = ""
                st.session_state.follow_up_questions = []

if st.session_state.last_error:
    st.error(st.session_state.last_error)

if st.session_state.last_answer:
    st.subheader("Assistant Response")
    st.write(st.session_state.last_answer)

if st.session_state.last_notes:
    st.info(st.session_state.last_notes)

if st.session_state.last_sql:
    with st.expander("Generated SQL", expanded=False):
        st.code(st.session_state.last_sql, language="sql")

if st.session_state.result_df is not None:
    st.subheader("Query Results")
    if st.session_state.result_df.empty:
        st.info("Query executed successfully but returned no rows.")
    else:
        st.dataframe(st.session_state.result_df)
        if st.session_state.result_truncated:
            st.caption("Showing the first 200 rows of the result.")

follow_ups = st.session_state.follow_up_questions or []
if follow_ups:
    with st.expander("Follow-up Questions", expanded=False):
        for item in follow_ups:
            st.markdown(f"- {item}")
