import pandas as pd
import streamlit as st


def load_dataframe(uploaded_file):
    """Return a DataFrame for CSV or Excel uploads."""
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Upload a CSV or Excel file.")


if "dataframe" not in st.session_state:
    st.session_state.dataframe = None
if "prompt" not in st.session_state:
    st.session_state.prompt = ""
if "submitted" not in st.session_state:
    st.session_state.submitted = False

uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file", type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    try:
        st.session_state.dataframe = load_dataframe(uploaded_file)
        st.session_state.submitted = False
    except Exception as exc:
        st.session_state.dataframe = None
        st.error(f"Unable to read file: {exc}")

prompt_input = st.text_input(
    "Prompt", placeholder="Enter prompt", value=st.session_state.prompt
)

if st.button("Proceed", type="secondary"):
    st.session_state.prompt = prompt_input
    st.session_state.submitted = True

if st.session_state.submitted:
    if st.session_state.dataframe is None:
        st.info("Upload a CSV or Excel file to display the data here.")
    else:
        st.subheader("Uploaded Data")
        if st.session_state.prompt:
            st.caption(f"Prompt: {st.session_state.prompt}")
        st.dataframe(st.session_state.dataframe)
