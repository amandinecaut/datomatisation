import streamlit as st
import pandas as pd
import json
import google.generativeai as genai

# load secrets from .streamlit/secrets.toml
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(layout="wide")


from app_utilities import (
    perform_pca,
    update_df,
    load_data,
    load_map,
    update_map,
)


height = 700  # height of the container
DEFAULT_N_COMPONENTS = 5

# track interaction prompt
if "interaction_prompt" not in st.session_state:
    st.session_state.interaction_prompt = ""
if "entity_col" not in st.session_state:
    st.session_state.entity_col = "Index"
if "features" not in st.session_state:
    st.session_state.features = []
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "pca_df" not in st.session_state:
    st.session_state.pca_df = None

# Add and app header
st.title("Automated PCA pipeline")

tab1, tab2, tab3 = st.tabs(["Load data", "PCA", "View"])

with tab1:

    left_t1, right_t1 = st.columns([0.25, 0.75])
    # Left pane title

    left_t1 = left_t1.container(height=height, border=0)
    right_t1 = right_t1.container(height=height, border=3)

    # left all box to upload data
    left_t1.markdown("### Upload data")
    left_t1.file_uploader(
        "Choose a file",
        type=["csv"],
        key="file",
        on_change=load_data,
    )

    if st.session_state.file is None:
        df = pd.read_csv("./data/data-final-sample.csv")
    else:
        df = pd.read_csv(st.session_state.file)
    update_df(df)

    left_t1.markdown("### Upload column name mapping")
    left_t1.file_uploader(
        "Choose a file",
        type=["json"],
        key="map",
        on_change=load_map,
    )

    # read json file
    if st.session_state.map is None:
        with open("./data/map.json", "r") as f:
            map = json.load(f)
    else:
        with open(st.session_state.map, "r") as f:
            map = json.load(f)

    update_map(map)

    # display the info of the data
    right_t1.markdown("### Data info")

    cols = ["Index"] + df.columns.to_list()
    # drop down "select entity", default to "Index"
    entity = right_t1.selectbox(
        "Select entity",
        cols,
        index=0,
        key="entity_col",
        on_change=update_df,
        args=(df,),
    )
    update_df(df)

    default_ignore = [c for c in df.columns.to_list() if c not in map.keys()]
    # print(default_ignore)

    # add check box to ignore certain columns
    ignore_cols = right_t1.multiselect(
        label="Ignore columns",
        options=df.columns.to_list(),
        default=default_ignore,
        on_change=update_df,
        args=(df,),
        key="ignore_cols",
    )
    update_df(df)

    # display the first 5 rows
    right_t1.write("Sample of data")
    right_t1.write(df[st.session_state.features].sample(10))

    # disply warning if there are rows with NaN
    if df[st.session_state.features].isnull().any(axis=1).sum() > 0:
        right_t1.warning("There are rows with NaN, these will be dropped.")
    # display rows with NaN

    right_t1.write("Rows with NaN")
    right_t1.write(df[df[st.session_state.features].isnull().any(axis=1)])

    right_t1.write("Data to use")
    right_t1.write(st.session_state.df)

    # show col_mapping as a table
    right_t1.write("Column mapping")
    right_t1.write(st.session_state.col_mapping)


with tab2:

    left_t2, right_t2 = st.columns([0.25, 0.75])
    # Left pane title

    left_t2 = left_t2.container(height=height, border=0)
    right_t2 = right_t2.container(height=height, border=3)

    left_t2.markdown("## PCA")

    # # if st.session_state.pca_df is and empty dataframe, perform PCA
    # if st.session_state.pca_df.empty:
    #     perform_pca(df)

    # left select the number of PCA components, max the length of features
    n_components = left_t2.slider(
        "Select the number of PCA components",
        1,
        len(st.session_state.features),
        len(
            st.session_state.features
        ),  # int(len(st.session_state.features) / 2),  # DEFAULT_N_COMPONENTS,
        key="n_components",
        on_change=perform_pca,
        args=(right_t2,),
    )
    right_t2.markdown("### PCA results")
    perform_pca(right_t2)


with tab3:

    left_t3, right_t3 = st.columns([0.25, 0.75])
    # Left pane title

    left_t3 = left_t3.container(height=height, border=0)
    right_t3 = right_t3.container(height=height, border=3)

    left_t3.markdown("## View")

# debug
# print()
# print(st.session_state)
# print("foo")
# for key, value in st.session_state.items():
#     print(key)  # , value)
