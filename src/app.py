import streamlit as st
import pandas as pd
import json
import google.generativeai as genai

# TODO: Fix upload bug by reevaluation necessary functions

# load secrets from .streamlit/secrets.toml
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(layout="wide")

import app_utilities

from app_utilities import (
    perform_pca,
    update_df,
    load_data,
    load_map,
    display_results,
)


height = 800  # height of the container
DEFAULT_DATA = "./data/data-final-sample.csv"
DEFAULT_MAP = "./data/map.json"


# track interaction prompt
if "entity_col" not in st.session_state:
    st.session_state.entity_col = "Index"
if "pca_df" not in st.session_state:
    st.session_state.pca_df = None

if "file" not in st.session_state:
    load_data()
if "map" not in st.session_state:
    load_map()

# for key, value in st.session_state.items():
#     print(key)  # , value)

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

    left_t1.markdown("### Upload column name mapping")
    left_t1.file_uploader(
        "Choose a file",
        type=["json"],
        key="map",
        on_change=load_map,
    )

    # display the info of the data
    right_t1.markdown("### Data information")

    expander_sample = right_t1.expander("Sample of the data")
    expander_sample.write(st.session_state.df_full.sample(5))

    cols = ["Index"] + st.session_state.df_full.columns.to_list()
    # drop down "select entity", default to "Index"
    entity = right_t1.selectbox(
        "Select column to index data",
        cols,
        index=0,
        key="entity_col",
        on_change=update_df,
    )

    if st.session_state.col_mapping != {}:
        default_ignore = [
            c
            for c in st.session_state.df_full.columns.to_list()
            if c not in st.session_state.col_mapping.keys()
            and c != st.session_state.entity_col
        ]
    else:
        default_ignore = []

    if "ignore_cols" not in st.session_state:
        update_df(default_ignore)

    # add check box to ignore certain columns
    ignore_cols = right_t1.multiselect(
        label="Ignore columns",
        options=st.session_state.df_full.columns.to_list(),
        default=default_ignore,
        on_change=update_df,
        key="ignore_cols",
    )

    # disply warning if there are rows with NaN
    if (
        st.session_state.df_full[st.session_state.features].isnull().any(axis=1).sum()
        > 0
    ):
        right_t1.warning("There are rows containing NaN, these will be dropped.")

    expander_nan = right_t1.expander("Rows containing NaN")
    if st.session_state.entity_col == "Index":
        expander_nan.write(
            st.session_state.df_full[
                st.session_state.df_full[st.session_state.features].isnull().any(axis=1)
            ][st.session_state.features]
        )
    else:
        expander_nan.write(
            st.session_state.df_full[
                st.session_state.df_full[
                    [st.session_state.entity_col] + st.session_state.features
                ]
                .isnull()
                .any(axis=1)
            ][[st.session_state.entity_col] + st.session_state.features]
        )

    expander_data = right_t1.expander("Show all data to be used")
    expander_data.write(st.session_state.df_filtered)
    # show st.session_state.df_filtered with the index column blue

    expander_map = right_t1.expander("Column mapping")
    expander_map.write(st.session_state.col_mapping)


with tab2:

    left_t2, right_t2 = st.columns([0.25, 0.75])
    # Left pane title

    left_t2 = left_t2.container(height=height, border=0)
    right_t2 = right_t2.container(height=height, border=3)

    left_t2.markdown("## PCA")

    if "cum_exp" not in st.session_state:
        perform_pca()

    # right_t2.write(st.session_state.pca_component_dict)
    right_t2.write("## Automated labeling")
    display_results(right_t2)

    cum_exp = left_t2.slider(
        "Select the cumulative explained variance",
        min_value=0.1,
        max_value=1.0,
        value=app_utilities.DEFAULT_CUM_EXP,
        step=0.05,
        key="cum_exp",
        on_change=perform_pca,
    )

    left_t2.write(f"Number of components: {st.session_state.N}")
    # right_t2.markdown("### PCA results")
    # perform_pca(right_t2)

    # horizontal line
    right_t2.markdown("---")

    right_t2.write("## PCA results")

    expander_pca = right_t2.expander("PCA results")
    expander_pca.write(st.session_state.pca_df)

    expander_exp = right_t2.expander("PCA explained variance")
    expander_exp.write(st.session_state.exp_ratio)

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
