import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json
import numpy as np
import google.generativeai as genai
from google.generativeai import GenerationConfig
from visualisation_utilities import Visualization, hex_to_rgb, rgb_to_color
import plotly.graph_objects as go


def set_default_data():
    clear_session_state()
    load_data("./data/data-final-sample.csv")
    load_map("./data/map.json")


def clear_session_state(skip=[]):
    for key in st.session_state.keys():
        if key not in skip:
            del st.session_state[key]

    if "entity_col" not in st.session_state:
        st.session_state.entity_col = "Index"
    if "pca_df" not in st.session_state:
        st.session_state.pca_df = None


def load_new_data():
    clear_session_state(skip=["file"])
    load_data()


def load_data(file=None):
    if file is None:
        file = st.session_state.file

    # st.session_state.file = "./data/data-final-sample.csv"
    # print("loading data")
    st.session_state.df_full = pd.read_csv(file)
    # remove current map
    st.session_state.col_mapping = {}

    # if "map" in st.session_state:
    #     del st.session_state["map"]

    update_df()


def update_df(ignore_cols=[]):
    df = st.session_state.df_full.copy()

    # if "ignore_cols" in st.session_state:
    #     features = [f for f in df.columns if f not in st.session_state.ignore_cols]
    # else:
    #     features = df.columns.tolist()
    if "ignore_cols" in st.session_state:
        ignore_cols = st.session_state.ignore_cols

    features = [f for f in df.columns if f not in ignore_cols]
    features = [f for f in features if f != st.session_state.entity_col]
    st.session_state.features = features

    if st.session_state.entity_col not in df.columns:
        cols = st.session_state.features
    else:
        cols = [st.session_state.entity_col] + st.session_state.features
    st.session_state.df_filtered = df[cols].dropna()
    # set the entity column as the index
    if st.session_state.entity_col != "Index":
        st.session_state.df_filtered.set_index(
            st.session_state.entity_col, inplace=True
        )

    # delete cum_exp from session state
    if "cum_exp" in st.session_state:
        del st.session_state["cum_exp"]


def load_map(file=None):

    if file is None:
        file = st.session_state.map
    # st.session_state.map = "./data/map.json"
    # print("loading map")
    if isinstance(file, str):
        with open(file, "r") as f:
            map = json.load(f)
    else:
        map = json.load(st.session_state.map)

    st.session_state.col_mapping = map


DEFAULT_CUM_EXP = 0.3
DEFAULT_SUM_THRESHOLD = 0.6
DEFAULT_MAX_COMPONENTS = 14


def get_defaults():
    return DEFAULT_CUM_EXP, DEFAULT_SUM_THRESHOLD, DEFAULT_MAX_COMPONENTS


def perform_pca(cum_exp=DEFAULT_CUM_EXP, threshold=DEFAULT_SUM_THRESHOLD):

    if st.session_state.features != []:
        x = st.session_state.df_filtered.loc[:, st.session_state.features].values
        x = StandardScaler().fit_transform(x)
        # PCA
        pca = PCA()
        principalComponents = pca.fit_transform(x)

        principalDf = pd.DataFrame(
            data=principalComponents,
            columns=[
                f"principal component {i}" for i in range(principalComponents.shape[-1])
            ],
        )

        # let N be the most significant principal components based on cumulative explained variance
        if "cum_exp" in st.session_state:
            cum_exp = st.session_state.cum_exp

        st.session_state.exp_ratio = pca.explained_variance_ratio_
        st.session_state.N = 1
        for i in range(1, len(st.session_state.exp_ratio)):
            if (
                sum(st.session_state.exp_ratio[: i + 1]) <= cum_exp
                and i < DEFAULT_MAX_COMPONENTS
            ):
                st.session_state.N = i + 1
            else:
                break
        # print([sum(st.session_state.exp_ratio[: i + 1]) for i in range(len(st.session_state.exp_ratio))])
        # print(st.session_state.N)

        pca_component_dict = {}
        components = pca.components_
        # first st.session_state.N columns of components
        st.session_state.components = components[:, : st.session_state.N]
        for i in range(st.session_state.N):

            n = 1
            c2 = components[i] ** 2  # np.abs(components[i])
            while sum(c2[np.argsort(c2)[::-1][:n]]) < threshold:
                n += 1
            # make n even
            if n % 2 != 0:
                n += 1

            top_components = [
                c for c in np.argsort(c2)[::-1][:n] if components[i][c] > 0
            ]
            bottom_components = [
                c for c in np.argsort(c2)[::-1][:n] if components[i][c] < 0
            ]

            # print(f"top: {top_components}")
            # print(f"bottom: {bottom_components}")

            # n = 5
            # top_components = np.argsort(components[i])[::-1][:n]
            top_values = [round(components[i][c], 2) for c in top_components]
            top_features = [st.session_state.features[c] for c in top_components]
            top_features = [
                st.session_state.col_mapping.get(f, f) for f in top_features
            ]

            # n = 5
            # bottom_components = np.argsort(components[i])[:n]
            bottom_values = [round(components[i][c], 2) for c in bottom_components]
            bottom_features = [st.session_state.features[c] for c in bottom_components]
            bottom_features = [
                st.session_state.col_mapping.get(f, f) for f in bottom_features
            ]

            # text = "Features:\n"
            # text += ",\n".join(top_features + bottom_features)

            text = "Top 5 features:\n"
            text += ",\n".join(top_features)
            text += "\n\nBottom 5 features:\n"
            text += ", ".join(bottom_features)

            label = get_component_labels(text)

            pca_component_dict[f"principal component {i}"] = {
                "explained_variance_ratio": round(st.session_state.exp_ratio[i], 2),
                "label": label,
                "top": top_features,
                "values_top": top_values,
                "bottom": bottom_features,
                "values_bottom": bottom_values,
            }

        st.session_state.pca_component_dict = pca_component_dict
        st.session_state.pca_df = principalDf
        # print("pca")
        vis = Visualization(
            st.session_state.pca_df,
            {k: v["label"] for k, v in st.session_state.pca_component_dict.items()},
        )
        st.session_state.fig_base = vis.fig
        st.session_state.df_z_scores = vis.df_z_scores

    else:
        st.session_state.pca_component_dict = {}
        st.session_state.pca_df = None


def get_component_labels(text):

    msgs = {
        "system_instruction": "You are a data analyst ans scientist",
        "history": [
            {
                "role": "user",
                "parts": "Make a label from the following texts that come from PCA analysis. The label should be of the form x vs y, where x is one or more adjectives that describes an entity that has the top features and y is one or more adjectives that describes an entity that has the bottom features. Output a label only.",
                # "parts": "Make a label from the following texts that come from PCA analysis. The label should be of the form 'x vs y', but if that is not possible a single label 'x'. Output the label only.",
            },
            {"role": "model", "parts": "Sure!"},
        ],
        "content": {"role": "user", "parts": text},
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=msgs["system_instruction"],
        generation_config=GenerationConfig(max_output_tokens=50),
    )
    chat = model.start_chat(history=msgs["history"])
    response = chat.send_message(
        content=msgs["content"],
    )

    return response.candidates[0].content.parts[0].text

    # return "text123"


def display_results(component):
    results_dict = st.session_state.pca_component_dict
    for key in results_dict.keys():
        # component.write(f"### {key.capitalize()}: {results_dict[key]['label']}")
        # component.write(
        #     f"Explained variance ratio: {results_dict[key]['explained_variance_ratio']}"
        # )
        # component.write(f"Top features:")
        # for i in range(len(results_dict[key]["top"])):
        #     # indent the text
        #     component.write(
        #         f"- {results_dict[key]['top'][i]}: {results_dict[key]['values_top'][i]}"
        #     )
        # component.write(f"Bottom features:")
        # for i in range(len(results_dict[key]["bottom"])):
        #     component.write(
        #         f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{results_dict[key]['bottom'][i]}: {results_dict[key]['values_bottom'][i]}"
        #     )

        # put description in an expander
        expander = component.expander(
            f"{key.capitalize()}: {results_dict[key]['label']}"
        )
        expander.write(
            f"Explained variance ratio: {results_dict[key]['explained_variance_ratio']}"
        )
        expander.write(f"Top features:")
        for i in range(len(results_dict[key]["top"])):
            # indent the text
            expander.write(
                f"- ({results_dict[key]['values_top'][i]}) {results_dict[key]['top'][i]}"
            )
        expander.write(f"Bottom features:")
        for i in range(len(results_dict[key]["bottom"])):
            expander.write(
                f"- ({results_dict[key]['values_bottom'][i]}) {results_dict[key]['bottom'][i]}"
            )


def add_to_fig():
    # print("updating fig")

    # find the index of the selected entity from st.session_state.df_filtered
    ind = st.session_state.df_filtered.index.tolist().index(
        st.session_state.selected_entity
    )
    print(ind)

    df = st.session_state.df_z_scores.iloc[ind, :].to_frame().T
    # print(df)

    color = st.get_option("theme.primaryColor")
    if color is None:
        color = "#FF4B4B"

    # fig = st.session_state.fig_base

    for col in df.columns.tolist():

        st.session_state.fig_base.update_traces(
            selector={"name": f"{col} selected"}, x=df[col]
        )

    if "fig" in st.session_state:
        del st.session_state["fig"]

    # st.session_state.fig = fig
