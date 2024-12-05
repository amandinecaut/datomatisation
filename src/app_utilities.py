import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json
import numpy as np
import google.generativeai as genai


def load_data():
    print("loading data")
    if "file" not in st.session_state:
        print("default data")
        st.session_state.df_full = pd.read_csv("./data/data-final-sample.csv")
        # df = pd.DataFrame()
    else:
        print(st.session_state.file)
        st.session_state.df_full = pd.read_csv(st.session_state.file)
        # remove current map
        st.session_state.col_mapping = {}
        # delete st.session_state["map"]
        del st.session_state["map"]

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


def load_map():
    # read json file
    print("loading map")
    if "map" not in st.session_state:
        print("default map")
        with open("./data/map.json", "r") as f:
            map = json.load(f)
    else:
        print(st.session_state.map.name)
        map = json.load(st.session_state.map)

    st.session_state.col_mapping = map


DEFAULT_CUM_EXP = 0.3


def perform_pca(cum_exp=DEFAULT_CUM_EXP):
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
            if sum(st.session_state.exp_ratio[: i + 1]) <= cum_exp:
                st.session_state.N = i + 1
            else:
                break
        # print([sum(st.session_state.exp_ratio[: i + 1]) for i in range(len(st.session_state.exp_ratio))])
        # print(st.session_state.N)

        # Get the first 5 principal components
        components = pca.components_
        pca_component_dict = {}
        for i in range(st.session_state.N):
            top_5_components = np.argsort(components[i])[-5:][::-1]
            top_5_values = [round(components[i][c], 2) for c in top_5_components]
            top_5_features = [st.session_state.features[c] for c in top_5_components]
            top_5_features = [
                st.session_state.col_mapping.get(f, f) for f in top_5_features
            ]

            bottom_5_components = np.argsort(components[i])[:5]
            bottom_5_values = [round(components[i][c], 2) for c in bottom_5_components]
            bottom_5_features = [
                st.session_state.features[c] for c in bottom_5_components
            ]
            bottom_5_features = [
                st.session_state.col_mapping.get(f, f) for f in bottom_5_features
            ]

            text = "Top 5 features:\n"
            text += ",\n".join(top_5_features)
            text += "\n\nBottom 5 features:\n"
            text += ", ".join(bottom_5_features)

            label = get_component_labels(text)

            pca_component_dict[f"principal component {i}"] = {
                "explained_variance_ratio": round(st.session_state.exp_ratio[i], 2),
                "label": label,
                "top": top_5_features,
                "values_top": top_5_values,
                "bottom": bottom_5_features,
                "values_bottom": bottom_5_values,
            }

        st.session_state.pca_component_dict = pca_component_dict
        st.session_state.pca_df = principalDf

    else:
        st.session_state.pca_component_dict = {}
        st.session_state.pca_df = None


def get_component_labels(text):
    # TODO: debug
    msgs = {
        "system_instruction": "you are a data analyst",
        "history": [
            {
                "role": "user",
                "parts": "You make a label from the following texts that come from PCA analysis. The label should be of the form x vs y where x describes an entity with the top features and y describes an entity with the bottom features. Output a single label only.",
            },
            {"role": "model", "parts": "Sure!"},
        ],
        "content": {"role": "user", "parts": text},
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=msgs["system_instruction"],
    )
    chat = model.start_chat(history=msgs["history"])
    response = chat.send_message(content=msgs["content"])

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
                f"- {results_dict[key]['top'][i]}: {results_dict[key]['values_top'][i]}"
            )
        expander.write(f"Bottom features:")
        for i in range(len(results_dict[key]["bottom"])):
            expander.write(
                f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{results_dict[key]['bottom'][i]}: {results_dict[key]['values_bottom'][i]}"
            )
