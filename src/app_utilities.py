import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json
import numpy as np
import google.generativeai as genai


def load_data():
    if st.session_state.file is None:
        df = pd.read_csv("./data/data-final-sample.csv")
    else:
        df = pd.read_csv(st.session_state.file)

    update_df(df)


def update_df(df):
    if "ignore_cols" in st.session_state:
        features = [f for f in df.columns if f not in st.session_state.ignore_cols]
    else:
        features = df.columns.tolist()

    features = [f for f in features if f != st.session_state.entity_col]
    st.session_state.features = features

    if st.session_state.entity_col not in df.columns:
        cols = st.session_state.features
    else:
        cols = [st.session_state.entity_col] + st.session_state.features
    st.session_state.df = df[cols].dropna()


def load_map():
    # read json file
    if st.session_state.map is None:
        with open("./data/map.json", "r") as f:
            map = json.load(f)
    else:
        with open(st.session_state.map, "r") as f:
            map = json.load(f)

    update_map(map)


def update_map(map):
    st.session_state.col_mapping = map


def perform_pca(component):
    try:
        x = st.session_state.df.loc[:, st.session_state.features].values
        x = StandardScaler().fit_transform(x)
        # PCA
        pca = PCA(n_components=st.session_state.n_components)
        principalComponents = pca.fit_transform(x)

        principalDf = pd.DataFrame(
            data=principalComponents,
            columns=[
                f"principal component {i}" for i in range(st.session_state.n_components)
            ],
        )

        # let N be the most significant principal components based on cumulative explained variance
        exp = pca.explained_variance_ratio_
        N = 0
        for i in range(st.session_state.n_components):
            if sum(exp[: i + 1]) <= 0.9:
                N = i + 1
        # print(N)
        # print([sum(exp[: i + 1]) for i in range(st.session_state.n_components)])

        # Get the first 5 principal components
        components = pca.components_
        pca_component_dict = {}
        for i in range(N):
            top_5_components = np.argsort(components[i])[-5:][::-1]
            top_5_features = [st.session_state.features[c] for c in top_5_components]
            top_5_features = [
                st.session_state.col_mapping.get(f, f) for f in top_5_features
            ]

            bottom_5_components = np.argsort(components[i])[:5]
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
                "label": label,
                "top": top_5_features,
                "bottom": bottom_5_features,
            }

        st.session_state.pca_component_dict = pca_component_dict
        st.session_state.pca_df = principalDf

    except:
        st.session_state.pca_component_dict = {}
        st.session_state.pca_df = None

    component.write(st.session_state.pca_df)
    component.write(st.session_state.pca_component_dict)


def get_component_labels(text):
    # TODO: debug
    # msgs = {
    #     "system_instruction": "you are a data analysit",
    #     "history": [
    #         {
    #             "role": "user",
    #             "parts": "You make a label from the following texts that come from PCA analysis. Output a single label only.",
    #         },
    #         {"role": "model", "parts": "Sure!"},
    #     ],
    #     "content": {"role": "user", "parts": text},
    # }

    # model = genai.GenerativeModel(
    #     model_name="gemini-1.5-flash",
    #     system_instruction=msgs["system_instruction"],
    # )
    # chat = model.start_chat(history=msgs["history"])
    # response = chat.send_message(content=msgs["content"])

    # return response.candidates[0].content.parts[0].text

    return "text123"
