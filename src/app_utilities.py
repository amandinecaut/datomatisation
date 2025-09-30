from visualisation_utilities import (
    Visualisation,
    hex_to_rgb,
    rgb_to_color,
    ClusterVisualisation,
    ClusterVisualisation3D,
)
from description import ModelHandler, CreateDescription

from clustering import Cluster
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from google.generativeai import GenerationConfig
import google.generativeai as genai
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import openai
import json
import ast
import re
import os


DEFAULT_CUM_EXP = 3
DEFAULT_SUM_THRESHOLD = 0.6
DEFAULT_MAX_COMPONENTS = 14
DEFAULT_NUM_CLUSTERS = 5

default_values = {
    "df": pd.DataFrame(), 
    "u_labels": np.array([]),
    "centroids": None,
    "ind_col_map": None,
    "selected_entity": None,
    "FA_component_dict": {},
    "list_cluster_name": None,
    "list_description_cluster": None,
    "fig_base": go.Figure(),
    "entity_col": "Index",
    "tab1_done": False,
    "tab2_done": False,
    "tab3_done": False,
    "tab4_done": False,
    "data_loading": False,
    "indice": 0,
}




### ---- Load data tab utilities ---- ###

def set_default_data():
    clear_session_state(skip=["file", "map"])
    load_data("./data/data-sample.csv")
    load_map("./data/map.xlsx")

def clear_session_state(skip=[]):
    for key in st.session_state.keys():
        if key not in skip:
            del st.session_state[key]

    for key, value in default_values.items():
        if key not in st.session_state:
            if key not in skip:
                st.session_state[key] = value

def load_new_data():
    clear_session_state(skip=["file", "map"])
    load_data()

def detect_delimiter(first_line):
    # Common delimiters to try
    delimiters = [",", ";", "\t", "|"]

    # Try each delimiter and count how many columns we get
    best_delimiter = None
    max_columns = 0

    for delimiter in delimiters:
        # Try to split the first line with the delimiter
        columns = first_line.split(delimiter)
        if len(columns) > max_columns:
            max_columns = len(columns)
            best_delimiter = delimiter

    return best_delimiter

def load_data(file=None):

    st.session_state.data_loading = True

    if file is None:
        file = st.session_state["file"]

    delimiter = None
    if isinstance(file, st.runtime.uploaded_file_manager.UploadedFile):
        for line in file:
            line = line.decode("utf-8")
            delimiter = detect_delimiter(line)
            break
        file.seek(0)

    st.session_state.df_full = pd.read_csv(file, sep=delimiter, engine="python")
    # remove current map
    st.session_state.col_mapping = {}

    # if "map" in st.session_state:
    #     del st.session_state["map"]

    update_df()

    st.session_state.data_loading = False

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

    if "num_clusters" in st.session_state:
        del st.session_state["num_clusters"]

def load_map(file=None):

    if file is None:
        file = st.session_state.map

    if isinstance(file, str):
        file_extension = os.path.splitext(file)[1].lower()
        if file_extension == ".json":
            with open(file, "r") as f:
                map = json.load(f)
        elif file_extension in [".xlsx", ".xls"]:
            df = pd.read_excel(file)
            map = dict(zip(df["Key"], df["Value"]))
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    else:
        file_extension = os.path.splitext(file.name)[1].lower()
        if file_extension == ".json":
            map = json.load(file)
        elif file_extension in [".xlsx", ".xls"]:
            df = pd.read_excel(file)
            print(df)
            
            map = dict(zip(df["Key"], df["Value"]))
            print(map)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    st.session_state.col_mapping = map
    
def get_defaults():
    return (
        DEFAULT_CUM_EXP,
        DEFAULT_SUM_THRESHOLD,
        DEFAULT_MAX_COMPONENTS,
        DEFAULT_NUM_CLUSTERS,
    )


### ----  Analysis tab utilities ---- ###

# Factor Analysis utilities
def perform_FA(cum_exp=DEFAULT_CUM_EXP, threshold=DEFAULT_SUM_THRESHOLD):

    if st.session_state.features != []:
        x = st.session_state.df_filtered.loc[:, st.session_state.features].values
        x = StandardScaler().fit_transform(x)
        if "cum_exp" in st.session_state:
            components = st.session_state.cum_exp
        else:
            components = DEFAULT_CUM_EXP

        # Factor Analysis
        FA = FactorAnalysis(n_components=components)
        principalComponents = FA.fit_transform(x)

        principalDf = pd.DataFrame(
            data=principalComponents,
            columns=[
                f"Principal component {i}" for i in range(principalComponents.shape[-1])
            ],
        )

        # st.session_state.exp_ratio = PCA.explained_variance_ratio_ ## This is only for PCA

        st.session_state.N = components

        FA_component_dict = {}
        components = FA.components_

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

            # top_components = [
            #    c for c in np.argsort(c2)[::-1][:n] if components[i][c] > 0
            # ]
            # bottom_components = [
            #    c for c in np.argsort(c2)[::-1][:n] if components[i][c] < 0
            # ]

            top_components = np.argsort(components[i])[::-1][:n]
            bottom_components = np.argsort(components[i])[:n]

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

            text = "Bottom 5 features:\n"
            text += ", ".join(bottom_features)
            text += "\n\nTop 5 features:\n"
            text += ", ".join(top_features)

            label = get_component_labels(text)

            FA_component_dict[f"Principal component {i}"] = {
                "label": label,
                "top": top_features,
                "values_top": top_values,
                "bottom": bottom_features,
                "values_bottom": bottom_values,
            }

        st.session_state.FA_component_dict = FA_component_dict
        st.session_state.df = principalDf

        vis = Visualisation(
            st.session_state.df,
            {k: v["label"] for k, v in st.session_state.FA_component_dict.items()},
        )
        st.session_state.fig_base = vis.fig
        st.session_state.df_z_scores = vis.df_z_scores

    else:
        st.session_state.FA_component_dict = {}
        st.session_state.df = None

def get_component_labels(text):
    MH = ModelHandler()
    msgs = {
        "system_instruction": "You are a data analyst and scientist",
        "history": [
            {
                "role": "user",
                "parts": (
                    "Make a label from the following texts that come from factor analysis."
                    "The label must strictly follow the format: 'bottom features vs top features'. "
                    "The label should be of the form x vs y, where x is one or more adjectives that describes an entity that has the bottom features and y is one or more adjectives that describes an entity that has the top features."
                    "The label should not have connotation negative."
                    "Output a label only."
                ),
            },
            {"role": "model", "parts": "Sure!"},
        ],
        "content": {"role": "user", "parts": text},
    }

    text_generate = MH.get_generate(msgs, max_output_token = 10)

    return text_generate

def display_results(component):
    results_dict = st.session_state.FA_component_dict
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
        # expander.write(
        #    f"Explained variance ratio: {results_dict[key]['explained_variance_ratio']}"
        # )
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


### ---- Clustering tab utilities ---- ###

# Cluster utilities
def perform_clustering(num_clusters=DEFAULT_NUM_CLUSTERS):
    if "num_clusters" in st.session_state:
        num_clusters = st.session_state.num_clusters
    else:
        num_clusters = DEFAULT_NUM_CLUSTERS
    
    # Reinitialise session state for cluster on rerun
    #for key in ["u_labels", "centroids", "ind_col_map", 
    #        "list_cluster_name", "list_description_cluster"]:
    #    st.session_state.pop(key, None)
   
    if "Cluster" in st.session_state.df.columns:
        st.session_state.df.drop(columns=["Cluster"], inplace=True)

    cluster = Cluster(
        st.session_state.df,
        {k: v["label"] for k, v in st.session_state.FA_component_dict.items()},
        num_clusters,
    )
    (
        st.session_state.u_labels,
        st.session_state.centroids,
        st.session_state.ind_col_map,
    ) = (cluster.u_labels, cluster.centroids, cluster.ind_col_map)
    st.session_state.df = cluster.df

# Cluster visualisation utilities
def update_fig_cluster():

    if "fig_cluster" in st.session_state:
        del st.session_state["fig_cluster"]

    vis_cluster = ClusterVisualisation(
        st.session_state.df,
        {k: v["label"] for k, v in st.session_state.FA_component_dict.items()},
        st.session_state.u_labels,
        st.session_state.centroids,
        st.session_state.ind_col_map,
    )
    st.session_state.fig_cluster = vis_cluster.fig

def update_fig_cluster3d():
    if "fig_cluster3d" in st.session_state:
        del st.session_state["fig_cluster3d"]

    vis_cluster = ClusterVisualisation3D(
        st.session_state.df,
        {k: v["label"] for k, v in st.session_state.FA_component_dict.items()},
        st.session_state.u_labels,
        st.session_state.centroids,
        st.session_state.ind_col_map,
    )
    st.session_state.fig_cluster3d = vis_cluster.fig

def display_cluster_color(cluster_name, color, size=40):
    square_html = f"""
    <div style="
        display: flex; 
        align-items: center; 
        gap: 10px;">
        <div style="
            width: {size}px; 
            height: {size}px; 
            background-color: {color}; 
            border-radius: 10px;">
        </div>
        <span style="font-size: 18px;">{cluster_name}</span>
    </div>
    """
    st.markdown(square_html, unsafe_allow_html=True)

# Q&A utility
def create_QandA(text: str | None):
    """
    Creates a dictionary of question and answer pairs based on component analysis 
    and optional additional text.
    """
    MH = ModelHandler()
    desc = CreateDescription()
    # --- Generate Q&A for the component analysis ---
    component_text = [
        part 
        for details in st.session_state.FA_component_dict.values() 
        for part in desc.split_qualities(details["label"])
        ]
           
    msgs = {
        "system_instruction": "You are a data analyst",
        "history": [
            {
                "role": "user",
                "parts": (
                    "You have a list of each component deduced from factor analysis."
                    "Deduce question and answer pairs, such that: the questions should be about each component, and the answers should explain them. "
                    "The question and answer are deduce from the factor analysis"
                    "The questions should be simple and the answers should be easy to understand."
                    "Make a dataframe with two columns: one column is 'User' for the question, one column is 'Assistant. for the answers"
                    "Provide just the data dictionary from the code snippet, excluding imports and the DataFrame creation, without the rest of the Python script."
                ),
            }
        ],
        "content": {"role": "user", "parts": component_text},
    }
    
    QandA = MH.get_generate(msgs, max_output_token = 200)
    QandA = clean_QandA(QandA)
    
    # --- Generate Q&A for additional information if provided ---
    if text is not None:
        msgs = {
        "system_instruction": "You are a data analyst and scientist",
        "history": [
            {
                "role": "user",
                "parts": (
                    "You have information about the data and more context "
                    "Deduce question and answer pairs from this text. "
                    "Make a dataframe with two columns: one column is 'User' for the question, one column is 'Assistant. for the answers"
                    "Provide just the data dictionary from the code snippet, excluding imports and the DataFrame creation, without the rest of the Python script."
                ),
            }
        ],
        "content": {"role": "user", "parts":  text },
    }
        
        QandA2 = MH.get_generate(msgs, max_output_token = 200)
        QandA2 = clean_QandA(QandA2)
        
        # Concatenate the two dictionaries
        QandA['User'].extend(QandA2['User'])
        QandA['Assistant'].extend(QandA2['Assistant'])

    return QandA

def clean_QandA(QandA):
    QandA = (
        QandA.replace("data = ", "").replace("python", "").replace("```", "").strip()
    )
    
    if "=" in QandA:
        QandA = QandA.split("=", 1)[1].strip()

    match = re.search(r"\{.*\}", QandA, re.DOTALL)
    if match:
        QandA = match.group(0)
        
    return ast.literal_eval(QandA)


### ---- View tab utilities ---- ###

# View utilities
def add_to_fig():
    # print("updating fig")

    # find the index of the selected entity from st.session_state.df_filtered
    ind = st.session_state.indice

    df = st.session_state.df_z_scores.iloc[ind, :].to_frame().T

    color = st.get_option("theme.primaryColor")
    if color is None:
        color = "#FF4B4B"

    for col in df.columns.tolist():
        st.session_state.fig_base.update_traces(
            selector={"name": f"{col} selected"}, x=df[col]
        )

    if "fig" in st.session_state:
        del st.session_state["fig"]

# Chat utility
def create_chat(to_hash, chat_class,*args, **kwargs):
    chat_hash_state = hash(to_hash)
    chat = chat_class(chat_hash_state, *args, **kwargs)
    return chat