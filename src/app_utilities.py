from visualisation_utilities import (
    Visualisation,
    hex_to_rgb,
    rgb_to_color,
    ClusterVisualisation,
    ClusterVisualisation3D,
    DistributionPlot
)
from wordalisation import ModelHandler, ClusterWordalisation, FALabel, QandAWordalisation,QandAWordalisation_from_text

from clustering import Cluster
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from google.generativeai import GenerationConfig
import google.generativeai as genai
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from kneed import KneeLocator
from scipy.stats import zscore
import streamlit as st
import pandas as pd
import numpy as np
import itertools
import openai
import json
import ast
import re
import os


DEFAULT_CUM_EXP = 3
DEFAULT_THRESHOLD = 0.2
DEFAULT_MAX_COMPONENTS = 7
DEFAULT_NUM_CLUSTERS = 4

default_values = {
    "df": pd.DataFrame(), 
    "u_labels": np.array([]),
    "centroids": None,
    "ind_col_map": None,
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
    "selected_entity" : None,
    "entity_id" : 'entity',
    
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
            map = dict(zip(df["Key"], df["Value"]))
            
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    st.session_state.col_mapping = map
    
def get_defaults():
    return (
        DEFAULT_CUM_EXP,
        DEFAULT_THRESHOLD,
        DEFAULT_MAX_COMPONENTS,
        DEFAULT_NUM_CLUSTERS,
    )

def choose_article(word):
    """
    Return 'a' or 'an' depending on whether the word starts with a vowel sound.
    """
    vowels = ("a", "e", "i", "o", "u")
    w = word.lower().strip()

    # Common special cases with silent 'h' or consonant-sounding vowels
    an_exceptions = ("honest", "honor", "hour", "heir")
    a_exceptions = ("university", "unicorn", "european", "one", "once")

    if w.startswith(an_exceptions):
        return "an"
    if w.startswith(a_exceptions):
        return "a"

    # Default rule
    return "an" if w.startswith(vowels) else "a"

### ----  Analysis tab utilities ---- ###

# Factor Analysis utilities
def perform_FA(cum_exp=DEFAULT_CUM_EXP, threshold=DEFAULT_THRESHOLD):

    if st.session_state.features != []:
        x = st.session_state.df_filtered.loc[:, st.session_state.features].values
        original_index = st.session_state.df_filtered.index

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
            columns=[f"Factor {i+1}" for i in range(principalComponents.shape[-1])],
            index=original_index,
        )
     

        # st.session_state.exp_ratio = PCA.explained_variance_ratio_ ## This is only for PCA

        st.session_state.N = components

        FA_component_dict = {}
        components = FA.components_

        # first st.session_state.N columns of components
        st.session_state.components = components[:, : st.session_state.N]

        for i in range(st.session_state.N):

            # n = 1
            # c2 = components[i] ** 2  # np.abs(components[i])
            # while sum(c2[np.argsort(c2)[::-1][:n]]) < threshold:
            #     n += 1
            # # make n even
            # if n % 2 != 0:
            #     n += 1

            # top_components = [
            #    c for c in np.argsort(c2)[::-1][:n] if components[i][c] > 0
            # ]
            # bottom_components = [
            #    c for c in np.argsort(c2)[::-1][:n] if components[i][c] < 0
            # ]

            #n = 2 # this is for the top 2 or bottom 2
            #top_components = np.argsort(components[i])[::-1][:n]
            #bottom_components = np.argsort(components[i])[:n]

           
            top = np.where(components[i] > threshold)[0]
            bottom = np.where(components[i]< -threshold)[0]
            top_components = top[np.argsort(components[i][top])[::-1]]
            bottom_components = bottom[np.argsort(components[i][bottom])]

            # Keep only top 5
            n = 5
            if len(top_components) > n:
                top_components = top_components[:n]


            if len(bottom_components) > n:
                bottom_components = bottom_components[:n]


            #print(f"top: {top_components}")
            #print(f"bottom: {bottom_components}")

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

            text = "Bottom features:\n"
            text += ", ".join(bottom_features)
            text += "\n\nTop features:\n"
            text += ", ".join(top_features)


            FA_component_dict[f"Factor {i+1}"] = {
                #"label": label,
                "top": top_features,
                "values_top": top_values,
                "bottom": bottom_features,
                "values_bottom": bottom_values,
            }

           
        get_component_labels(FA_component_dict)
        st.session_state.FA_component_dict = FA_component_dict
        

        
        st.session_state.df = principalDf.apply(zscore, nan_policy="omit")
    
        #st.session_state.df = principalDf


        vis = DistributionPlot(
            st.session_state.df,
            {k: v["label"] for k, v in st.session_state.FA_component_dict.items()},
        )
        st.session_state.fig_base = vis.fig
        st.session_state.df_z_scores = vis.df_z_scores

    else:
        st.session_state.FA_component_dict = {}
        st.session_state.df = None

def get_component_labels(FA_component_dict):
    FALabeler = FALabel()
    dict_label = []
    list_FA_labels = []

    for key, details in FA_component_dict.items():
        FALabeler.existing_labels(list_FA_labels)
        FALabeler.tell_it_what_data_to_use(details)
        FALabeler.messages = FALabeler.setup_messages()
        label = FALabeler.stream_gpt().lower()

        list_FA_labels.append(label)
        
        # update the dict directly
        FA_component_dict[key]["label"] = label
        dict_label.append({"User": f"What does the factor '{label}' mean?", "Assistant": f"{FALabeler.tell_it_what_data_to_use(details)}" })
    
    path = "./data/describe/generate/tell_it_what_it_knows.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_label = pd.DataFrame.from_dict(dict_label)
    df_label.to_csv(path, index=False)

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


# Q&A utility
def create_QandA(text: str | None):
    dictionary = st.session_state.FA_component_dict
    entity = st.session_state.entity_id
    article = choose_article(entity)
    QandA = QandAWordalisation()
    list_QandA = []
    for key, details in  dictionary.items():
        key1, key2 = split_qualities(dictionary[key]["label"])
        for factor in (key1, key2):
            question = f"What does it mean when {article} {entity} is described as {factor}?"
            QandA.tell_it_what_data_to_use(question)
            QandA.messages = QandA.setup_messages()
            answer = QandA.stream_gpt()
            list_QandA.append({"Question": question, "Answer": answer})

    
    if text is not None:
     
        QandA_text = QandAWordalisation_from_text()
        QandA_text.tell_it_what_data_to_use(text)
        QandA_text.messages = QandA_text.setup_messages()
        sublist = QandA_text.stream_gpt()
        print(sublist)
        cleaned_list =  clean_qanda_list_text(sublist)
        print(cleaned_list)
        list_QandA.append(cleaned_list)

    return qa_to_dataframe(list_QandA)

def split_qualities(text):
    # Use a regular expression to split on " vs " (case-insensitive)
    parts = re.split(r"\s+vs\.?\s+", text, flags=re.IGNORECASE)
    text1, text2 = parts[0].strip(), parts[1].strip()

    return text1, text2

def clean_qanda_list_text(text_or_list):
    # If input is a list, join all items into one big text block
    if isinstance(text_or_list, list):
        text = "\n".join(text_or_list)
    else:
        text = text_or_list
    text = re.sub(r'---+', '', text)

    qa_pairs = []

    # Regex pattern to capture both bold and plain question/answer formats
    pattern = r'\*\*Question\s*\d*\*\*:\s*(.*?)\s*\*\*Answer\s*\d*\*\*:\s*(.*?)(?=(\*\*Question|\Z))'

    matches = re.findall(pattern, text, flags=re.DOTALL)

    for q, a in matches:
        qa_pairs.append({
            "Question": q.strip(),
            "Answer": a.strip()
        })

    return qa_pairs


def qa_to_dataframe(qa_list):
    """
    Convert a list of question-answer pairs into a pandas DataFrame 
    with columns 'user' (questions) and 'assistant' (answers).

    Expected input format:
    [
        {"Question": "text", "Answer": "text"},
        ...
    ]
    """
    rows = []
    
    #for item in itertools.chain.from_iterable(qa_list):
    for item in qa_list:
        # Normalise keys to handle slight variations like "Anwer"
        question = item.get("Question") or item.get("question")
        answer   = item.get("Answer") or item.get("Anwer") or item.get("answer")
        
        rows.append({"User": question, "Assistant": answer})
    
    return pd.DataFrame(rows)
  
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

### ---- Clustering tab utilities ---- ###

# Find optimal number of clusters
def find_optimal_k_elbow(X, k_min=1, k_max=10, random_state=42):
    """
    Determines the optimal number of clusters (k) using the Elbow Method.
    
       Parameters
    ----------
     X : array-like, shape (n_samples, n_features)
        The input data for clustering.
    k_min : int, optional (default=1)
        The minimum number of clusters to test.
    k_max : int, optional (default=10)
        The maximum number of clusters to test.
    random_state : int, optional (default=42)
        Random state for reproducibility.

    Returns
    -------
    optimal_k : int
        The estimated optimal number of clusters.
    """

    inertias = []
    Ks = range(k_min, k_max + 1)

    # Compute KMeans inertia for each k
    for k in Ks:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    # Find the "elbow" point
    kl = KneeLocator(Ks, inertias, curve="convex", direction="decreasing")
    optimal_k = kl.elbow

    return optimal_k

# perfom clustering
def perform_clustering(num_clusters = DEFAULT_NUM_CLUSTERS):
    num_clusters = st.session_state.get("num_clusters", DEFAULT_NUM_CLUSTERS)
    
    #if "Cluster" in st.session_state.df.columns:
    #    st.session_state.df.drop(columns=["Cluster"], inplace=True)

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


### ---- View tab utilities ---- ###

# View utilities
def add_to_fig():

    ind = st.session_state.indice
    #df = st.session_state.df_z_scores.iloc[ind, :].to_frame().T
    df = st.session_state.df.iloc[ind, :].to_frame().T
    
    color = st.get_option("theme.primaryColor")
    if color is None:
        color = "#FF4B4B"

    for col in df.columns.tolist():
        st.session_state.fig_base.update_traces(
            selector={"name": f"{col} selected"}, x=df[col]
        )

    st.session_state.fig_base.update_traces(
        selector={"uid" : "dummy_legend_name"}, 
        name = f"{st.session_state.selected_entity}"
        )



    if "fig" in st.session_state:
        del st.session_state["fig"]

# Chat utility
def create_chat(to_hash, chat_class,*args, **kwargs):
    chat_hash_state = hash(to_hash)
    chat = chat_class(chat_hash_state, *args, **kwargs)
    return chat