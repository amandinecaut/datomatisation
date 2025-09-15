from visualisation_utilities import Visualisation, ClusterVisualisation, ClusterVisualisation3D
from clustering import Cluster
from google.generativeai import GenerationConfig
import google.generativeai as genai
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import json


import app_utilities
from app_utilities import (
    perform_FA,
    perform_clustering,
    update_df,
    load_new_data,
    load_map,
    display_results,
    get_defaults,
    set_default_data,
    add_to_fig,
    update_fig_cluster,
    update_fig_cluster3d,
    display_cluster_color,
    create_QandA
)


from chat import EntityChat
from description import CreateDescription










st.set_page_config(layout="wide")



default_cum_exp, default_sum_threshold, default_max_components, default_num_clusters = get_defaults()

height = 1500  # height of the container



default_values = {
    "u_labels": np.array([]),
    "centroids": None,
    "ind_col_map": None,
    "selected_entity": None,
    "FA_component_dict": {},
    "list_cluster_name": None,
    "list_description_cluster": None,
    "fig_base": go.Figure(),
    "entity_col": "Index",
    "FA_df": pd.DataFrame(),
    "tab1_done": False,
    "tab2_done": False,
    "tab3_done": False,
    "tab4_done": False
}

for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value


# Add and app header
st.title("Automated DatAnalyst Pipeline")

tabs = st.tabs(["Load data", "Analysis tools", "Clustering", "View"])


# Load Data
with tabs[0]:

    left_t1, right_t1 = st.columns([0.3, 0.7])
    # Left pane title

    left_t1 = left_t1.container(height=height, border=0)
    right_t1 = right_t1.container(height=height, border=3)

    # clear data button
    if left_t1.button("Clear data"):
        app_utilities.clear_session_state()


    # run default data
    if left_t1.button("Load demo data"):
        app_utilities.set_default_data()


    # left all box to upload data
    left_t1.markdown("#### Upload data")
    left_t1.file_uploader(
        "Choose a file",
        type=["csv"],
        key="file",
        on_change=load_new_data,
    )

    left_t1.markdown("#### Upload column name mapping")
    left_t1.file_uploader(
        "Choose a file",
        type=["json"],
        key="map",
        on_change=load_map,
    )

    # display the info of the data
    right_t1.markdown("### Data information")

    if "df_full" not in st.session_state:
        right_t1.write("Load data to view information")
    else:
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
            st.session_state.df_full[st.session_state.features]
            .isnull()
            .any(axis=1)
            .sum()
            > 0
        ):
            right_t1.warning("There are rows containing NaN, these will be dropped.")

        expander_nan = right_t1.expander("Rows containing NaN")
        if st.session_state.entity_col == "Index":
            expander_nan.write(
                st.session_state.df_full[
                    st.session_state.df_full[st.session_state.features]
                    .isnull()
                    .any(axis=1)
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

        st.session_state.tab1_done = True
        
    
# "Analysis Tools"    
with tabs[1]:
    if not st.session_state.get("tab1_done", False):
        st.warning("You must load your data first!")
    else:

        left_t2, right_t2 = st.columns([0.3, 0.7])
        left_t2 = left_t2.container(height=height, border=0)
        right_t2 = right_t2.container(height=height, border=3)

        left_t2.markdown("### Select a tool")

        # Choose the analysis
        if left_t2.button("Factor Analysis"):
            left_t2.markdown("### Factor Analysis")

            cum_exp = left_t2.slider(
                "Select the number of components",
                min_value=1,
                max_value=default_max_components,
                value=app_utilities.DEFAULT_CUM_EXP,
                step=1,
                key="cum_exp",
                on_change=perform_FA,
            )

            if left_t2.button("Run Factor Analysis"):
                perform_FA()

            if "df_full" not in st.session_state:
                right_t2.write("Load data to perform Factor Analysis")
            elif len(st.session_state.df_filtered) < 10:
                right_t2.write("Not enough data to perform Factor Analysis")
            elif "N" not in st.session_state:
                right_t2.write("Select a number of factor to perform Factor Analysis")
            elif "N" in st.session_state:
                right_t2.write("## Automated labeling")
                display_results(right_t2)

                left_t2.write(
                    f"Number of components: {st.session_state.N} (max {default_max_components})"
                )

                right_t2.markdown("---")

                right_t2.write("## Factor Analysis results")

                expander_FA = right_t2.expander("Factor Analysis results")
                expander_FA.write(st.session_state.FA_df)

                expander_exp = right_t2.expander("Factors components")
                expander_exp.write(st.session_state.components)

        if left_t2.button("Logistic Regression"):
            right_t2.markdown("---")
            right_t2.write("## Logistic Regression results")
            
            right_t2.markdown("---")
            right_t2.write("## Question and Answer pairs")





        activate = ["Yes", "No"]
        introduction_choice = left_t2.radio("Do you want an introduction?", activate, key="intro_choice")

        # Show text area only if "Yes" is selected
        text = right_t2.text_area("Enter your introduction here:") if introduction_choice == "Yes" else None

        # Disable button if "Yes" is selected but text is empty
        generate_disabled = introduction_choice == "Yes" and (not text or not text.strip())

        if right_t2.button("Generate Q&A", disabled=generate_disabled):
            QandA = create_QandA(text)

            if QandA and "User" in QandA and "Assistant" in QandA:
            # Display Q&A
                for i, (q, a) in enumerate(zip(QandA['User'], QandA['Assistant']), start=1):
                    right_t2.markdown(f"### **Question {i}:** {q}")
                    right_t2.markdown(f"**Answer:** {a}")
                    right_t2.write("\n")

            # Save Q&A as CSV
            QandA_df = pd.DataFrame(QandA)
            csv_path = './data/describe/QandA_data.csv'
            QandA_df.to_csv(csv_path, index=False)

            st.success("Q&A generated and saved!")
            st.session_state.tab2_done = True
        else:
            st.error("Failed to generate Q&A. Please check your input.")

     

            
        
# Clustering
with tabs[2]:
    if not st.session_state.get("tab2_done", False):
        st.warning("You must complete the factor analysis first!")
        st.stop()
        
    else:
        # Create left and right containers
        left_t3, right_t3 = st.columns([0.3, 0.7])
        left_t3 = left_t3.container(height=height, border=0)
        right_t3 = right_t3.container(height=height, border=3)

        left_t3.markdown("### Clustering")

        # Slider for number of clusters
        num_clusters = left_t3.slider(
            "Select the number of clusters",
            min_value=2,
            max_value=10,
            value= app_utilities.DEFAULT_NUM_CLUSTERS, 
            step=1,
            key="num_clusters",
            on_change=perform_clustering,  
        )
    
        # Button to trigger clustering
        if left_t3.button("Run Clustering"):
            perform_clustering()
            right_t3.write("Clustering complete")
            
   
            
            

        # Factor selection for dimensions
        left_t3.markdown("### Select Factors for Each Dimension")
        factors = [v["label"] for k, v in st.session_state.FA_component_dict.items()]

        if len(factors) <2: 
            right_t3.write("Perform Factor Analysis with at least 2 components to view clustering results")

        elif len(factors) >= 3:
            ## HERE FOR 3D PLOT. THE USER CAN PICK 2D or 3D PLOT
            
            plot_type = ["2D", "3D"]
            plot_choice = left_t3.radio("Select plot type", plot_type, key="plot_choice")

            if plot_choice == "2D":

                # First dimension selection
                dimension_x = left_t3.selectbox(
                    "Select a factor for X-axis:", 
                    factors, 
                    key="dim_x",
                    on_change=update_fig_cluster
                    )
                st.session_state.dimension_x = dimension_x

                # Second dimension selection (excluding first)
                available_for_y = [f for f in factors if f != dimension_x]
                dimension_y = left_t3.selectbox(
                    "Select a factor for Y-axis:", 
                    available_for_y, 
                    key="dim_y",
                    on_change=update_fig_cluster
                    )
                st.session_state.dimension_y = dimension_y

                left_t3.write(f"You selected **{dimension_x}** for the X-axis and **{dimension_y}** for the Y-axis.")
                
                vis_cluster = ClusterVisualisation(
                    st.session_state.FA_df,
                    {k: v["label"] for k, v in st.session_state.FA_component_dict.items()}, 
                    st.session_state.u_labels, 
                    st.session_state.centroids, 
                    st.session_state.ind_col_map
                    )
                st.session_state.fig_cluster = vis_cluster.fig

                right_t3.plotly_chart(
                st.session_state.fig_cluster, use_container_width=True, theme="streamlit"
                )
                st.session_state.tab3_done = True

            else:
                # First dimension selection
                dimension_x = left_t3.selectbox(
                    "Select a factor for X-axis:", 
                    factors, 
                    key="dim_x",
                    on_change=update_fig_cluster3d
                    )
                st.session_state.dimension_x = dimension_x

                # Second dimension selection (excluding first)
                available_for_y = [f for f in factors if f != dimension_x]
                dimension_y = left_t3.selectbox(
                    "Select a factor for Y-axis:", 
                    available_for_y, 
                    key="dim_y",
                    on_change=update_fig_cluster3d
                    )
                st.session_state.dimension_y = dimension_y

                # Third dimension selection (excluding first and second)
                dimension_z = left_t3.selectbox(
                    "Select a factor for Z-axis:", 
                    [f for f in factors if f not in [dimension_x, dimension_y]], 
                    key="dim_z",
                    on_change=update_fig_cluster3d
                    )
                st.session_state.dimension_z = dimension_z

                left_t3.write(f"You selected **{dimension_x}** for the X-axis, **{dimension_y}** for the Y-axis, and **{dimension_z}** for the Z-axis.")

                vis_cluster = ClusterVisualisation3D(
                    st.session_state.FA_df,
                    {k: v["label"] for k, v in st.session_state.FA_component_dict.items()}, 
                    st.session_state.u_labels, 
                    st.session_state.centroids, 
                    st.session_state.ind_col_map
                    )
                st.session_state.fig_cluster3d = vis_cluster.fig

                right_t3.plotly_chart(
                    st.session_state.fig_cluster3d, use_container_width=True, theme="streamlit"
                )
                st.session_state.tab3_done = True



        else: 
            ### HERE FOR 2D PLOT ONLY WHEN THERE IS ONLY 2 FACTORS

            # First dimension selection 
            dimension_x = left_t3.selectbox(
                "Select a factor for X-axis:", 
                factors, 
                key="dim_x",
                on_change=update_fig_cluster
            )
            st.session_state.dimension_x = dimension_x

            # Second dimension selection (excluding first)
            available_for_y = [f for f in factors if f != dimension_x]
            dimension_y = left_t3.selectbox(
                "Select a factor for Y-axis:", 
                available_for_y, 
                key="dim_y",
                on_change=update_fig_cluster
            )
            st.session_state.dimension_y = dimension_y
      
            left_t3.write(f"You selected **{dimension_x}** for the X-axis and **{dimension_y}** for the Y-axis.")

            if left_t3.button("Run Visualisation"):
                vis_cluster = ClusterVisualisation(
                    st.session_state.FA_df,
                    {k: v["label"] for k, v in st.session_state.FA_component_dict.items()}, 
                    st.session_state.u_labels, 
                    st.session_state.centroids, 
                    st.session_state.ind_col_map
                    )
                st.session_state.fig_cluster = vis_cluster.fig

                right_t3.plotly_chart(
                st.session_state.fig_cluster, use_container_width=True, theme="streamlit"
                )

        
        with right_t3:
            # Cluster description
            st.markdown("<h3><b>Description of each cluster</b></h3>", unsafe_allow_html=True)
            list_cluster_name = st.session_state.list_cluster_name
            list_color_cluster = st.session_state.ind_col_map
            list_description_cluster = st.session_state.list_description_cluster
            if list_color_cluster is None:
                pass
            else:
                for i in list_color_cluster:
                    display_cluster_color(list_cluster_name[i], list_color_cluster[i])
                    st.write(list_description_cluster[i])
                
           


# View
with tabs[3]:
    if not st.session_state.get("tab3_done", False):
        st.warning("You must complete the clustering first!")
    else:
        left_t4, right_t4 = st.columns([0.3, 0.7])
        left_t4 = left_t4.container(height=height, border=0)
        right_t4 = right_t4.container(height=height, border=3)
        left_t4.markdown("### Select entity")

        # drop down with entity column, default to first column
        entity = left_t4.selectbox(
            label="Select entity",
            options=(st.session_state.df_filtered.index.to_list()),
            key="selected_entity",
            index=0,
            on_change=add_to_fig,
        )

        right_t4.plotly_chart(
            st.session_state.fig_base, use_container_width=True, theme="streamlit"
        )
        
        

        with right_t4:
            if st.session_state.selected_entity == None:
                indice = 0
            else:
                indice = st.session_state.df_filtered.index.tolist().index(
                    st.session_state.selected_entity
                )
        
            st.write("Wordalisation")
            chat = EntityChat()
            if chat.state == "empty":
                chat.add_message(
                "Please can you summarise the data for me?",
                role="user",
                user_only=False,
                visible=False,
                )
                description =  CreateDescription()
                summary = description.stream_gpt(indice)
                st.session_state.entity_description = summary
                chat.add_message(summary)
                chat.state = "default"
            chat.get_input()
            chat.display_messages()
            chat.save_state()
            
        st.session_state.tab4_done = True 


        #indice = st.session_state.df_filtered.index.tolist().index(st.session_state.selected_entity_tab5)

        #st.write("# Entity description")
        #clust = entity_description_cluster()
            
        #st.write(clust)




# debug
# print()
# print(st.session_state)
# print("foo")
# for key, value in st.session_state.items():
#     print(key)  # , value)

# print("\t run through")
# for key, value in st.session_state.items():
#     print("\t" + key)  # , value)
