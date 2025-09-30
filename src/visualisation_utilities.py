import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import zscore
import streamlit as st


def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)

def rgb_to_color(rgb_color: tuple, opacity=1):
    return f"rgba{(*rgb_color, opacity)}"



class Visualisation:
    def __init__(self, df_FA, FA_label_map):

        # print("creating visualization")
        cols = [k for k in FA_label_map.keys()]
        self.df_FA = df_FA[cols]
        self.FA_label_map = FA_label_map

        self.df_z_scores = self.get_z_scores()
        # map self.df_z_scores using FA_label_map
        self.df_z_scores.rename(columns=self.FA_label_map, inplace=True)

        self.fig = go.Figure()
        self.set_visualization()

    def get_z_scores(self):
        return self.df_FA.apply(zscore, nan_policy="omit")

    def set_visualization(self):
        # streamlit primary color
        color = st.get_option("theme.primaryColor")
        if color is None:
            color = "#FF4B4B"

        sample_size = 100
        if len(self.df_z_scores) > sample_size:
            df = self.df_z_scores.sample(sample_size)
        else:
            df = self.df_z_scores

        # Ensure correct order by using FA_label_map
        df = df[list(self.FA_label_map.values())]
        df_entity = self.df_z_scores.iloc[0, :].to_frame().T[list(self.FA_label_map.values())]
        
        # add a scatter plot for each principal component
        for i, col in enumerate(df.columns):
            

            self.fig.add_trace(
                go.Scatter(
                    x=df[col],
                    y=np.ones(len(df)) * i,
                    mode="markers",
                    marker={
                        "color": rgb_to_color(hex_to_rgb(color), opacity=0.2),
                        "size": 10,
                    },
                    showlegend=False,
                )
            )
            self.fig.add_annotation(
                x=0,
                y=i - 0.4,
                text=f"<span style=''>{col}</span>",
                showarrow=False,
                font={
                    "color": rgb_to_color(hex_to_rgb("#9340ff")),
                    "family": "Gilroy-Light",
                    "size": 20,
                },
                name=f"{col} text",
            )

        #  Add entity points in correct order
        for i, col in enumerate(df_entity.columns):
            
            self.fig.add_trace(
                go.Scatter(
                    x=df_entity[col],
                    y=np.ones(len(df_entity)) * i,
                    mode="markers",
                    marker={
                        "size": 13,
                        "color": rgb_to_color(hex_to_rgb("#9340ff"), opacity=1),
                        "symbol": "square",
                    },
                    showlegend=False,
                    name=f"{col} selected",
                )
            )

        # show grid line x axis
        self.fig.update_xaxes(
            showgrid=True,
            fixedrange=True,
            gridwidth=1,
            gridcolor=rgb_to_color(hex_to_rgb("#6a5acd"), 0.7),
        )
        # show y axis or grid
        self.fig.update_yaxes(
            showgrid=False,
            showticklabels=False,
            fixedrange=True,
            visible=False,
            autorange="reversed"
        )

class ClusterVisualisation:
    def __init__(self, df, FA_label_map, u_labels, centroids, ind_col_map):
        self.df = df
        self.FA_label_map = FA_label_map
        self.u_labels = u_labels
        self.centroids = centroids  
        self.ind_col_map = ind_col_map
        self.list_cluster_name = st.session_state.list_cluster_name
   
        self.fig = go.Figure()
        self.set_visualization_cluster()

    def set_visualization_cluster(self):
        """Visualize K-Means clustering results using Plotly."""

        # Ensure required attributes are initialized
        required_attributes = ['df', 'centroids', 'u_labels', 'ind_col_map']
        for attr in required_attributes:
            if attr not in st.session_state:
                raise RuntimeError(f"Missing attribute: {attr}. Ensure clustering is run first.")


        
        dim_x = st.session_state["dim_x"]
        dim_y = st.session_state["dim_y"]

        inv_map = {st.session_state.FA_component_dict[k]["label"]: k for k in st.session_state.FA_component_dict.keys()}
        
        for i in st.session_state.u_labels:
            cluster_points = st.session_state.df[st.session_state.df['Cluster'] == i]
            #color=st.session_state.ind_col_map[i]
           
            self.fig.add_trace(
                go.Scatter(
                x=cluster_points.loc[:, inv_map[dim_x]],
                y=cluster_points.loc[:, inv_map[dim_y]],
                mode='markers',
                marker=dict(color=st.session_state.ind_col_map[i], size = 5, opacity=0.3),
                #name=f'Cluster {i}'
                name = self.list_cluster_name[i]
                )
            )

        
        # Plot centroids
        self.fig.add_trace(
            go.Scatter(
            x=st.session_state.centroids[:, int(inv_map[dim_x].split()[-1])],
            y=st.session_state.centroids[:, int(inv_map[dim_y].split()[-1])],
            mode='markers',
            marker=dict(color='black', size=6, symbol='x'),
            name='Centroids'
        ))

        #text = [f'Cluster {i}' for i in range(num_clusters)]
    
        #legend_list = []
        #for key in ind_col_map.keys():
        #    legend_list.append(mpatches.Patch(color=ind_col_map[key],label=f'$Cluster {key}$'))
        #    first_legend=ax.legend(title='Cluster',bbox_to_anchor=(1.02, 1),handles=legend_list, loc='upper left', borderaxespad=0)
        
        # Update layout
    
        self.fig.update_layout(
            title='K-Means Clustering Visualization',
            xaxis_title= dim_x,
            yaxis_title= dim_y,
            legend_title='Clusters',
            width=900,
            height=900,
            legend= {'itemsizing': 'constant'}
        )

class ClusterVisualisation3D:
    def __init__(self, df, FA_label_map, u_labels, centroids, ind_col_map):
        self.df = df
        self.FA_label_map = FA_label_map
        self.u_labels = u_labels
        self.centroids = centroids  
        self.ind_col_map = ind_col_map
        self.list_cluster_name = st.session_state.list_cluster_name
   
        self.fig = go.Figure()
        self.set_visualization_cluster3D()

    def set_visualization_cluster3D(self):
        """Visualize K-Means clustering results using Plotly."""

        # Ensure required attributes are initialized
        required_attributes = ['df', 'centroids', 'u_labels', 'ind_col_map']
        for attr in required_attributes:
            if attr not in st.session_state:
                raise RuntimeError(f"Missing attribute: {attr}. Ensure clustering is run first.")


        
        dim_x = st.session_state["dim_x"]
        dim_y = st.session_state["dim_y"]
        dim_z = st.session_state["dim_z"]

        inv_map = {st.session_state.FA_component_dict[k]["label"]: k for k in st.session_state.FA_component_dict.keys()}
        


        for i in st.session_state.u_labels:
            cluster_points = st.session_state.df[st.session_state.df['Cluster'] == i]
            #color=st.session_state.ind_col_map[i]
            self.fig.add_trace(
                go.Scatter3d(
                x=cluster_points.loc[:, inv_map[dim_x]],
                y=cluster_points.loc[:, inv_map[dim_y]],
                z=cluster_points.loc[:, inv_map[dim_z]],
                mode='markers',
                marker=dict(color=st.session_state.ind_col_map[i], size = 5, opacity=0.3),
                name = self.list_cluster_name[i]
                )
            )
        
        # Plot centroids
        self.fig.add_trace(
            go.Scatter3d(
            x=st.session_state.centroids[:, int(inv_map[dim_x].split()[-1])],
            y=st.session_state.centroids[:, int(inv_map[dim_y].split()[-1])],
            z=st.session_state.centroids[:, int(inv_map[dim_z].split()[-1])],   
            mode='markers',
            marker=dict(color='black', size=6, symbol='x'),
            name='Centroids'
        ))

        #text = [f'Cluster {i}' for i in range(num_clusters)]
    
        #legend_list = []
        #for key in ind_col_map.keys():
        #    legend_list.append(mpatches.Patch(color=ind_col_map[key],label=f'$Cluster {key}$'))
        #    first_legend=ax.legend(title='Cluster',bbox_to_anchor=(1.02, 1),handles=legend_list, loc='upper left', borderaxespad=0)
        
        
        # Update layout
        self.fig.update_layout(
            title='K-Means Clustering 3D Visualization',
            scene=dict(
            xaxis_title=dim_x,
            yaxis_title=dim_y,
            zaxis_title=dim_z
            ),
        legend_title='Clusters',
        width=900,
        height=900,
        legend= {'itemsizing': 'constant'}
        )
