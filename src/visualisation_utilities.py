import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import zscore
import streamlit as st
import plotly.express as px

def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)

def rgb_to_color(rgb_color: tuple, opacity=1):
    return f"rgba{(*rgb_color, opacity)}"


def wrap_text(text, max_len=15):
    words = text.split()
    wrapped_text = ""
    current_len = 0
    for word in words:
        if current_len + len(word) > max_len:
            wrapped_text += "<br>"
            current_len = 0
        wrapped_text += word + " "
        current_len += len(word) + 1
    return wrapped_text.strip()

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
        self.set_visualisation()

    def get_z_scores(self):
        return self.df_FA.apply(zscore, nan_policy="omit")

    def set_visualisation(self):
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

        u_labels = getattr(st.session_state, "u_labels", np.array([]))
        centroids = getattr(st.session_state, "centroids", None)
        ind_col_map = getattr(st.session_state, "ind_col_map", None)
        FA_component_dict = getattr(st.session_state, "FA_component_dict", {})
        list_cluster_name = getattr(st.session_state, "list_cluster_name", None)
        dim_x = getattr(st.session_state, "dim_x", None)
        dim_y = getattr(st.session_state, "dim_y", None)

        # Check if required data is available
        if u_labels.size == 0 or ind_col_map is None:
            st.warning("Clustering not yet run. Please run clustering first.")
            return
        if centroids is None:
            st.info("Centroids not yet available. Only plotting cluster points.")
        if not dim_x or not dim_y:
            st.warning("Select dimensions (dim_x and dim_y) before plotting.")
            return

        # Ensure required attributes are initialized
        #required_attributes = ['df', 'centroids', 'u_labels', 'ind_col_map']
        #for attr in required_attributes:
        #    if attr not in st.session_state:
        #        raise RuntimeError(f"Missing attribute: {attr}. Ensure clustering is run first.")


        
        #dim_x = st.session_state["dim_x"]
        #dim_y = st.session_state["dim_y"]

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
            marker=dict(color='black', size=4, symbol='x'),
            name='Centroids'
        ))

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



class DistributionPlot:
    def __init__(self, df_FA, FA_label_map, *args, **kwargs):
        self.background = hex_to_rgb("#faf9ed")
       
        cols = [k for k in FA_label_map.keys()]
        self.df_FA = df_FA[cols]
        self.FA_label_map = FA_label_map

        self.df_z_scores = self.get_z_scores()

        self.df_z_scores.rename(columns=self.FA_label_map, inplace=True)

        self.fig = go.Figure()
        self.set_visualisation()
        
        self._setup_axes()


    def _setup_axes(self):
        self.fig.update_xaxes(
            range=[-4, 4],
            fixedrange=True,
            showgrid=False,
            gridcolor=rgb_to_color(hex_to_rgb("#6a5acd"), 0.7),
        )
        self.fig.update_yaxes(
            showticklabels=True,
            fixedrange=True,
            showgrid=False,
            zerolinecolor=rgb_to_color(hex_to_rgb("#ffffff")),
        )

    def get_z_scores(self):
        return self.df_FA.apply(zscore, nan_policy="omit")


    def set_visualisation(self):
        #color = st.get_option("theme.primaryColor")
        #if color is None:
        #    color = "#FF4B4B"
        colors = px.colors.qualitative.Set2

        
        dataframe = self.df_z_scores

        # Ensure correct order by using FA_label_map
        dataframe = dataframe[list(self.FA_label_map.values())]
        df = self.df_z_scores.iloc[0, :].to_frame().T
        cols = dataframe.columns.tolist()
     
       
        # Create subplots
        self.fig = make_subplots(
            rows=len(dataframe.columns),
            cols=1,
            shared_xaxes=True, 
            vertical_spacing=0.0
        )


        for i, col in enumerate(dataframe.columns):
            
            self.fig.add_trace(
                go.Violin(
                    x=dataframe[col].tolist(),
                    name=cols[i],
                    marker=dict(color=colors[i % len(colors)]),
                    opacity=0.65,
                    side='positive',
                    showlegend = False,
                    #hovertemplate=f"<b>{cols[i]}</b><br>Value: %{{x}}<br>Count: %{{y}}<extra></extra>",
                    hoverinfo="skip", hovertemplate=None,
                    points=False,
                ),
                row=i+1,
                col=1
            )
            hovertext=(
                  f"<b>{cols[i]}</b><br>Value: %{{x}}<br><extra></extra>"
            )

            self.fig.add_trace(
                go.Scatter(
                    x=df[col],
                    y=[cols[i]],
                    mode="markers", # if we want marker and text do "markers+text"
                    marker=dict(symbol="diamond", size=6, color="#9340ff"),
                    name= f"{col} selected",
                    legendgroup ="Selected entity",
                    showlegend=False,
                    #showlegend=(i == 0),  # ensures legend is shown
                    hovertemplate= hovertext,#f"<b>{cols[i]}</b><br>Value: %{{x}}<br><extra></extra>",
                    ),
                    row=i+1,
                    col=1
                    )

        self.fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(symbol="diamond", size=6, color="#9340ff"),
                name=f"{st.session_state.selected_entity}",
                showlegend=True,
                uid = "dummy_legend_name"
                )
            )

        # Update layout
        self.fig.update_layout(
            template="plotly_white",
            title=dict(text="<b>Distribution of Metrics</b>",x=0.55, font=dict(size=14)),
            showlegend=True,
            margin=dict(t=50, b=50, l=45, r=25),
            font = dict(size=14),
            autosize=True,
            legend=dict(
                yanchor="bottom",
                y=-0.2,
                xanchor="right",
                x=1,
                font=dict(size=10)
            )
        )

        # Add grid & font styling
        self.fig.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
        self.fig.update_yaxes(showgrid=False)
        


class RadarPlot:
    def __init__(self, FA_label_map):

        self.cols = metrics
        self.entity = entity.ser_metrics
        self.color = hex_to_rgb("#faf9ed")
        self.fig = go.Figure()
        self.set_visualization()


    def set_visualization(self):
        # Streamlit primary color
        color = st.get_option("theme.primaryColor")
        if color is None:
            color = "#FF4B4B"

    
        df_entity = st.session_state.df_z_scores.iloc[ind, :].to_frame().T
        r_values = df_entity.values.tolist()
        #theta_values=self.cols
        theta_values=[wrap_text(c) for c in self.cols]

        # Repeat the first element at the end to close the polygon
        r_values.append(r_values[0])
        theta_values = theta_values + [theta_values[0]]

        # Add the entity as a highlighted polygon
        self.fig.add_trace(
            go.Scatterpolar(
                r = r_values,
                theta = theta_values,
                mode="lines+markers",
                line=dict(color=rgb_to_color(hex_to_rgb("#9340ff")), width=3),
                marker=dict(size=8, color=rgb_to_color(hex_to_rgb("#9340ff"))),
                fill="toself",
                hovertemplate="<b>%{theta}</b>: %{r}<extra></extra>", 
                showlegend=False,
            )
        )


        self.fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-4, 4],
                    gridcolor=rgb_to_color(hex_to_rgb("#E0E0E0")),
                    linecolor=rgb_to_color(hex_to_rgb("#CCCCCC")),
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    tickfont=dict(size=10, family="Gilroy-Light", color="#333")
                )
            ),
            margin=dict(l=75, r=85, t=55, b=55),
            plot_bgcolor="white",
            showlegend=False,
            autosize=True,
        )