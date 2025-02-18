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


class Visualization:
    def __init__(self, df_pca, pca_label_map):

        # print("creating visualization")
        cols = [k for k in pca_label_map.keys()]
        self.df_pca = df_pca[cols]
        self.pca_label_map = pca_label_map

        self.df_z_scores = self.get_z_scores()
        # map self.df_z_scores using pca_label_map
        self.df_z_scores.rename(columns=self.pca_label_map, inplace=True)

        self.fig = go.Figure()
        self.set_visualization()

    def get_z_scores(self):
        return self.df_pca.apply(zscore, nan_policy="omit")

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

        df = self.df_z_scores.iloc[0, :].to_frame().T

        for i, col in enumerate(df.columns):

            self.fig.add_trace(
                go.Scatter(
                    x=df[col],
                    y=np.ones(len(df)) * i,
                    mode="markers",
                    marker={
                        "size": 15,
                        "color": rgb_to_color(hex_to_rgb(color), opacity=1),
                        "symbol": "square",
                    },
                    # down add to legend
                    showlegend=False,
                    name=f"{col} selected",
                )
            )

        # show grid line x axis
        self.fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=rgb_to_color(hex_to_rgb("#808080"), 0.8),
        )
        # do not show y axis or grid
        self.fig.update_yaxes(showgrid=False, visible=False)
