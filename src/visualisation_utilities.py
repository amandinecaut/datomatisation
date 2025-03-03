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
        print(df_entity)
        # add a scatter plot for each principal component
        for i, col in enumerate(df.columns):
            print(i, col)

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
            print('entity', i, col)
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
