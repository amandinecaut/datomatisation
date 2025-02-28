from scipy.spatial.distance import euclidean
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import plotly


class Cluster:
    def __init__(self, FA_df, FA_label_map, num_clusters):
        cols = [k for k in FA_label_map.keys()]
        self.FA_df = FA_df
        self.df = FA_df.values
        self.FA_label_map = FA_label_map
        self.num_clusters = num_clusters

        self.clustering()

        

    def clustering(self):
        """Perform K-Means clustering and store results."""
        
        kmeans = KMeans(n_clusters=self.num_clusters, init='k-means++', max_iter=100, n_init=50, random_state=42)
        
        # Fit and predict cluster labels
        labels = kmeans.fit_predict(self.FA_df)

        
        self.FA_df['Cluster'] = labels

        
        # Store centroids and unique labels
        self.centroids = kmeans.cluster_centers_
        #self.u_labels = np.unique(labels)
        self.u_labels = self.FA_df['Cluster'].unique()

        # Find the closest point to each cluster center
        #self.closest_pt_idx = self.find_closest_points(kmeans)

        # Create cluster color map
        #self.ind_col_map = {label: color for label, color in zip(self.u_labels, sns.color_palette('tab20', len(self.u_labels)))}
        self.ind_col_map = {label: color for label, color in zip(self.u_labels, plotly.colors.qualitative.Set1[:len(self.u_labels)])}
        self.ind_col_map = dict(sorted(self.ind_col_map.items()))

        st.session_state.u_labels, st.session_state.centroids, st.session_state.ind_col_map = self.u_labels , self.centroids,  self.ind_col_map
        st.session_state.FA_df = self.FA_df

       

    def find_closest_points(self, kmeans):
        """Find indices of the closest points to each cluster center."""
        closest_pt_idx = []
        for iclust in range(kmeans.n_clusters):
            cluster_pts = self.df[kmeans.labels_ == iclust]
            cluster_pts_indices = np.where(kmeans.labels_ == iclust)[0]
            cluster_cen = kmeans.cluster_centers_[iclust]
            min_idx = np.argmin([euclidean(self.df[idx], cluster_cen) for idx in cluster_pts_indices])
            closest_pt_idx.append(cluster_pts_indices[min_idx])

        return closest_pt_idx

    
class ClusterVisualisation:
    def __init__(self, FA_df, FA_label_map, u_labels, centroids, ind_col_map):
        self.FA_df = FA_df
        self.FA_label_map = FA_label_map
        self.u_labels = u_labels
        self.centroids = centroids  
        self.ind_col_map = ind_col_map
   
        self.fig = go.Figure()
        self.set_visualization_cluster()

    def set_visualization_cluster(self):
        """Visualize K-Means clustering results using Plotly."""

        # Ensure required attributes are initialized
        required_attributes = ['FA_df', 'centroids', 'u_labels', 'ind_col_map']
        for attr in required_attributes:
            if attr not in st.session_state:
                raise RuntimeError(f"Missing attribute: {attr}. Ensure clustering is run first.")


        
        dim_x = st.session_state["dim_x"]
        dim_y = st.session_state["dim_y"]

        inv_map = {st.session_state.FA_component_dict[k]["label"]: k for k in st.session_state.FA_component_dict.keys()}
        


        for i in st.session_state.u_labels:
            cluster_points = st.session_state.FA_df[st.session_state.FA_df['Cluster'] == i]
            color=st.session_state.ind_col_map[i]
           
            self.fig.add_trace(
                go.Scatter(
                x=cluster_points.loc[:, inv_map[dim_x]],
                y=cluster_points.loc[:, inv_map[dim_y]],
                mode='markers',
                marker=dict(color=st.session_state.ind_col_map[i], size = 2, opacity=0.15),
                name=f'Cluster {i}'
                )
            )

        # inv_map[dim_x]
        # gives "Principal Component 1"
        # int(inv_map[dim_x].split()[-1])
        # will return 1
        
        # Plot centroids
        self.fig.add_trace(
            go.Scatter(
            x=st.session_state.centroids[:, int(inv_map[dim_x].split()[-1])],
            y=st.session_state.centroids[:, int(inv_map[dim_y].split()[-1])],
            mode='markers',
            marker=dict(color='black', size=3, symbol='x'),
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
            height=900
        )

class ClusterVisualisation3D:
    def __init__(self, FA_df, FA_label_map, u_labels, centroids, ind_col_map):
        self.FA_df = FA_df
        self.FA_label_map = FA_label_map
        self.u_labels = u_labels
        self.centroids = centroids  
        self.ind_col_map = ind_col_map
   
        self.fig = go.Figure()
        self.set_visualization_cluster3D()

    def set_visualization_cluster3D(self):
        """Visualize K-Means clustering results using Plotly."""

        # Ensure required attributes are initialized
        required_attributes = ['FA_df', 'centroids', 'u_labels', 'ind_col_map']
        for attr in required_attributes:
            if attr not in st.session_state:
                raise RuntimeError(f"Missing attribute: {attr}. Ensure clustering is run first.")


        
        dim_x = st.session_state["dim_x"]
        dim_y = st.session_state["dim_y"]
        dim_z = st.session_state["dim_z"]

        inv_map = {st.session_state.FA_component_dict[k]["label"]: k for k in st.session_state.FA_component_dict.keys()}
        


        for i in st.session_state.u_labels:
            cluster_points = st.session_state.FA_df[st.session_state.FA_df['Cluster'] == i]
            color=st.session_state.ind_col_map[i]
            self.fig.add_trace(
                go.Scatter3d(
                x=cluster_points.loc[:, inv_map[dim_x]],
                y=cluster_points.loc[:, inv_map[dim_y]],
                z=cluster_points.loc[:, inv_map[dim_z]],
                mode='markers',
                marker=dict(color=st.session_state.ind_col_map[i], size = 1, opacity=0.15),
                name=f'Cluster {i}'
                )
            )

        # inv_map[dim_x]
        # gives "Principal Component 1"
        # int(inv_map[dim_x].split()[-1])
        # will return 1
        
        # Plot centroids
        self.fig.add_trace(
            go.Scatter3d(
            x=st.session_state.centroids[:, int(inv_map[dim_x].split()[-1])],
            y=st.session_state.centroids[:, int(inv_map[dim_y].split()[-1])],
            z=st.session_state.centroids[:, int(inv_map[dim_z].split()[-1])],   
            mode='markers',
            marker=dict(color='black', size=5, symbol='x'),
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
        height=900
        )
