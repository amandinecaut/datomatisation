from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import plotly

import description
from description import CreateDescription
class Cluster:
    def __init__(self, df, FA_label_map, num_clusters):
        cols = [k for k in FA_label_map.keys()]
        self.df = df
        self.df_values = df.values
        self.FA_label_map = FA_label_map
        self.num_clusters = num_clusters

        self.clustering()

    def clustering(self):
        """Perform K-Means clustering and store results."""
        
        kmeans = KMeans(n_clusters=self.num_clusters, init='k-means++', max_iter=100, n_init=50, random_state=42)
        
        # Fit and predict cluster labels
        labels = kmeans.fit_predict(self.df)

        
        self.df['Cluster'] = labels

        
        # Store centroids and unique labels
        self.centroids = kmeans.cluster_centers_
        # Get the cluster name
        self.list_cluster_name = self.name_the_cluster(self.centroids)
        # Get the cluster (long) description
        self.list_description_cluster = self.description_cluster(self.centroids)
        
        self.u_labels = self.df['Cluster'].unique()


        # Create cluster color map
        self.ind_col_map = {label: color for label, color in zip(self.u_labels, plotly.colors.qualitative.Set1[:len(self.u_labels)])}
        self.ind_col_map = dict(sorted(self.ind_col_map.items()))

        st.session_state.u_labels, st.session_state.centroids, st.session_state.ind_col_map = self.u_labels , self.centroids,  self.ind_col_map
        st.session_state.df = self.df
        st.session_state.list_cluster_name =  self.list_cluster_name
        st.session_state.list_description_cluster = self.list_description_cluster
 
    def name_the_cluster(self, centroids):
        create_description = CreateDescription()
        liste_name_dim = []
        
        for _ , details in st.session_state.FA_component_dict.items():
            liste_name_dim.append(details['label'])
        
        
        list_name_cluster = []

        for center in self.centroids:
            describe_centroid = []
            for dim in np.arange(len(center)):
                value_dim = center[dim]
                text_dim = create_description.describe_level_cluster(value_dim)
                text_low, text_high = create_description.split_qualities(liste_name_dim[dim])


                if value_dim >= 0:
                    text_dim += text_high
                else:
                    text_dim += text_low
                describe_centroid.append(text_dim)
            text = ", ".join(describe_centroid)  
            
            text = create_description.get_cluster_label(text)
            list_name_cluster.append(text)
        return list_name_cluster

    def description_cluster(self, centroids):
        create_description = CreateDescription()
        liste_name_dim = []
        for _ , details in st.session_state.FA_component_dict.items():
            liste_name_dim.append(details['label'])
        
        list_description_cluster = []

        for center in self.centroids:
            describe_centroid = []
            for dim in np.arange(len(center)):
                value_dim = center[dim]
                text_dim = create_description.describe_level_cluster(value_dim)
                text_low, text_high = create_description.split_qualities(liste_name_dim[dim])

                if value_dim >= 0:
                    text_dim += text_high
                else:
                    text_dim += text_low
                describe_centroid.append(text_dim)
            text = ", ".join(describe_centroid)  
            
            text = create_description.get_cluster_description(text)
            list_description_cluster.append(text)
        return list_description_cluster


    # Useless for now
    def find_closest_points(self, kmeans):
        """Find indices of the closest points to each cluster center."""
        closest_pt_idx = []
        for iclust in range(kmeans.n_clusters):
            cluster_pts_indices = np.where(kmeans.labels_ == iclust)[0]


            cluster_pts = self.df_values.iloc[cluster_pts_indices]
            cluster_cen = kmeans.cluster_centers_[iclust]

            # Efficient distance calculation
            distances = cdist(cluster_pts, [cluster_cen])
            min_idx = np.argmin(distances)
            closest_pt_idx.append(cluster_pts_indices[min_idx])

        return closest_pt_idx


