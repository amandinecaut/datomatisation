import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
import matplotlib.patches as mpatches
import streamlit as st

class Cluster:
    def __init__(self, FA_df, FA_label_map, num_clusters):
        cols = [k for k in FA_label_map.keys()]
        self.FA_df = FA_df
        self.df = FA_df.values
        self.FA_label_map = FA_label_map
        self.num_clusters = num_clusters

        self.clustering()
        self.fig = go.Figure()
        self.set_visualization_cluster()
        

    def clustering(self):
        """Perform K-Means clustering and store results."""
        
        kmeans = KMeans(n_clusters=self.num_clusters, init='k-means++', max_iter=100, n_init=50, random_state=42)
        
        # Fit and predict cluster labels
        labels = kmeans.fit_predict(self.FA_df)

        self.FA_df['Cluster'] = labels

        
        # Store centroids and unique labels
        self.centroids = kmeans.cluster_centers_
        self.u_labels = np.unique(labels)

        # Find the closest point to each cluster center
        #self.closest_pt_idx = self.find_closest_points(kmeans)

        # Create cluster color map
        self.ind_col_map = {x: y for x, y in zip(self.u_labels, sns.color_palette('tab20', self.num_clusters))}
        self.ind_col_map = dict(sorted(self.ind_col_map.items()))

        st.session_state.u_labels, st.session_state.centroids, st.session_state.ind_col_map = self.u_labels , self.centroids,  self.ind_col_map

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


    def set_visualization_cluster(self):
        """Visualize K-Means clustering results using Plotly."""

        # Ensure required attributes are initialized
        required_attributes = ['FA_df', 'centroids', 'u_labels', 'ind_col_map']
        for attr in required_attributes:
            if attr not in st.session_state:
                raise RuntimeError(f"Missing attribute: {attr}. Ensure clustering is run first.")


        # Plot each cluster
        for i in st.session_state.u_labels:
            cluster_points = st.session_state.FA_df[st.session_state.FA_df['Cluster'] == i]
            self.fig.add_trace(
                go.Scatter(
                x=cluster_points.iloc[:, 0],
                y=cluster_points.iloc[:, 1],
                mode='markers',
                marker=dict(color=st.session_state.ind_col_map[i], opacity=0.25),
                name=f'Cluster {i}'
            ))
            #ax.scatter(df2[label == i , 0] , df2[label == i , 1] ,color=ind_col_map[i],  label = i, alpha= 0.25)

            # Plot centroids
        self.fig.add_trace(go.Scatter(
            x=st.session_state.centroids[:, 0],
            y=st.session_state.centroids[:, 1],
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
            title='K-Means Clustering Visualization',
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            legend_title='Clusters',
            width=800,
            height=600
        )

        # Display the figure in Streamlit
        #st.plotly_chart(fig) # Render the chart

    def set_visualization_cluster_1(self):
        """Visualize K-Means clustering results."""

        # Ensure required attributes are initialized
        required_attributes = ['FA_df', 'centroids', 'u_labels', 'ind_col_map']
        for attr in required_attributes:
            if attr not in st.session_state:
                raise RuntimeError(f"Missing attribute: {attr}. Ensure clustering is run first.")

        # Create a matplotlib figure
        self.fig, ax = plt.subplots()

        # Plot each cluster
        for i in st.session_state.u_labels:
            cluster_points = st.session_state.FA_df[st.session_state.FA_df['Cluster'] == i].values
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], color=st.session_state.ind_col_map[i],
                   label=f'Cluster {i}', alpha=0.25)

        # Plot centroids
        ax.scatter(st.session_state.centroids[:, 0], st.session_state.centroids[:, 1],
               s=50, color='k', marker='x', label='Centroids')

        # Create legend
        legend_patches = [mpatches.Patch(color=color, label=f'Cluster {i}')
                      for i, color in st.session_state.ind_col_map.items()]
        ax.legend(handles=legend_patches, title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')

