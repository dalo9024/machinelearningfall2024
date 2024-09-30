import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

#read in data
df = pd.read_csv("encoded_data.csv")

##########
#Data Prep
##########

#take only the numeric columns with no label
numeric_df = df[["gold_left", "level", "total_damage_to_players", "players_eliminated", "num_traits"]]
#save label values for later plotting
dflabel = df["placement"]
placements = sorted(dflabel.unique())

#scale the data using standard scalar.
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)
scaled_df = pd.DataFrame(scaled_data, columns=numeric_df.columns)
scaled_df.to_csv("clustering_data.csv", index=False)

######################
# PCA with 3 components
######################

pca = PCA(n_components=3)
pca.fit(scaled_df)
transformed = pd.DataFrame(pca.transform(scaled_df))
transformed.columns = ["PC1","PC2","PC3"]
transformed

###########################
#KMeans silhouette analysis 
###########################

range_n_clusters = list(range(2, 11))  

#calculate s scores for each cluster
for n_clusters in range_n_clusters:
    #kmeans with n clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = kmeans.fit_predict(transformed)

    #calcualte the s score
    s_avg = silhouette_score(transformed, cluster_labels)
    print(f"For n_clusters = {n_clusters}, the average silhouette score is: {s_avg}")

    #calculate s scores for each sample
    sample_s_values = silhouette_samples(transformed, cluster_labels)

    # Create a subplot
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(7, 5)
    ax1.set_xlim([-1, 1])

    # The (n_clusters+1)*10 is for inserting blank space between silhouette plots of individual clusters
    ax1.set_ylim([0, len(transformed) + (n_clusters + 1) * 10])

    # Plot silhouette values
    y_lower = 10
    for i in range(n_clusters):
        #s scores for samples belonging to cluster i
        i_cluster_s_values = sample_s_values[cluster_labels == i]
        i_cluster_s_values.sort()

        # Compute the size of the cluster
        size_cluster_i = i_cluster_s_values.shape[0]
        y_upper = y_lower + size_cluster_i

        #fill s plot
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, i_cluster_s_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples separation between clusters

    ax1.set_title(f"Silhouette plot for the clusters = {n_clusters}")
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # Vertical line for average silhouette score of all samples
    ax1.axvline(x=s_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the y-axis labels
    ax1.set_xticks(np.arange(-1.1, 1.1, 0.2))

    plt.savefig(f"silhouette_plot_n_clusters_{n_clusters}.png", bbox_inches='tight')
    plt.show()
    plt.close()
    
##################
#Plotting Clusters
##################

def plot_clusters(n_clusters):
    #kmeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = kmeans.fit_predict(transformed)

    #initialize figure
    fig = go.Figure()

    #colors for label
    unique_placements = sorted(dflabel.unique())
    colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black']
    color_map = {placement: colors[i % len(colors)] for i, placement in enumerate(unique_placements)}

    #add points for each cluster
    for cluster in np.unique(cluster_labels):
        cluster_points = transformed[cluster_labels == cluster]
        
        #add trace for cluster
        fig.add_trace(go.Scatter3d(
            x=cluster_points['PC1'],
            y=cluster_points['PC2'],
            z=cluster_points['PC3'],
            mode='markers',
            marker=dict(size=5, opacity=0.25, color=[color_map[placement] for placement in dflabel[cluster_labels == cluster]]),
            name=f'Cluster {cluster}',
            legendgroup=f'Cluster {cluster}'
        ))

    #add centroids
    for i, center in enumerate(kmeans.cluster_centers_):
        fig.add_trace(go.Scatter3d(
            x=[center[0]],
            y=[center[1]],
            z=[center[2]],
            mode='markers',
            marker=dict(size=10, color='red', symbol='x'),  # Use 'X' marker for centroids
            name=f'Centroid {i}',
            showlegend=False\
        ))

    #placement legend
    for placement in unique_placements:
        fig.add_trace(go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode='markers',
            marker=dict(size=10, color=color_map[placement]),
            name=f'Placement: {placement}',
            legendgroup='Placement',
            showlegend=True
        ))

    #axis
    fig.update_layout(
        title=f'Clustering {n_clusters} Clusters',
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3',
            xaxis=dict(range=[transformed['PC1'].min(), transformed['PC1'].max()]),
            yaxis=dict(range=[transformed['PC2'].min(), transformed['PC2'].max()]),
            zaxis=dict(range=[transformed['PC3'].min(), transformed['PC3'].max()]),
            aspectmode='cube',
        ),
        plot_bgcolor='white',  
        paper_bgcolor='white',  
        legend=dict(title='Legend')
    )

    #save as html
    fig.write_html(f'3D_clustering_{n_clusters}.html')

#plot and save for each number of clusters
for clusters in [2, 3, 4, 8]:
    plot_clusters(clusters)
