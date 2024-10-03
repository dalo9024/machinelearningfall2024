import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

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

    # plot s values
    y_lower = 10
    for i in range(n_clusters):
        #s scores for samples belonging to cluster i
        i_cluster_s_values = sample_s_values[cluster_labels == i]
        i_cluster_s_values.sort()

        #size of cluster
        size_cluster_i = i_cluster_s_values.shape[0]
        y_upper = y_lower + size_cluster_i

        #fill s plot
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, i_cluster_s_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # label the s plots with cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        #new y_lower for next plot
        y_lower = y_upper + 10

    ax1.set_title(f"Silhouette plot for the clusters = {n_clusters}")
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    #line for average s score of all samples
    ax1.axvline(x=s_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks(np.arange(-1.1, 1.1, 0.2))

    plt.savefig(f"silhouette_plot_n_clusters_{n_clusters}.png", bbox_inches="tight")
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
    colors = ["blue", "green", "orange", "purple", "cyan", "magenta", "yellow", "black"]
    color_map = {placement: colors[i % len(colors)] for i, placement in enumerate(unique_placements)}

    #add points for each cluster
    for cluster in np.unique(cluster_labels):
        cluster_points = transformed[cluster_labels == cluster]
        
        #add trace for cluster
        fig.add_trace(go.Scatter3d(
            x=cluster_points["PC1"],
            y=cluster_points["PC2"],
            z=cluster_points["PC3"],
            mode="markers",
            marker=dict(size=5, opacity=0.25, color=[color_map[placement] for placement in dflabel[cluster_labels == cluster]]),
            name=f"Cluster {cluster}",
            legendgroup=f"Cluster {cluster}"
        ))

    #add centroids
    for i, center in enumerate(kmeans.cluster_centers_):
        fig.add_trace(go.Scatter3d(
            x=[center[0]],
            y=[center[1]],
            z=[center[2]],
            mode='markers',
            marker=dict(size=10, color="red", symbol="x"),
            name=f"Centroid {i}",
            showlegend=False
        ))

    #placement legend
    for placement in unique_placements:
        fig.add_trace(go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode="markers",
            marker=dict(size=10, color=color_map[placement]),
            name=f"Placement: {placement}",
            legendgroup="Placement",
            showlegend=True
        ))

    #axis
    fig.update_layout(
        title=f"Clustering {n_clusters} Clusters",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
            xaxis=dict(range=[transformed["PC1"].min(), transformed["PC1"].max()]),
            yaxis=dict(range=[transformed["PC2"].min(), transformed["PC2"].max()]),
            zaxis=dict(range=[transformed['PC3'].min(), transformed["PC3"].max()]),
            aspectmode="cube",
        ),
        plot_bgcolor="white",  
        paper_bgcolor="white",  
        legend=dict(title="Legend")
    )

    #save as html
    fig.write_html(f"3D_clustering_{n_clusters}.html")

#plot and save for each number of clusters
for clusters in [2, 3, 4, 8]:
    plot_clusters(clusters)

#results for each cluster
cluster_results = []
for clusters in [2, 3, 4, 8]:

    #kmeans
    kmeans = KMeans(n_clusters=clusters, random_state=10)
    cluster_labels = kmeans.fit_predict(transformed)

    #calculate means of pcs and placment for each cluster
    for cluster in np.unique(cluster_labels):
        cluster_points = transformed[cluster_labels == cluster]
        mean_values = cluster_points.mean().values
        mean_placement = dflabel[cluster_labels == cluster].mean()

        cluster_results.append({'Cluster': cluster,'Mean PC1': mean_values[0],'Mean PC2': mean_values[1], 'Mean PC3': mean_values[2], 'Mean Placement': mean_placement
        })

cluster_means_df = pd.DataFrame(cluster_results)
print(cluster_means_df)

########################
#Hierarchical Clustering
########################

np.random.seed(100)

#take a random sample from the data crashes if using full dataset
sample_size = 7000
random_indices = np.random.choice(transformed.index, size=sample_size, replace=False)
sampled_data = transformed.loc[random_indices]
sampled_placement = dflabel.loc[sampled_data.index]

#creates data for dendrogram
Z = linkage(sampled_data, method='complete', metric = 'cosine') 

#dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z, truncate_mode='level',color_threshold = 1.5, no_labels = True, show_leaf_counts = True, p = 10)  # p=3 means showing only the top 3 levels
plt.title('Hierarchical Clustering Dendrogram')
plt.ylabel('Distance')
plt.savefig("dendrogram.png", bbox_inches='tight')
plt.show()

#same cutoff as dendrogram
cut_off = 1.5
#adds a cluster label each point
cluster_labels = fcluster(Z, t=cut_off, criterion='distance')
sampled_data['cluster'] = cluster_labels
sampled_data['placement'] = sampled_placement
cluster_char = sampled_data.groupby('cluster').mean()
print(cluster_char)

###########################
# DBSCAN Clustering
###########################

#parameters for dbscan
eps_values = np.arange(0.1, 1, 0.1)  
min_samples = 8 

#s score for different eps values
for eps in eps_values:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(transformed[['PC1', 'PC2', 'PC3']])
    
    #stop computing if there is only one cluster 
    if len(set(cluster_labels)) > 2:
        score = silhouette_score(transformed_df[['PC1', 'PC2', 'PC3']], cluster_labels)
        print(f"For eps={eps}, silhouette score is: {score}")

#parameters for dbscan based on s score
eps = 0.8  
min_samples = 8  

#fit bdscan
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan_labels = dbscan.fit_predict(transformed)

#add cluster labels and original placement back on
transformed['Cluster'] = dbscan_labels
transformed['Placement'] = dflabel

#
fig = go.Figure()

#colors for placement
unique_placements = sorted(dflabel.unique())
colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black']
color_map = {placement: colors[i % len(colors)] for i, placement in enumerate(unique_placements)}

#add points for each cluster
for cluster in np.unique(dbscan_labels):
    cluster_points = transformed[transformed['Cluster'] == cluster]
    
    #trace for clusters
    if cluster != -1:
        fig.add_trace(go.Scatter3d(
            x=cluster_points['PC1'],
            y=cluster_points['PC2'],
            z=cluster_points['PC3'],
            mode='markers',
            marker=dict(size=5, opacity=0.25, color=[color_map[placement] for placement in cluster_points['Placement']]),
            name=f'Cluster {cluster}',
            legendgroup=f'Cluster {cluster}'
        ))
      
    #trace for noise
    else:  
        fig.add_trace(go.Scatter3d(
            x=cluster_points['PC1'],
            y=cluster_points['PC2'],
            z=cluster_points['PC3'],
            mode='markers',
            marker=dict(size=5, opacity=0.5, color='red'),
            name='Noise',
            legendgroup='Noise'
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
    title='DBSCAN Clustering',
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

#save to html and show
fig.write_html('DBScan.html')
fig.show()
