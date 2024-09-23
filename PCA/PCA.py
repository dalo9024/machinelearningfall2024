import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



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
scaled_df.to_csv('PCA_data.csv', index=False)  

######################
# PCA with 2 components
######################

pca1 = PCA(n_components=2)
pca1.fit(scaled_df)
transformed_2 = pca1.transform(scaled_df)

# Plot of transformed data 2 components
colors = plt.cm.viridis(np.linspace(0, 1, len(placements)))

plt.figure(figsize=(8, 6))
for i, placement in enumerate(placements):
    mask = dflabel == placement
    plt.scatter(transformed_2[mask, 0], 
                transformed_2[mask, 1], 
                color=colors[i], 
                s=1, 
                label=placement)

plt.title('PCA Scatter Plot: 2 Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Placement', markerscale=8, bbox_to_anchor=(1, 1), loc='upper left')
plt.tight_layout()
plt.savefig('PCA2.png', format='png', dpi=300)
plt.show()

######################
# PCA with 3 components
######################

pca2 = PCA(n_components=3)
pca2.fit(scaled_df)
transformed_3 = pca2.transform(scaled_df)


# Plot of transformed data 3 components
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')
for i, placement in enumerate(placements):
    mask = dflabel == placement
    ax.scatter(transformed_3[mask, 0], 
               transformed_3[mask, 1], 
               transformed_3[mask, 2], 
               color=colors[i], 
               s=1, 
               label=placement)

ax.set_title('PCA Scatter Plot: 3 Components')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')

plt.legend(title='Placement', markerscale=8, bbox_to_anchor=(1, 1), loc='upper left')
plt.tight_layout()
plt.savefig('PCA3.png', format='png', dpi=300)
plt.show()

#########################
# PCA with all components
#########################
pca5 = PCA(n_components=5)
pca5.fit(scaled_df)
variance_retained = pca5.explained_variance_ratio_.cumsum()
print("The cumulative explained variance as a ratio goes up as follows:", variance_retained)
p_comp = range(1,6)


#plot of cumulative variance retained
plt.plot(p_comp, variance_retained, marker='o', linestyle='--')
plt.axhline(y=0.95, color='red', label='95% Variance Retained')
plt.title('Cumulative Variance Retained by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Retained')
plt.xticks(p_comp)
plt.legend() 
plt.ylim(0, 1)
plt.savefig('VarianceRetained.png', format='png', dpi=300)
plt.show()


#eigenvalues
eigenvalues = pca5.explained_variance_
print("The top three eigenvalues are: ",eigenvalues[:3])