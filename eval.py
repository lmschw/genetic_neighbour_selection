

import pandas  as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import davies_bouldin_score, silhouette_score


#location = "results/own_order_basic_n=5/"
#base_filepath = f"{location}best_own_order_basic_n=5_b5_normalised"
base_filepath = "best_own_random_order_basic_n=10_b5_normalised"

df = pd.read_csv(f"{base_filepath}.csv")

print(df.head())

print(f"avg fitness: {np.average(df['fitness'])}, min: {np.min(df['fitness'])}, max: {np.max(df['fitness'])}")
cols = {f"individual_{i}": f"{i}" for i in range(len(df.columns)-2)}
df = df.rename(columns=cols)
data = df.drop(columns=['iter', 'fitness'])

plt.plot(data)
plt.savefig(f"{base_filepath}_full_line.jpeg")


plt.figure(figsize = (10,7))
idx_list = [2, 4, 3, 0]
example_data = data.iloc[idx_list]
plt.plot(example_data.T)
plt.legend(idx_list)
plt.savefig(f"{base_filepath}_clusters.jpeg")
plt.show()

plt.figure(figsize = (10,7))
idx_list = [11, 0, 1]
example_data = data.iloc[idx_list]
plt.plot(example_data.T)
plt.legend(idx_list)
plt.savefig(f"{base_filepath}_cluster_line_similar.jpeg")
plt.show()


pca = PCA(n_components=2)
pca.fit(data)
df_pca = pca.transform(data)
df_pca = pd.DataFrame(df_pca, columns = ['P1', 'P2']) 

# plt.scatter(df_pca.P1, df_pca.P2)
# plt.xlabel('PCA 1')
# plt.ylabel('PCA 2')
# plt.show()



plt.figure(figsize = (10,7))
plt.title('Dendrograms')
plt.axhline(y=3, color='r', linestyle='--')
dend = sch.dendrogram(sch.linkage(df_pca, method='ward'))
plt.savefig(f"{base_filepath}_dendrogram.svg")
plt.savefig(f"{base_filepath}_dendrogram.jpeg")

model = AgglomerativeClustering(n_clusters = 2)
y_means = model.fit_predict(df_pca)     
plt.figure(figsize=(10, 8))
plt.title('Agglomerative Clustering with 2 Clusters')
plt.scatter(df_pca.P1, df_pca.P2, c=y_means, s=100)
plt.savefig(f"{base_filepath}_agglo_scatter.jpeg")
plt.show()
