from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)



scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)


pca = PCA().fit(scaled_data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.show()


pca = PCA(n_components=2)
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)


plt.scatter(pca_data[:,0], pca_data[:,1], c=wine.target)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()











