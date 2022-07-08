
import pandas as pd

from pandas import read_csv
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('union_normalized.csv', index_col='Team_name').fillna(0)
# data = data.iloc[:, 1:-1]

# label_encoder = LabelEncoder()
# data.iloc[:, 0] = label_encoder.fit_transform(data.iloc[:, 0]).astype('float64')
array = data.values
print(array.shape)

X = array[0:43]
Y = array[43]

# feature extraction
pca = PCA(n_components=43)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
# print(fit.components_)
save = fit.components_
print(save)