import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# from matplotlib import inline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA

df = pd.read_csv('union_normalized.csv').fillna(0)
columns_names = df.columns.tolist()
print("Columns names:")
print(columns_names)

cols = df.columns.tolist()


df_drop = df.reindex(columns= cols)

X = df_drop.iloc[:,1:149].values
y = df_drop.iloc[:,0].values

X_std = StandardScaler().fit_transform(X)
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)
print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))
# plt.figure(figsize=(8,8))
# sns.heatmap(cov_mat, vmax=1, square=True,annot=True,cmap='cubehelix')
#
# plt.title('Correlation between different features')
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
with plt.style.context('classic'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(148), var_exp, alpha=0.5, align='center',
            label='variance of each feature')
    plt.ylabel('Variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()

matrix_w = np.hstack((eig_pairs[0][1].reshape(148,1),
                      eig_pairs[1][1].reshape(148,1)
                    ))
print('Matrix W:\n', matrix_w)
Y = X_std.dot(matrix_w)
pca = PCA().fit(X_std)

# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlim(0,47,1)
# plt.xlabel('Number of components')
# plt.ylabel('Cumulative explained variance')

sklearn_pca = PCA(n_components=44)
Y_sklearn = sklearn_pca.fit_transform(X_std)
print(Y_sklearn)
print(Y_sklearn.shape)