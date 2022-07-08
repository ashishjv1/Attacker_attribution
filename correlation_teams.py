
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('normalized_union_all_teams_no_time.csv').fillna(0)
data = data.iloc[:, 1:-1]

# label_encoder = LabelEncoder()
# data.iloc[:, 0] = label_encoder.fit_transform(data.iloc[:, 0]).astype('float64')

corr = data.corr()
sns.heatmap(corr)
# plt.figure(figsize=(200, 200))
# plt.show()
# mask = np.triu(np.ones_like(corr, dtype=bool))
# f, ax = plt.subplots(figsize=(11, 9))
# cmap = sns.diverging_palette(230, 20, as_cmap=True)
# sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5})
# plt.show()

columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i + 1, corr.shape[0]):
        if corr.iloc[i, j] >= 1:
            if columns[j]:
                columns[j] = False
selected_columns = data.columns[columns]

data1 = data[selected_columns]
corr1 = data1.corr()

mask1 = np.triu(np.ones_like(corr1, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap1 = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr1, mask=mask1, cmap=cmap1, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# sns.heatmap(corr1)
# plt.figure(figsize=(200, 200))
# plt.show()
# print(data1.shape)
corr1.to_csv('correlated_data.csv', index=True)

# print(df.head())
# print(df.info())
# print(df.dtypes())
# print(df.shape)
# print(df.describe())
# print(df.columns)


# corrMatrix = df.corr(df.iloc[:,1:-1])

# print(corrMatrix)
# sn.heatmap(corrMatrix, annot=True)
# plt.figure(figsize=(200, 100))
# plt.show()

# columns = np.full((corrMatrix.shape[0],), True, dtype=bool)
# for i in range(corrMatrix.shape[0]):
#   for j in range(i + 1, corrMatrix.shape[0]):
#        if corrMatrix.iloc[i, j] >= 0.9:
#           if columns[j]:
#              df = df[selected_columns]

# print(df)
# data_returns = df.pct_change()
# sns.jointplot(x='linux_bandwidth_av_rxPackets_PS', y='linux_bandwidth_std_rxKB_PS', data=data_returns)
# plt.show()


# array = df.values
# X = array[:, 0:100]
# Y = array[:, 8]
# X = StandardScaler().fit_transform(X)
# X = X.astype('float32')
# print(X)
# print(Y)
# feature extraction
# pca = PCA(n_components=44)
# fit = pca.fit(X)
# summarize components
# print("Explained Variance: %s" % fit.explained_variance_ratio_)
# test = fit.components_
# sn.heatmap(test, annot=True)
# plt.figure(figsize=(200, 100))
# plt.show()
