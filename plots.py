
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph
import seaborn as sns # used for plot interactive graph.


df = pd.read_csv('central_team0_all-data.csv', parse_dates=['time'], infer_datetime_format=True, index_col='time')

print(df.head())
print(df.info())
#print(df.dtypes())
print(df.shape)
print(df.describe())
print(df.columns)

droping_list_all=[]
for j in range(0,3):
    if not df.iloc[:, j].notnull().all():
        droping_list_all.append(j)
        #print(df.iloc[:,j].unique())
print(droping_list_all)
for j in range(0,3):
        df.iloc[:,j]=df.iloc[:,j].fillna(df.iloc[:,j].mean())

print(df.isnull().sum())



df.txKB_PS.resample('60S').sum().plot(title='transmitted_packets_sum')
#df.txKB_PS.resample('D').mean().plot(title='Global_active_power resampled over day', color='red')
plt.tight_layout()
plt.show()

df.txKB_PS.resample('60S').mean().plot(title='transmitted_packets_mean', color='red')
#print(df.txKB_PS.resample('D').mean())
plt.tight_layout()
plt.show()

cols = [0, 1, 2, 3]
i = 1
groups=cols
values = df.resample('60S').mean().values
# plot each column
plt.figure(figsize=(15, 10))
for group in groups:
	plt.subplot(len(cols), 1, i)
	plt.plot(values[:, group])
	plt.title(df.columns[group], y=0.75, loc='right')
	i += 1
plt.show()

df.rxPackets_PS.resample('5H').mean().plot(color='y', legend=True)
df.txPackets_PS.resample('5H').mean().plot(color='r', legend=True)
df.rxKB_PS.resample('5H').mean().plot(color='b', legend=True)
df.txKB_PS.resample('5H').mean().plot(color='g', legend=True)
plt.show()


data_returns = df.pct_change()
sns.jointplot(x='txPackets_PS', y='txKB_PS', data=data_returns)
plt.show()