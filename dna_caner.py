import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
data=pd.read_csv('DNA_Dataset_Normalized.csv')
print(data.columns)
print(data.describe())
print(data.info())
print(data.isna().sum())
for i in data.columns.values:
    sn.boxplot(data[i])
    plt.show()
for i in data.columns.values:
    if len(data[i].value_counts()) <=5:
        index=data[i].value_counts().index
        valu=data[i].value_counts().values
        plt.pie(valu,labels=index,autopct='%1.1f%%')
        plt.title(f'PIE CHART INFORMATION FOR {index} in {i} column')
        plt.legend()
        plt.show()

correlation_matrix = data.corr()
sn.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()


plt.figure(figsize=(17, 6))
corr = data.corr(method='spearman')
my_m = np.triu(corr)
sn.heatmap(corr, mask=my_m, annot=True, cmap="Set2")
plt.show()

for i in data.columns.values:
    for j in data.columns.values:
        sn.distplot(data[i],color='black',label=f'{i}')
        sn.distplot(data[j],color='magenta',label=f'{j}')
        plt.xlabel(f'{i}')
        plt.ylabel(f'{j}')
        plt.title(f'Its {i} vs {j}')
        plt.legend()
        plt.show()

for i in data.columns.values:
    for j in data.columns.values:
        sn.histplot(data[i],color='black',label=f'{i}')
        sn.histplot(data[j],color='magenta',label=f'{j}')
        plt.xlabel(f'{i}')
        plt.ylabel(f'{j}')
        plt.title(f'Its {i} vs {j}')
        plt.legend()
        plt.show()

kmeans=KMeans(n_clusters=5,verbose=1,)
kmeans.fit_predict(data)
labels=kmeans.labels_
color=['red','blue','green','magenta','black']
data['cluster']=labels
center=kmeans.cluster_centers_
for k in range(len(color)):
    df=data[data['cluster']==k]
    for i in df.columns.values:
        for j in df.columns.values:
            plt.plot(df[i], color='red', label=f"{i} cluster")
            plt.plot(df[j], color='magenta', label=f"{j} with cluster")
            plt.title(f'Line plot based on the cluster {k}')
            plt.legend()
            plt.show()

            sn.distplot(df[i], label=f"{i} cluster")
            sn.distplot(df[j] == j, label=f"{j} cluster")
            plt.title(f' Dist plot based on the cluster {k}')
            plt.legend()
            plt.show()

            sn.histplot(df[i], label=f"{i} cluster")
            sn.histplot(df[j], label=f"{j} with cluster")
            plt.title(f'plotting based on the cluster {k}')
            plt.legend()
            plt.show()

for i in range(kmeans.n_clusters):
    for j in range(kmeans.n_clusters):
        plt.plot(data['cluster']==i,color='red',label=f"{i} cluster")
        plt.plot(data['cluster']==j,color='magenta',label=f"{j} with cluster")
        plt.title('plotting based on the cluster')
        plt.legend()
        plt.show()

for i in range(kmeans.n_clusters):
    for j in range(kmeans.n_clusters):
        sn.distplot(data['cluster']==i,label=f"{i} cluster")
        sn.distplot(data['cluster']==j,label=f"{j} with cluster")
        plt.title('plotting based on the cluster')
        plt.legend()
        plt.show()

center=kmeans.cluster_centers_
for i in range(5):
    for j in range(5):
        plt.scatter(data[data['cluster']==i].head(25),data[data['cluster']==j].head(25),s=50)
    plt.scatter(center[:,i],center[:,j],color='black',marker='x')
    plt.show()




sn.countplot(data['cluster'])
plt.title('Kmeans clusters')
plt.legend()
plt.show()

print(len(data['cluster']))
print(len(data))

print(len(data['cluster'].value_counts()))
p=data[data['cluster']==0]

pr=kmeans.cluster_centers_
print(pr[:,:])