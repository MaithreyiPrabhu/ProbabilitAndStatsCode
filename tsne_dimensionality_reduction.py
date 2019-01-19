import numpy as np
import pandas as pd
import matplotlib.pyplot as py
import seaborn as sn
from sklearn.preprocessing import StandardScaler
#Read the csv training data
d0 = pd.read_csv('C:\\Users\Maithreyi.Prabhu\Downloads\mnist_train.csv')
#print(d0.head(5))
l = d0['label']
d = d0.drop("label",axis=1)
#print(d.head(5))
#print(l)
#print(d.shape,l.shape) #(42000, 784) (42000,)
py.figure(figsize=(10,10))
#idx = 100
#grid_data = d.iloc[idx].as_matrix().reshape(28,28)
#py.imshow(grid_data,interpolation="none",cmap="gray")
#py.show()
#print(l[idx])
labels = l.head(40000)
data = d.head(40000)
#print(data.shape,labels.shape) #(15000, 784) (15000,)
from sklearn.preprocessing import StandardScaler
standardized_data = StandardScaler().fit_transform(data)
sample_data = standardized_data
from sklearn.manifold import TSNE
#Picking up 1000 points from standaridized data
data_1000 = standardized_data[0:1000,:]
print(data_1000)
labels_10000 = labels[0:1000]
#Configuring properties  - Number of components - 2, Default perplexity is 30, default number of iterations is 1000
model = TSNE(n_components=2,random_state=0,perplexity=50,n_iter=5000)

tsne_data = model.fit_transform(data_1000)
tsne_data = np.vstack((tsne_data.T, labels_10000)).T
print("TSNE DATA")
print(tsne_data)
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))
print("TSNE DF")
print(tsne_df)
sn.FacetGrid(tsne_df,hue="label", height=6).map(py.scatter,"Dim_1","Dim_2").add_legend()
py.show()




