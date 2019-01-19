import numpy as np
import pandas as pd
import matplotlib.pyplot as py
from sklearn.preprocessing import StandardScaler
#Read the csv training data
d0 = pd.read_csv('C:\\Users\Maithreyi.Prabhu\Downloads\mnist_train.csv')
print(d0.head(5))
l = d0['label']
d = d0.drop("label",axis=1)
print(d.head(5))
print(l)
print(d.shape,l.shape) #(42000, 784) (42000,)
py.figure(figsize=(10,10))
#idx = 100
#grid_data = d.iloc[idx].as_matrix().reshape(28,28)
#py.imshow(grid_data,interpolation="none",cmap="gray")
#py.show()
#print(l[idx])
labels = l.head(15000)
data = d.head(15000)
print(data.shape,labels.shape) #(15000, 784) (15000,)


from sklearn.preprocessing import StandardScaler
standardized_data = StandardScaler().fit_transform(data)
print(standardized_data)
sample_data = standardized_data
print(sample_data)

# Matrix multiplication using numpy. Multiply sample_data matrix with a transpose of sample_data matrix A^T * A
CoVariance_Matrix = np.matmul(sample_data.T,sample_data)
print(CoVariance_Matrix)

#Find top 2 eigen values and corresponding eigen vector projecting into 2-D space
from scipy.linalg import eigh

#Eigh function will return values in ascending order
values,vectors = eigh(CoVariance_Matrix,eigvals=(782,783))
print(values)
print(vectors, vectors.shape)
