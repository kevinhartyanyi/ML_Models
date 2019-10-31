import numpy as np
from tqdm import tqdm

# Distance functions

# calculate the Euclidean distance between two vectors
def L2_distance(a, b):
	return np.linalg.norm(a-b,ord=2,axis=1)

# calculate the Manhattan distance between two vectors
def L1_distance(a, b):
	return np.linalg.norm(a-b,ord=1,axis=1)

class K_Nearest_Neighbour:
	def __init__(self):
		pass

	def train(self, X, y, k=1):
		# Simply remember all the training data
		self.Xtr = X
		self.ytr = y
		# Number of neighbours
		self.k = k

	def predict(self, X):
		num_test = X.shape[0]
		Ypred = np.zeros(num_test, dtype = self.ytr.dtype)		
		# loop over all test rows
		for i in tqdm(range(num_test)):
			# Chosse distance function
			distance = L2_distance(self.Xtr, X[i])
			k_nearest_neighbour = np.argpartition(distance,self.k)[:self.k] # Get the first k indexes with the smallest distance
			k_predicted = self.ytr[k_nearest_neighbour] # Get classes for the k nearest neighbour
			Ypred[i] = np.bincount(k_predicted).argmax()# Get the most frequently occured class 

		return Ypred


# k-nearest neighbors on the Iris Flowers Dataset
from csv import reader

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup




# Test the kNN on the Iris Flowers dataset
filename = 'iris.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)

train = dataset[:int((len(dataset)/2))]
val   = dataset[int((len(dataset)/2)):int((3*len(dataset)/4))]
test  = dataset[int((3*len(dataset)/4)):]

X_train = np.array([x[:-1] for x in train])
y_train = np.array([x[-1] for x in train])

X_val = np.array([x[:-1] for x in val])
y_val = np.array([x[-1] for x in val])

# evaluate algorithm
num_neighbors = 5
K_NN = K_Nearest_Neighbour()
K_NN.train(X_train,y_train, k=num_neighbors)

predicted = K_NN.predict(X_val)
accuracy = (predicted == y_val).mean()
print('Mean Accuracy: %.3f%%' % (accuracy*100))


# Mnist dataset
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def make_flat(x): # From 2d to 1d
	return x.reshape(x.shape[0],-1)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Only use a smaller amount of the whole dataset, because it runs slowly
x_train, y_train = x_train[:1000], y_train[:1000]
x_test, y_test = x_test[:100], y_test[:100]

x_train = make_flat(x_train) 
x_test = make_flat(x_test) 


k_nn = K_Nearest_Neighbour()
k_nn.train(x_train, y_train,k=3)

predicted = k_nn.predict(x_test)

accuracy = accuracy_score(y_test, predicted)
print('Mean Accuracy: %.3f%%' % (accuracy*100))

