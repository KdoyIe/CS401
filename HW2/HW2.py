
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from csv import reader
import matplotlib.pyplot as plt


# In[2]:

# Attempt at loading in txt file and converting to csv within code

#def load_csv(filename):
#    file = open(filename, "rt")
#    lines = reader(file)
#    dataset = list(lines)
#    return dataset
#
#
#filename = 'train.txt'
#data = load_csv(filename)


# In[3]:


train_data = pd.read_csv('train.csv', sep=',', header=None)
dataset = train_data.values
source = dataset[:, :-1]
target = dataset[:, len(dataset[0])-1]


# In[4]:

# Attempt at loading in txt file and converting to csv within code

#def load_csv(filename):
#    file = open(filename, "rt")
#    lines = reader(file)
#    dataset = list(lines)
#    return dataset
#
#
#filename = 'test.txt'
#data = load_csv(filename)


# In[7]:


test_data = pd.read_csv('test.csv', sep=',', header=None)
dataset2 = test_data.values
source2 = dataset2[:, :]


# In[8]:


model = DecisionTreeRegressor()  # DecisionTreeClassifier()
model.fit(source, target)
prediction = model.predict(source2)
file = open("test-labels.txt", "w")
file = open("test-labels.txt", "a")

for i in range(len(prediction)):
    file.write(str(int(prediction[i])))
    file.write("\n")

file.close()


# In[9]:


#data = data_train.values
#df = data[:, :]
#y = data[:, len(data[0])-1]
#
#
## create training and testing vars
#X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)
#print(X_train.shape, y_train.shape)
#print(X_test.shape, y_test.shape)
#
## fit a model
#lm = linear_model.LinearRegression()
#model = lm.fit(X_train, y_train)
#predictions = lm.predict(X_test)
#
## The line / model
#plt.scatter(y_test, predictions)
#plt.xlabel("True Values")
#plt.ylabel("Predictions")
#
#print("Score:", model.score(X_test, y_test))
