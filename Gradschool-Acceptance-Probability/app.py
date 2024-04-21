#data pre-processing using pandas
import pandas as pd

data = pd.read_csv('deep-learning-projects/Gradschool-Acceptance-Probability/gpascore.csv')

#check number of empty cells 
#print(data.isnull().sum())
# #fill the empty data
# data.fillna()

#drop null, empty data
data = data.dropna()

y_data = data['admit'].values
x_data = []

for i, rows in data.iterrows():
   x_data.append([rows['gre'], rows['gpa'], rows['rank']])
    

import tensorflow as tf
import numpy as np

#deep learning model using keras 
#For activation functions, we can use sigmoid, tanh, relu, softmax...etc
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation ="tanh"),
    tf.keras.layers.Dense(1, activation ="sigmoid")
])


#for optimizer function, we can use adam, adagrad, adadelta,rmsprop,sgd..etc 
#if the output we are looking for is between 0 and 1 like probability, most of time, we use binary_crossentropy
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])


#learning
#numpy array or tensor
model.fit(np.array(x_data),np.array(y_data),epochs=1000)


# prediction
probability = model.predict([[750,3.70,3]])
print(probability)