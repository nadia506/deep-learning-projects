import numpy as np
import tensorflow as tf
data = open('deep-learning-projects/Composition-AI/data.txt', 'r').read()

#numbering (text to number), utilities
text_element = list(set(data))
text_element.sort()
text_to_num = {}
num_to_text = {}
for i,element in enumerate(text_element):
    text_to_num[element]=i
    num_to_text[i] = element

data_to_num = []
for i in data:
    data_to_num.append(text_to_num[i])



X = []
Y = []
for i in range(0, len(data_to_num)-25):
   X.append(data_to_num[i:i+25])
   Y.append(data_to_num[i+25])

X = tf.one_hot(X,31)
Y = tf.one_hot(Y,31)

#model
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(100, input_shape=(25,31)),
    tf.keras.layers.Dense(31, activation ='softmax')
]
)

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.fit(X,Y,batch_size=64, epochs = 40, verbose=2)



#1. firt input value
#2. predict following letter based on first input value
#3. discard first letter in the input value and add the predicted value at the back
#4. one hot encoding 

music = []

first_input = data_to_num[110:110+25]
first_input = tf.one_hot(first_input,31)
first_input = tf.expand_dims(first_input, axis =0) 

for i in range(200):
    predicted = model.predict(first_input)
    predicted = model = np.argmax(predicted[0])

    #if stuck in the same notes, use this predicted value
    #new_predicted = np.random.choice(text_element, 1, p= predicted[0])
    
    music.append(predicted)

    next_input = first_input.numpy()[0][1:]
    one_hot_num = tf.one_hot(predicted, 31)

    first_input = np.vstack([next_input, one_hot_num.numpy()])
    first_input = tf.expand_dims(first_input, axis = 0)

music_text = []
for i in music:
    music_text.append(num_to_text[i])
    