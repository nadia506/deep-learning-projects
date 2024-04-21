import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt #to visualize

# ((train_x, train_y), (text_x, test_y))
(train_x, train_y), (test_x, test_y)= tf.keras.datasets.fashion_mnist.load_data()


# if want to condense all data to be in between 0 and 1
# train_x = train_x/255.0
# test_x = test_x/255.0


train_x.reshape((train_x.shape[0],28,28,1))
test_x.reshape((test_x.shape[0],28,28,1))


#[T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot]
class_names= ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#we use sigmoid when binary
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape = (28,28,1) ), #32 feature and kernel size, padding if needed + inputshape needs to be (28,28,3) when color images
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape = (28,28,1) ), 
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])


# model.summary()

model.compile(loss ="sparse_categorical_crossentropy", optimizer = 'adam', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=5)

#if I want to see the loss val and accuracy every round
# model.fit(train_x, train_y, validation_data=(test_x,test_y ), epochs=5)

score = model.evaluate(test_x, test_y)
print(score)
