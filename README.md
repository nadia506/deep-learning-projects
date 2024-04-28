# deep-learning-projects

This repository contains mini projects in deep learning that I have done while studying the concepts using tensorFlow.

## Saving and Loading Models

Below is an example of how to save the entire model to a directory and then load it:

```python
# Save the model
model.save('folder route/model1')
# Load the model
model01 = tf.keras.models.load_model('folder route/model1')
```

Another way is to save checkpoints every epoch:

```python
# call back function
checkpointFunc = tf.keras.callbacks.Modelcheckpoint(
    filepath='chekcpoint/mnist',
    mointor = 'val_acc',
    mode='max'
    save_weights_only = True,
    save_freq='epoch'
)
# add the checkpointFunc in model.fit
model.fit(train_x, train_y, epochs=5, callbacks = [checkpointFunc])
```

---

## Project1. Gradschool Acceptance Prediction Model

[Source Code](https://github.com/nadia506/deep-learning-projects/tree/main/Gradschool-Acceptance-Probability)

For this project, I used synthetic datasets which consists of GRE scores, GPA, the rankings of the graduate school to which applicants applied, and whether or not they got accepted. Based on this data, I create a model that predicts the likelihood of acceptance of graudeate school.

## Project2. Image Classification Model with Convolution Layer (CNN)

[Source Code](https://github.com/nadia506/deep-learning-projects/tree/main/Image-Classification-Clothing)

In this project, I developed a Convoluional Neural network to classify gray-scale images of clothings. I used the open dataset that are already pre-processed provided by keras. I used convolution layers to extract features from the images and pooling layers to retain important information.

## Project3. Cat vs. Dog Image Classification

[Source Code](https://github.com/nadia506/deep-learning-projects/tree/main/Image-Classification-CatDog)

This project features a deep learning model that classifies images as either cats or dogs. I utilized a dataset from Kaggle and preprocessed the data with keras to prepare the data. Additionaly, to prevent overfitting and enhance the model's generalizability, I used On-the-fly image augmentation strategies. The project was done on Google Colaboratory to accommodate the large data size.

## Project4. Composition AI

[Source Code](https://github.com/nadia506/deep-learning-projects/tree/main/Composition-AI)

This project involves a deep learning model that utilizes a Recurrent Neural Network (RNN) to compose classical piano music. The training data comprises classical piano melodies written in ABC notation. For accurate results, the model should be trained for at least 40 epochs. To listen to the music created by the model, you can copy the output into this website: [check out your music](http://www.tradtunedb.org.uk/#/editor)
