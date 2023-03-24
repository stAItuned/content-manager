---
author: Marcello Politi
cover: https://unsplash.com/@pietrozj?utm_source=medium&utm_medium=referral
date: null
language: English
meta: TensorFlow CNN for Multilabel Image Classification Task
target: Medium
title: TensorFlow CNN for Multilabel Image Classification Task
topics:
- Machine Learning
- Deep Learning
- Data Science
- Python
---

### [Cover](https://unsplash.com/@pietrozj?utm_source=medium&utm_medium=referral)
# TensorFlow CNN for Intel Image Classification Task

### Set up a simple CNN with TensorFlow and use data augmentation

Convolutional Neural Networks (CNN) were born in 1990 from the research of Yann LeCun and his team based on the functioning of the visual cortex of the human brain [1][2]. Thanks to the excellent performance that they have been able to obtain, especially in the field of image recognition, even today CNNs are considered the \"state of the art\" in pattern and image recognition. Yann LeCun received, in 2019, the Turing Prize for his work.

Today to set up a convolutional network and get good results is relatively simple and painless. We will see in this brief guide how to use such networks to solve the Intel Image Classification task that you can find at the following link: [https://www.kaggle.com/puneet6060/intel-image-classification](https://www.kaggle.com/puneet6060/intel-image-classification).

### Dataset

Puneet Bansal. (2019, January). Intel Image Classification, Version 2. Retrieved November 16, 2021, from [https://www.kaggle.com/rtatman/r-vs-python-the-kitchen-gadget-test](https://www.kaggle.com/puneet6060/intel-image-classification).

### Data preparation

The first thing to do after downloading the zip file of the dataset and extracting it, is to organize all the images. The training images are all in the \"seg_train\" folder. We will create a new folder \"training\" that contains the subfolders \"train\" and \"val\". Let's then write our code to split the images from \"seg_train\" in these two new subfolders.

The _split_data_ function we are going to define will have 3 input parameters, the path to the original folder and the paths to the two new sub-folders.

![Split your data into T_rain and Val folders_](https://miro.medium.com/1*cD4uLNd35kMlJ-kMGeczGQ.png)

Now that we have cleaned up our file system we can use our data to actually create the training, validation and test set to feed to our network. TensorFlow provides us with the _ImageDataGenerator_class to write basic data processing in a very simple way.

The training set preprocessor will perform a scaling of the input image pixels dividing them by 255. The _rotation_range_and _width_shift_range_do a bit of data augmentation by modifying some characteristics of the images. Notice that the preprocessor of the validation data has no data augmentation features because we want to leave it unchanged to better validate our model.

Afterwards, we use these processors to read data from the directory with the _flow_from_directory_function. Noteice that this function can automatically figure out the label of each image because it will label as _forest_all images in the _forest_folder etc\u2026

The other things that need to be specified are the path to the images, the size of the images, the 3 RGB channels, the data shuffle, the batch sizes and specify that we are talking about categories.

![Use generators to create the actual datasets](https://miro.medium.com/1*urp9L-OtNSppMJNOxAXHJA.png)

### Model Definition

Finally, we move on to defining the convolutional model. To keep this guide simple the model will be formed only by an _Input layer_that defines the size of the input image. Then there will be a couple of _convolutional layers_followed by _max-pooling_ layers. In the end, two _dense layers_, where the number of output neurons is equal to the number of classes to be classified so that the softmax function returns a probability distribution. (The _Flatten_ layer is used to flatten the multi-dimensional input tensors into a single dimension)

![Let's define the deep learning model](https://miro.medium.com/1*_N5QvYVR_cQwSo2netborw.png)

**Training**

Import the necessary libraries :

![Import the necessary libraries](https://miro.medium.com/1*jYy-wTG0xfMDS-WGjYlI-g.png)

In the training step, we are going to use a callback _ModelCheckPoint_that allows us from time to time to save the best model (evaluated on the validation loss) found at each epoch. The _EarlyStopping_callback instead is used to interrupt the training phase if after a _patience=x_ times there was no improvement. We compile and fit the model as usual. Remember to include the two callbacks.

![Training phase with callbacks definition](https://miro.medium.com/1*Cu21xdwn5CRHYxZ3WeuaIQ.png)

### Evaluating

Now let's load the best model we saved. We can check again our model architecture using the _summary()_function. Let's then evaluate this model on the validation set and then on the test set!

![Model evaluation](https://miro.medium.com/1*-UWqBn0SPfk5NucHwhZbFQ.png)

### Predicting

Now that the model has been trained and saved we can use it to predict new images!
In the function _predict_with_model_we must first do some boring pre-processing steps in order to resize the input image to make it 150x150 so that it can be fed to the network.
The predict function will return the probability distribution of the various classes, and with _argmax_function we return the most probable class. We can use the dictionary MAPPING to convert the obtained number to the final label!

![Let's predict new images!](https://miro.medium.com/1*hmd-QtK_i3YiUbif0Oh-zA.png)

## The End

Marcello Politi

---

## Bibliography

[1] Y. LeCunn e team: Handwritten Digit Recognition With A Back-Propagation Network, NeurIPS conference,(1989)

[2] David H. Hubel: Our First Paper, on Cat Cortex, Oxford website,(1959)"