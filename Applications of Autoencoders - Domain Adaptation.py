# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 07:33:54 2021

@author: abc
"""

"""

Applications of Autoencoders - Domain Adaptation


"""

from matplotlib.pyplot import imshow
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D

#Define size of images
SIZE = 256 

#Create empty list for first image
img_data = []

#READ IMAGE
img = cv2.imread("einstein_original.jpg", 1)   #Change 1 to 0 for gray scale image

#Change image color BGR TO RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Resize our image
img = cv2.resize(img, (SIZE, SIZE))

#Add image into array
img_data.append(img_to_array(img))

#reshape our image
img_array = np.reshape(img_data, (len(img_data), SIZE, SIZE, 3))

#Convert integer value into floating point of image
img_array = img_array.astype("float32")/ 255.

#################################################

#Create empty list for second image
img_data2 = []

#Read image
img2 = cv2.imread("monalisa_original.jpg")

#Change image color BGR To RGB
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

#Resize our image
img2 = cv2.resize(img2, (SIZE, SIZE))

#Add image into array
img_data2.append(img_to_array(img2))

#Reshape our image
img_array2 = np.reshape(img_data2, (len(img_data2), SIZE, SIZE ,3))

#Convert integer value into floating point of image
img_array2 = img_array2.astype("float32")/ 255.


######################################################

#Define Autoencoder model

model = Sequential()
model.add(Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(SIZE, SIZE, 3)))
model.add(MaxPooling2D((2,2),padding='same'))
model.add(Conv2D(8, (3,3), activation="relu", padding="same"))
model.add(MaxPooling2D((2,2), padding="same"))

model.add(MaxPooling2D((2,2), padding="same"))

model.add(Conv2D(8, (3,3), activation="relu", padding="same"))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(8, (3,3), activation="relu", padding="same"))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(32, (3,3), activation="relu", padding="same"))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(3, (3,3), activation="relu", padding="same"))

model.compile(optimizer="adam", loss="mean_squared_error", metrics=['accuracy'])
model.summary()

#Fit the model
model.fit(img_array, img_array2, epochs=5000, shuffle=True)

#print the model
print("Output")

#Predict the model
pred = model.predict(img_array) #Predict model on the same input array

#Let's print and see our model
imshow(pred[0].reshape(SIZE, SIZE, 3), cmap="gray")




