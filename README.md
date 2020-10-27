# GAN-UPSAMPLING-LAYER
#Increasing the size of an input image which is a part of the Generator in a GAN using the KERAS UpSamplimg@D

#Upsampling with GANS
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from PIL import Image
from numpy import asarray 
from numpy import reshape

##Upsampling is adding information to an image to increase its size and then it is followed by a normal convolution laye
#to decipher the information that has been added 
#It doubles the input dimensions of the image 
#The layer transforms from salient feautures to more dense feautures and works by repating rows and columns 

##Loading a sample image, any image of your choice 
image = Image.open('9.jpg')
#Converting the image into a numpy array
data = asarray(image) 
#print(data)
#Number of dimensions
print(data.ndim)
#Sape of the array
print(data.shape)
#output is (1476, 980, 3)

##Importing and using the UpSample2D from keras in a model
from keras.layers import UpSampling2D
from keras.models import Sequential


#Reshape the data to turn it into a format that we can input into the model indicating that we are only inputting only one image

data = data.reshape((1,1476, 980, 3))
#print(data)

#Define model
model = Sequential()
model.add(UpSampling2D(input_shape = (1476, 980 ,3)))
#summarize the model
model.summary()

#make a prediction of the model 
yhat = model.predict(data)
##shape of the yhat and the initial shape
print(yhat.shape)
print(data.shape)
# output is (1, 2952, 1960, 3)
##Lets reshape the data for easy printing
yhat = yhat.reshape((2952, 1960, 3))

#summarize output
#print(yhat)

##When you want to alter the height and width differently 
# example of using different scale factors for each dimension
#model.add(UpSampling2D(size=(2, 3)))

##The output image yhat that we got is double the input (data) that we put in the model, however for the UpSampling2D
#to make sense it must be followed by a CONV layer
