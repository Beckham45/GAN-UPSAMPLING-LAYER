#Simple Generator Model With the UpSampling2D Layer

#Imporing essential ML libraries
import numpy as np 
from numpy import asarray
import PIL
from PIL import Image
import pandas as pd 

##UpSampling does not perform any learning and to be useful it must be followed by a CONV layer so that the it can understand the 
#the informations of the doubled dimensions
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Reshape, UpSampling2D

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import UpSampling2D
from keras.layers import Conv2D
# define model
model = Sequential()
# define input shape, output enough activations for for 128 5x5 image
model.add(Dense(128 * 5 * 5, input_dim=100))
# reshape vector of activations into 128 feature maps with 5x5
model.add(Reshape((5, 5, 128)))
# double input from 128 5x5 to 1 10x10 feature map
model.add(UpSampling2D())
# fill in detail in the upsampled feature maps and output a single image
model.add(Conv2D(1, (3,3), padding= 'same' ))
# summarize model
model.summary()



