from keras.layers import Convolution2D, UpSampling2D
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf

tf.python.control_flow_ops = tf

# Image transformer
datagen = ImageDataGenerator(
		rescale=1.0/255,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True)

# Get images
X = []
for filename in os.listdir('face_images'):
	X.append(img_to_array(load_img('face_images/'+filename)))
X = np.array(X)

# Set up train and test data
split = int(0.9*len(X))
Xtrain = X[:split]
Xtest = rgb2lab(1.0/255*X[split:])[:,:,:,0]
Xtest = Xtest.reshape(Xtest.shape+(1,))
Ytest = rgb2lab(1.0/255*X[split:])[:,:,:,1:]

# Set up model
N = 5
model = Sequential()
num_maps1 = [4, 8, 16, 32, 64]
num_maps2 = [8, 16, 32, 64, 128]

# Convolutional layers
for i in range(N):
	if i == 0:
		model.add(Convolution2D(num_maps1[i], 3, 3, border_mode='same', subsample=(2, 2), input_shape=(128, 128, 1)))
	else:
		model.add(Convolution2D(num_maps1[i], 3, 3, border_mode='same', subsample=(2, 2)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Convolution2D(num_maps2[i], 3, 3, border_mode='same', subsample=(1, 1)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

# Upsampling layers
for i in range(N):
	model.add(UpSampling2D(size=(2, 2)))
	model.add(Convolution2D(num_maps2[-(i+1)], 3, 3, border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	if i != N-1:
		model.add(Convolution2D(num_maps1[-(i+1)], 3, 3, border_mode='same'))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
	else:
		model.add(Convolution2D(2, 3, 3, border_mode='same'))

# Finish model
model.compile(optimizer='rmsprop',
			loss='mse')

# Generate training data
batch_size = 10
def image_a_b_gen(batch_size):
	for batch in datagen.flow(Xtrain, batch_size=batch_size):
		if batch == None:
			break		
		lab_batch = rgb2lab(batch)
		X_batch = lab_batch[:,:,:,0]
		Y_batch = lab_batch[:,:,:,1:]
		yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)

# Train model
model.fit_generator(
	image_a_b_gen(batch_size),
	samples_per_epoch=1000,
	nb_epoch=15)

# Test model
print model.evaluate(Xtest, Ytest, batch_size=batch_size)
output = model.predict(Xtest)

# Output colorizations
for i in range(len(output)):
	cur = np.zeros((128, 128, 3))
	cur[:,:,0] = Xtest[i][:,:,0]
	cur[:,:,1:] = output[i]
	imsave("colorizations/img_"+str(i)+".png", lab2rgb(cur))
	imsave("colorizations/img_gray_"+str(i)+".png", rgb2gray(lab2rgb(cur)))
