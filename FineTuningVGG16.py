import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, regularizers, initializers
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
import seaborn as sns
import scipy.ndimage as ndimage
from keras import callbacks
from keras.utils import to_categorical
from keras.models import Model
from numpy import expand_dims

#We split the execution from the cell before because of excessive RAM usage of Colab

x_train_224 = np.load('x_train_224.npy')
y_train = np.load('y_train.npy')

#Print the first 25 images of the train dataset

print("x_train shape:", x_train_224.shape, "y_train shape:", y_train.shape)
class_names = ["normal", "pneumonia", "COVID-19"]

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train_224[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[y_train[i]])
plt.show()

#Data augmentation
def random_rotate_image(image):
  image = ndimage.rotate(image, np.random.uniform(-50, 50), reshape=False)
  return image

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(random_rotate_image(x_train_224[i]))
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[y_train[i]])

#Print the first 25 rotated images of the train dataset
plt.show()

def blurred_image(image):
  image = ndimage.gaussian_filter(image, 3)
  return image
  
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(blurred_image(x_train_224[i]), cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[y_train[i]])

#Print the first 25 blurred images of the dataset
plt.show()

def noise_image(image):
  x = len(np.unique(image))
  x = 2 ** np.ceil(np.log2(x))
  noisy = np.random.poisson(image * x) / float(x)
  return noisy

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(noise_image(x_train_224[i]), cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[y_train[i]])

#Print the first 25 noisy images of the dataset
plt.show()

def flip_image(image):
  img=np.array(image)
  x=np.flip(img,(0,1))
  return x

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(flip_image(x_train_224[i]), cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[y_train[i]])

#Print the first 25 flipped images
plt.show()

#Multiple Data augmentation
data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("vertical"),
  layers.experimental.preprocessing.RandomRotation((-0.15, 0.15)),
  layers.GaussianNoise(stddev=0.10),
  layers.experimental.preprocessing.RandomContrast(0.20)
])
augmented_data=np.empty_like(x_train_224)
augmented_data = data_augmentation(x_train_224)

#Load the test set 
x_test_224 = np.load('x_test_224.npy')
y_test = np.load('y_test.npy')

#Create augmented Dataset 

rotated_data=np.empty_like(x_train_224)
for i in range(3000):
  rotated_data[i]=random_rotate_image(x_train_224[i])
flipped_data=np.empty_like(x_train_224)
for i in range(3000):
  flipped_data[i]=flip_image(x_train_224[i])
blurred_data=np.empty_like(x_train_224)
for i in range(3000):
  blurred_data[i]=blurred_image(x_train_224[i])  
noise_data=np.empty_like(x_train_224)
for i in range(3000):
  noise_data[i]=noise_image(x_train_224[i])    


#Create augmented validation sets
rotated_val_set=np.empty_like(x_test_224)
for i in range(300):
  rotated_val_set[i]=random_rotate_image(x_test_224[i])  
flipped_val_set=np.empty_like(x_test_224)
for i in range(300):
  flipped_val_set[i]=flip_image(x_test_224[i])  
noise_val_set=np.empty_like(x_test_224)
for i in range(300):
  noise_val_set[i]=noise_image(x_test_224[i])  

blurred_val_set=np.empty_like(x_test_224)
for i in range(300):
  blurred_val_set[i]=blurred_image(x_test_224[i])  

#Setup Transfer Learning on VGG16 model 
def VGG16():
  return tf.keras.applications.VGG16(include_top=False,input_shape=(224,224,3))#exclde fully connected layers 

train_features = vgg.predict(np.array(flipped_data), batch_size=64, verbose=1)
validation_features = vgg.predict(np.array(flipped_val_set), batch_size=64, verbose=1)
test_features = vgg.predict(np.array(x_test_224), batch_size=64, verbose=1)#no data augm on test set 
#Now we have to flatten features before giving them to classifier
train_features_flat = np.reshape(train_features, (3000, 7*7*512))
validation_features_flat = np.reshape(validation_features, (300, 7*7*512))
test_features_flat = np.reshape(test_features, (300, 7*7*512))
#define the classifier for fine tuning the model :
NB_TRAIN_SAMPLES = train_features_flat.shape[0]
NB_EPOCHS = 100
num_classes=3
#classifier with leaky relu 
classifier = models.Sequential()
classifier.add(layers.Dense(512, activation=tf.keras.layers.LeakyReLU(), input_dim=(7*7*512)))
classifier.add(layers.Dense(3, activation='softmax'))
#compile the model
classifier.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.Adam(0.0001),
    metrics=['acc'])
#some utility callbacks
reduce_learning = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    verbose=1,
    mode='auto',
    min_delta=0.0001,
    cooldown=2,
    min_lr=0)

eary_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=7,
    verbose=1,
    mode='auto')
callbacks_x = [reduce_learning, eary_stopping]
#One hot encoding
train_Y_one_hot = to_categorical(y_train)
val_Y_one_hot=to_categorical(y_test)
test_Y_one_hot = to_categorical(y_test)
#train
history = classifier.fit(
    train_features_flat,
    train_Y_one_hot,
    epochs=NB_EPOCHS,
    validation_data=(validation_features_flat,val_Y_one_hot),
    callbacks=callbacks_x
)
test_loss, test_acc = classifier.evaluate(test_features_flat,val_Y_one_hot, verbose=2)

#print activation map filters (Explainable AI)
filters, biases = vgg.layers[1].get_weights()

# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# plot first few filters
n_filters, ix = 6, 1

for i in range(n_filters):
	# get the filter
	f = filters[:, :, :, i]
	# plot each channel separately
	for j in range(3):
		# specify subplot and turn of axis
		ax = plt.subplot(n_filters, 3, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		plt.imshow(f[:, :,j], cmap='gray')
		ix += 1

# show the figure
plt.show()

#Trining and validation accuracy and loss graphs 
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')
plt.legend()

plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')

plt.legend()

plt.show()





