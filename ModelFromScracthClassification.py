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

#Model

initializer = initializers.glorot_normal()

def my_CNN(input_shape, num_classes, regl2 = 0.001):
  model = models.Sequential()

  #Convolutional layer 1
  model.add(layers.Conv2D(32, (3,3), input_shape = input_shape, kernel_initializer=initializer, kernel_regularizer=regularizers.l2(regl2)))
  model.add(layers.MaxPooling2D((2,2), strides = 2))
  model.add(layers.Activation('relu'))
  model.add(layers.BatchNormalization())

  #Convolutional layer 2
  model.add(layers.Conv2D(128, (1,1), padding = 'same', kernel_initializer=initializer, kernel_regularizer=regularizers.l2(regl2)))
  model.add(layers.MaxPooling2D((2,2), strides = 2, padding = 'same'))
  model.add(layers.Activation('relu'))
  model.add(layers.BatchNormalization())

  #Convolutional layer 3
  model.add(layers.Conv2D(128, (3,3), padding = 'same', kernel_initializer=initializer, kernel_regularizer=regularizers.l2(regl2)))
  model.add(layers.MaxPooling2D((2,2), strides = 2, padding = 'same'))
  model.add(layers.Activation('relu'))
  model.add(layers.BatchNormalization())

  #Convolutional layer 4
  model.add(layers.Conv2D(256, (1,1), padding = 'same', kernel_initializer=initializer, kernel_regularizer=regularizers.l2(regl2)))
  model.add(layers.MaxPooling2D((2,2), strides = 2, padding = 'same'))
  model.add(layers.Activation('relu'))
  model.add(layers.BatchNormalization())

  #Convolutional layer 5
  model.add(layers.Conv2D(64, (3,3), kernel_initializer=initializer, kernel_regularizer=regularizers.l2(regl2)))
  model.add(layers.MaxPooling2D((2,2), strides = 2))
  model.add(layers.Activation('relu'))
  model.add(layers.BatchNormalization())

  #Convolutional layer 6
  model.add(layers.Conv2D(32, (3,3), kernel_initializer=initializer, kernel_regularizer=regularizers.l2(regl2)))
  model.add(layers.MaxPooling2D((2,2), strides = 2))
  model.add(layers.Activation('relu'))
  model.add(layers.BatchNormalization())

  #Flatten layer
  model.add(layers.Flatten())

  #Dense layer 1
  model.add(layers.Dense(128, activation='relu', kernel_initializer=initializer,
                         kernel_regularizer=regularizers.l2(regl2)))
  model.add(layers.Dropout(0.5))
  model.add(layers.BatchNormalization())

  #Dense layer 2
  model.add(layers.Dense(64, activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizers.l2(regl2)))
  model.add(layers.Dropout(0.5))
  model.add(layers.BatchNormalization())

  #Dense layer 3
  model.add(layers.Dense(32, activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizers.l2(regl2)))
  model.add(layers.Dropout(0.5))
  model.add(layers.BatchNormalization())

  #Output layer
  model.add(layers.Dense(num_classes, activation='softmax', name='OutputLayer'))

  return model
  
#EXPLAINABLE AI 
#Print the feature maps
model = Model(inputs=cnn.inputs, outputs=cnn.layers[33].output)
# get feature map for first hidden layer
image_x=x_train_224[1] 
label_x=y_train[1]
image_x = expand_dims(image_x, axis=0)
feature_maps = model.predict(image_x)
print(feature_maps.shape)
square = 2
ix = 1
for _ in range(square):
	for _ in range(square):
		# specify subplot and turn of axis
		ax = plt.subplot(square, square, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		plt.imshow(feature_maps[0, :, :, ix-1],cmap='gray')
		ix += 1
# show the figure
plt.show()
#feature_maps = model.predict()

for layer in cnn.layers:
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# get filter weights
	filters, biases = layer.get_weights()
	print(layer.name, filters.shape)

#Callbacks function for the model 
tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
    baseline=None, restore_best_weights=False
)
tf.keras.callbacks.ModelCheckpoint(
    filepath, monitor='val_loss', verbose=0, save_best_only=False,
    save_weights_only=False, mode='auto', save_freq='epoch'
)
reduce_learning=tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss',
    factor=0.2,
    patience=2,
    verbose=1,
    mode='auto',
    min_delta=0.0001,
    cooldown=2,
    min_lr=0)

callbacks_y = [
    TensorBoard(update_freq=521),
    EarlyStopping(monitor='val_accuracy', patience=2),
    ModelCheckpoint("my_cnn_model.h5",save_best_only=True),
    reduce_learning
]

#Compile the model 
cnn.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizers.Adam(0.0001),
    metrics=['acc'])

#Train the model 
NB_EPOCHS = 100

history = cnn.fit(
    augmented_data,
    y_train,
    epochs=NB_EPOCHS,
    validation_split=0.1,
    shuffle=True, 
    batch_size=32
)
test_loss, test_acc = cnn.evaluate(x_test_224, y_test, verbose=1)

#Showe graphs 
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

y_pred = np.argmax(cnn.predict(x_test_224), axis=1).astype('int')
print(y_pred)

cnn.evaluate(x_test_224, y_test)

classes = ['Normal', 'Pneumonia', 'Covid-19']

confusion_mtx = tf.math.confusion_matrix(y_test, y_pred) 
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, xticklabels=classes, yticklabels=classes, 
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

