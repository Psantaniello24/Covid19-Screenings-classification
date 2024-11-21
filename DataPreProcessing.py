import cv2
import os
import numpy as np
import random

label_mapping = {'normal': 0, 'pneumonia': 1,  'COVID-19': 2}
train_filepath = 'train_COVIDx5.txt'
test_filepath = 'test_COVIDx5.txt'

# Load in the train and test files
file = open(train_filepath, 'r')
trainfiles = file.readlines()
file = open(test_filepath, 'r')
testfiles = file.readlines()

print('Total samples for train: ', len(trainfiles))
print('Total samples for test: ', len(testfiles))

#Building of the labels arrays of the initial dataset

x_train_128 = []
x_train_224 = []
x_train_229 = []

x_test_128 = []
x_test_224 = []
x_test_229 = []

y_train = []
y_test = []

def build_label_array(y_array, val):
  if val: 
    for i in range(len(trainfiles)):
      train_i = trainfiles[i].split()
      y_array.append(label_mapping[train_i[2]])
  else:
    for i in range(len(testfiles)):
      test_i = testfiles[i].split()
      y_array.append(label_mapping[test_i[2]])

build_label_array(y_train, True)
build_label_array(y_test, False)

#We build a (1500, 1000, 500) dataset (Normal, Pneumonia, Covid)
#Execute only ONCE

def build_img_array_train(x_array, y_array, y_array_3000, size, n):
  norm = 0; pneum = 0; covid = 0
  input_size = (size, size)
  for i in range(len(trainfiles)):
      train_i = trainfiles[i].split()
      imgpath = train_i[1]
      if y_array[i] == 0 and norm < 1500:
        img = cv2.imread(os.path.join('train', imgpath))
        img = cv2.resize(img, input_size) # resize
        img = img.astype('float32') / 255.0
        norm = norm + 1
        print(norm, ' su 1500: normal, ', size)
        x_array.append(img)
        y_array_3000.append(y_array[i])
      if y_array[i] == 1 and pneum < 1000:
        img = cv2.imread(os.path.join('train', imgpath))
        img = cv2.resize(img, input_size) # resize
        img = img.astype('float32') / 255.0
        pneum = pneum + 1
        print(pneum, ' su 1000: pneumonia, ', size)
        x_array.append(img)
        y_array_3000.append(y_array[i])
      if y_array[i] == 2 and covid < 500:
        img = cv2.imread(os.path.join('train', imgpath))
        img = cv2.resize(img, input_size) # resize
        img = img.astype('float32') / 255.0
        covid = covid + 1
        print(covid, ' su 500: covid, ', size)
        x_array.append(img)
        y_array_3000.append(y_array[i])

  print('Shape of train images: ', x_array[0].shape)

def build_img_array_test(x_array, size, n):
  input_size = (size, size)
  for i in range(len(testfiles)):
      test_i = testfiles[i].split()
      imgpath = test_i[1]
      img = cv2.imread(os.path.join('test', imgpath))
      img = cv2.resize(img, input_size) # resize
      img = img.astype('float32') / 255.0
      n = n + 1
      print(n, ' su ', len(testfiles), 'test set, ', size)
      x_array.append(img)

y_train_3000_128 = []
y_train_3000_224 = []
y_train_3000_229 = []

build_img_array_test(x_test_128, 128, 0)
build_img_array_test(x_test_224, 224, 0)
build_img_array_test(x_test_229, 229, 0)

build_img_array_train(x_train_128, y_train, y_train_3000_128, 128, 0)
build_img_array_train(x_train_224, y_train, y_train_3000_224, 224, 0)
build_img_array_train(x_train_229, y_train, y_train_3000_229, 229, 0)

shuffle_list = list(zip(x_train_128, x_train_224, x_train_229, y_train_3000_128))

random.shuffle(shuffle_list)

x_train_128, x_train_224, x_train_229, y_train_3000_128 = zip(*shuffle_list)

