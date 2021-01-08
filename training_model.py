
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import  ModelCheckpoint
from keras.initializers import  RandomNormal
import os
import cv2
from keras.preprocessing.image import ImageDataGenerator

def prepare_Data(path="F:/Final Project VienAI/Facial_expression/fer2013",
                 image_shape = (48, 48)):

    TRAINING_DATA_PATH = os.path.join(path, 'Training')
    TESTING_DATA_PATH = os.path.join(path, 'PrivateTest')

    x_train = [] # is the training data set
    y_train = [] # is the set of labels to all the data in x_train
    x_test = []
    y_test = []

    label_id = 0

    num_classes = len(os.listdir(TRAINING_DATA_PATH)) # get number of classes 7
    for label in os.listdir(TRAINING_DATA_PATH): # get label in training path

        # Read training data
        for img_file in os.listdir(os.path.join(TRAINING_DATA_PATH, label)):
            img = cv2.imread(os.path.join(TRAINING_DATA_PATH, label, img_file)) # read all image in training path
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img = cv2.resize(gray, image_shape) # resize all image with size = 48 x 48
            # img = np.mean(img, axis=3)
            # img = img.reshape(img.shape)
            x_train.append(gray) # append all training image in x_train

            y = np.zeros(num_classes) # create one hot vector with dimension of 7
            # print(y)
            y[label_id] = 1
            y_train.append(y)
        # Read testing data
        for img_file in os.listdir(os.path.join(TESTING_DATA_PATH, label)):
            img = cv2.imread(os.path.join(TESTING_DATA_PATH, label, img_file))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img = cv2.resize(gray, image_shape)
            x_test.append(gray)
            # x_test = x_test.reshape(x_test.shape[0], 48, 48 ,1)
            y = np.zeros(num_classes)
            y[label_id] = 1
            y_test.append(y)
        label_id += 1
    x_train = np.array(x_train)
    x_train = x_train.reshape(x_train.shape[0],48,48, 1)
    x_test = np.array(x_test)
    x_test = x_test.reshape(x_test.shape[0], 48, 48,1)
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

# input_shape = (48, 48, 1)

x_train, y_train,x_test, y_test = prepare_Data()
# print("Number of images in Training set:", len(x_train))
# print("Number of images in Test set:", len(x_test))
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

model = Sequential()

# 1st convolution layer
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1),
                 bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1),
                 bias_initializer=RandomNormal(stddev=1), kernel_initializer=RandomNormal(stddev=1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Dropout(0.25))

# 3rd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1),
                 kernel_initializer=RandomNormal(stddev=1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1),
                 kernel_initializer=RandomNormal(stddev=1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Dropout(0.25))

# 5th convolution layer
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1),
                 kernel_initializer=RandomNormal(stddev=1)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1),
                 kernel_initializer=RandomNormal(stddev=1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Dropout(0.25))

# 7th convolution layer
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1),
                 kernel_initializer=RandomNormal(stddev=1)))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', bias_initializer=RandomNormal(stddev=1),
                 kernel_initializer=RandomNormal(stddev=1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
# Fully connected layers
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
#
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,
    zoom_range = 0.05)  # zoom images in range [1 - zoom_range, 1+ zoom_range]

datagen.fit(x_train)
#
filepath = "F:/Final Project VienAI/facial_expressionn.hdf5"
checkpointer = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
# tensorboard = TensorBoard(log_dir='./logs')
epochs=250
history = model.fit_generator(datagen.flow(x_train, y_train,
                    batch_size=32),
                    epochs=250,
                    validation_data=(x_test, y_test),
                    steps_per_epoch=x_train.shape[0]/32,
                    callbacks=[checkpointer])

