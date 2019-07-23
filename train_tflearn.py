import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical

import models_configuration as mc
import tensorflow as tf
import math
import os
import cv2
# dimensions of our images.
img_width, img_height = 64, 64

# weights_path = 'models/vgg16_weights.h5'
top_model_weights_path = 'shoe_model_shape_vgg16_3.h5'
train_data_dir = 'shoe-dataset-shape/train/'
validation_data_dir = 'shoe-dataset-shape/validation'
train_feature = 'shoe_features_shape_train.npy'
validation_feature = 'shoe_features_shape_validation.npy'
epochs = 50
batch_size = 30
classes = 9

nb_train_samples = 32365
nb_validation_samples = 7926

tf.device('/device:GPU:0')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras import backend as K
K.set_image_dim_ordering('th')

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)
    
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    train_generator = datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='categorical', shuffle=False)
    predict_size_train = int(math.ceil(train_generator.samples / batch_size))
    # predict_size_train = train_generator.samples // batch_size
    bottleneck_features_train = model.predict_generator(train_generator, predict_size_train, verbose = 1)
    # np.save(open(train_feature, 'wb'), bottleneck_features_train)
    np.save(train_feature, bottleneck_features_train)

    val_generator = datagen.flow_from_directory(validation_data_dir, target_size=(img_width, img_height), batch_size=batch_size,class_mode='categorical', shuffle=False)
    predict_size_validation = int(math.ceil(val_generator.samples / batch_size))
    # predict_size_validation = val_generator.samples // batch_size
    bottleneck_features_validation = model.predict_generator(val_generator, predict_size_validation, verbose = 1)
    np.save(validation_feature, bottleneck_features_validation)
    # np.save(open(validation_feature, 'wb'), bottleneck_features_validation)

def train_top_model():
    train_data = np.load(train_feature)
    # train_data = np.load(open(train_feature,'rb'))
    train_labels = np.array([0] * 4684 + [1] * 2022 + [2] * 3775 + [3] * 4544 + [4] * 3195 + [5] * 4562 + [6] * 2300 + [7] * 2002 + [8] * 5281)

    validation_data = np.load(validation_feature)
    # validation_data = np.load(open(validation_feature,'rb'))
    validation_labels = np.array([0] * 1171 + [1] * 441 + [2] * 949 + [3] * 1121 + [4] * 799 + [5] * 1141 + [6] * 575 + [7] * 409 + [8] * 1320)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    # model.add(Flatten())
    model.add(Dense(256, activation='relu', name = 'fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax', name = 'prediction'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    train_labels = to_categorical(train_labels, classes)
    validation_labels = to_categorical(validation_labels, classes)

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels),
              verbose = 1)
    model.save(top_model_weights_path)


save_bottlebeck_features()
train_top_model()