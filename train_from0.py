from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt
import threading
import pickle
import plot as p

start = time.time()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# dimensions of our images.
img_width, img_height = 64, 64

train_data_dir = r'shoe-dataset-color-mod/train'
validation_data_dir = r'shoe-dataset-color-mod/validation'
epochs = 50
batch_size = 30
classes = 15

tf.device('/device:GPU:0')

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

def create_model():
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.8))
	model.add(Dense(classes, activation='softmax'))

	adam = Adam(lr=0.0001)
	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

	return model

model = create_model()

# this is the augmentation configuration we will use for training
# train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_datagen = ImageDataGenerator()

# this is the augmentation configuration we will use for testing:
# only rescaling
# test_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='categorical')

model.fit_generator(train_generator, steps_per_epoch=(train_generator.samples // batch_size), epochs=epochs, validation_data=validation_generator, validation_steps=(validation_generator.samples // batch_size), verbose = 1)

end = time.time()

print(train_generator.class_indices)
print('Training time : ', (end - start) // 60, ' minutes')

model.save('shoe_model_color_scratch_mod_1.h5')

# summarize history for accuraccy
p.gettraingraph(model, 'acc', 1, 'scratch')

# summarize history for loss
p.gettraingraph(model, 'loss', 1, 'scratch')



