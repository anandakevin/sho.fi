import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from keras_preprocessing import image
from keras.utils import to_categorical

# Images Dimensions
img_width, img_height = 224, 224
train_data_dir = 'data/train'
val_data_dir = 'data/validation'
nb_train_samples = 2160
nb_validation_samples = 360
epochs = 40
batch_size = 12

# Build VGG16
model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Training Data Augmentation
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, height_shift_range=0.2, width_shift_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest', rotation_range=20)
val_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, horizontal_flip=True)

# Train Data Feature Extraction
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)
train_features = model.predict_generator(
    train_generator, train_generator.samples // batch_size, verbose=1)
np.save('train_features.npy', train_features)

# Testing Data Feature Extraction
validation_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)
validation_features = model.predict_generator(
    validation_generator, validation_generator.samples // batch_size, verbose=1)
np.save('val_features.npy', validation_features)

# Build Train Data
train_data = np.load('train_features.npy')
train_labels = np.array([])
for i in range(36):
    train_labels = np.append(train_labels, [i]*(train_generator.samples // 36))
train_labels = train_labels.astype(int)
train_labels = to_categorical(train_labels, num_classes=36)

# Build Validation Data
validation_data = np.load('val_features.npy')
validation_labels = np.array([])
for i in range(36):
    validation_labels = np.append(validation_labels, [i]*(validation_generator.samples // 36))
validation_labels = validation_labels.astype(int)
validation_labels = to_categorical(validation_labels, num_classes=36)

# Build FC Layer
fc_model = Sequential()
fc_model.add(Flatten(input_shape=model.output_shape[1:]))
fc_model.add(Dense(512, activation='relu'))
fc_model.add(Dropout(0.2))
fc_model.add(Dense(36, activation='softmax'))

# Adam Optimizer and Cross Entropy Loss
adam = Adam(lr=0.0001)
fc_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

fc_model.fit(train_data, train_labels,
             epochs=epochs,
             batch_size=batch_size,
             validation_data=(validation_data, validation_labels), verbose=1)

fc_model.save('test_model.h5')

