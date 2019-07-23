from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras import optimizers, Model
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
import plot as p
import time

start = time.time()
# path to the model weights files.
weights_path = '../keras/examples/vgg16_weights.h5'
top_model_weights_path = 'shoe_weight_shape_vgg16_1.h5'
top_model_path = 'shoe_model_shape_vgg16_1.h5'
# dimensions of our images.
img_width, img_height = 64, 64

train_data_dir = 'shoe-dataset-shape/train'
validation_data_dir = 'shoe-dataset-shape/validation'
epochs = 50
batch_size = 30
classes = 9

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# build the VGG16 network
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape = input_shape)
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
# top_model = Sequential()
# top_model.add(Flatten(input_shape=model.output_shape[1:]))
# top_model.add(Dense(256, activation='relu'))
# top_model.add(Dropout(0.5))
# top_model.add(Dense(classes, activation='softmax'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
# top_model.load_weights(top_model_weights_path)

top_model = load_model(top_model_path)

# add the model on top of the convolutional base
# model.add(top_model)
model = Model(input= base_model.input, output= top_model(base_model.output))

# set the first 15 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:11]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
# model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# prepare data augmentation configuration
# train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
# test_datagen = ImageDataGenerator(rescale=1. / 255)

train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')

# fine-tune the model
model.fit_generator(train_generator, 
    steps_per_epoch=train_generator.samples // batch_size, 
    epochs=epochs, 
    validation_data=validation_generator, 
    validation_steps=validation_generator.samples // batch_size, 
    verbose = 1)

end = time.time()

model.save('shoe_model_shape_finetuned_1.h5')

print(train_generator.class_indices)
print('Training time : ', (end - start) // 60, ' minutes')

# summarize history for accuraccy
p.gettraingraph(model, 'acc', 1)

# summarize history for loss
p.gettraingraph(model, 'loss', 1)


