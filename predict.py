from keras.models import Sequential, load_model
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import applications
import cv2
import keras
import numpy as np
import os
from keras.optimizers import Adam
from directory_management import *
from models_configuration import *
# import image_processing as ip
import image_threshold as it
from PIL import ImageFont, ImageDraw, Image

shape_model_path = 'models/shape/shoe_model_shape_scratch_4.h5'
color_model_path = 'models/color/shoe_model_color_scratch_1.h5'

shape_labels = ['ankle boots', 'knee high boots', 'mid-calf boots', 'sandals',
 'flat', 'heels', 'loafer', 'oxford', 'sneakers & athlethic']
color_labels = ['beige', 'black', 'blue', 'brown', 'gold', 'gray', 'green',
 'multicolor', 'orange', 'pink', 'purple', 'red', 'silver', 'white', 'yellow']

img_width, img_height = 64, 64
b, g, r, a = 255, 255, 255, 0

# for feature extraction
# pre_model = applications.VGG16(include_top=False, weights='imagenet')

def load_models():
    shape = load_model(shape_model_path)
    color = load_model(color_model_path)

    adam = Adam(lr=0.0001)
    shape.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    color.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return shape, color

shape_model, color_model = load_models()

def resize_image(event):
    new_width = event.width
    new_height = event.height
    image = copy_of_image.resize((new_width, new_height))
    photo = ImageTk.PhotoImage(image)
    label.config(image = photo)
    label.image = photo #avoid garbage collection

def loadimage(input_image, image_source):
    if image_source == 'camera':
        # for scratch
        img = cv2.resize(input_image, (img_height, img_width))
        img = np.reshape(img, [1, img_height, img_width, 3])

        # for feature extraction
        # cv2.imwrite('temp.jpg', input_image)
        # img = getfeature('temp.jpg')
    elif image_source == 'image':
        # for scratch
        img = cv2.imread(input_image)
        img = cv2.resize(img, (img_height, img_width))
        img = np.reshape(img, [1, img_height, img_width, 3])

        # for feature extraction
        # img = getfeature(input_image)
    return img

def show_top_three(class_prob, pred_type):
    pred_list = np.argsort(class_prob)[0]
    print (pred_list)
    topidx = []
    toplabels = []
    j = 0
    if pred_type == 'shape':
        labels = shape_labels
    elif pred_type == 'color':
        labels = color_labels
    for i in range(-1, -4, -1):
        idx = pred_list[i]
        topidx.append(idx)
        toplabels.append(labels[idx])
        print(topidx[j])
        print(toplabels[j])
        j += 1
    return topidx, toplabels

def getprediction(input_image, image_source, pred_type):
    img = loadimage(input_image, image_source)
    prediction, topidx, toplabels = [], [], []
    # predict result
    if pred_type == 'shape':
        prediction = shape_model.predict(img)
        topidx, toplabels = show_top_three(prediction, pred_type)
        print(prediction)
        return toplabels
    elif pred_type == 'color':
        prediction = color_model.predict(img)
        topidx, toplabels = show_top_three(prediction, pred_type)
        print(prediction)
        return toplabels

#extract feature from image
def getfeature(input_image):
    img = image.load_img(input_image, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = pre_model.predict(x)
    return features
    
def live_processing():
    cap = cv2.VideoCapture(0)
    cam_width = cap.get(3)  # float
    cam_width = int(cam_width)
    window_width = cam_width
    cam_height = cap.get(4) # float
    cam_height = int(cam_height)
    window_height = cam_height
    while True:
        _, frame = cap.read()

        cv2.rectangle(frame, (140, 100), (500, 380), (0, 255, 0), 2)
        
        croppedframe = frame[100:-100, 140:-140]
        # background crop, safe to remove if not needed
        # cv2.imwrite('temp.jpg', croppedframe)
        mask = it.threshold(croppedframe, 'camera')
        # croppedframe = cv2.imread(temppath)

        # get prediction in array of string
        # shapetoplabels = getprediction(mask, 'camera', 'shape')
        # colortoplabels = getprediction(mask, 'camera', 'color')        

        shapetoplabels = getprediction(croppedframe, 'camera', 'shape')
        colortoplabels = getprediction(croppedframe, 'camera', 'color')        

        # font = cv2.FONT_HERSHEY_SIMPLEX
        fontpath = 'D:\\AIProject\\fonts\\bold-italic.otf'
        font = ImageFont.truetype(fontpath, 28)
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        draw.text((10, 0), '#1 ' + colortoplabels[0], font = font, fill = (b, g, r, a))
        draw.text((10, 28), '#2 ' + colortoplabels[1], font = font, fill = (b, g, r, a))
        draw.text((10, 56), '#3 ' + colortoplabels[2], font = font, fill = (b, g, r, a))
        draw.text((145, 95), 'press SPACE to capture an image', font = font, fill = (b, g, r, a))
        draw.text((10, window_height - 96), '#1 ' + shapetoplabels[0], font = font, fill = (b, g, r, a))
        draw.text((10, window_height - 68), '#2 ' + shapetoplabels[1], font = font, fill = (b, g, r, a))
        draw.text((10, window_height - 40), '#3 ' + shapetoplabels[2], font = font, fill = (b, g, r, a))
        img = np.array(img_pil)

        cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
        cv2.imshow('camera', img)
        cv2.imshow('mask', mask)
        cv2.imshow('cropped', croppedframe)

        key = cv2.waitKey(1)
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            break
        elif key == 32:
            _, currframe = cap.read()
            cv2.imwrite('capture.jpg', currframe)
            cap.release()
            cv2.destroyAllWindows()
            break
    return shapetoplabels, colortoplabels

def non_live_processing():
    window_height, window_width = 540, 720
    image_path = getfile('Select the input')

    # background crop
    # image_path = ip.remove_background(image_path)

    pic = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # background = cv2.imread('black.png')

    # merged_image = cv2.addWeighted(background, 0.4, pic, 0.1, 0)

    # cv2.imwrite('merged.png', merged_image)

    # image_path = 'merged.png'
    # mask = it.threshold(pic, 'image')

    # shapetoplabels = getprediction(mask, 'camera', 'shape')
    # colortoplabels = getprediction(mask, 'camera', 'color')


    shapetoplabels = getprediction(image_path, 'image', 'shape')
    colortoplabels = getprediction(image_path, 'image', 'color')

    height, width, channel = pic.shape
    # pic = cv2.resize(pic, (window_width, window_height))
    
    if height > width:
        window_height = 720
        window_width = 540

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(pic, '#1 ' + shapetoplabels[0], (10, window_height - 120), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(pic, '#2 ' + shapetoplabels[1], (10, window_height - 70), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(pic, '#3 ' + shapetoplabels[2], (10, window_height - 20), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(pic, '#1 ' + colortoplabels[0], (10, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(pic, '#2 ' + colortoplabels[1], (10, 100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(pic, '#3 ' + colortoplabels[2], (10, 150), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.namedWindow('res', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('res', window_width, window_height)
    cv2.imshow('res', pic)

    while True:
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            break      

# live_processing()
