import cv2, glob, os, sys
import pgmagick as pg
import directory_management as dm
import numpy as np
from matplotlib import pyplot as plt

def trans_mask_sobel(img):
    """ Generate a transparency mask for a given image """

    image = pg.Image(img)

    # Find object
    image.negate()
    image.edge()
    image.blur(1)
    image.threshold(24)
    image.adaptiveThreshold(5, 5, 5)

    # Fill background
    image.fillColor('magenta')
    w, h = image.size().width(), image.size().height()
    image.floodFillColor('0x0', 'magenta')
    image.floodFillColor('0x0+%s+0' % (w-1), 'magenta')
    image.floodFillColor('0x0+0+%s' % (h-1), 'magenta')
    image.floodFillColor('0x0+%s+%s' % (w-1, h-1), 'magenta')

    image.transparent('magenta')
    return image

def alpha_composite(image, mask):
    """ Composite two images together by overriding one opacity channel """

    compos = pg.Image(mask)
    compos.composite(
        image,
        image.size(),
        pg.CompositeOperator.CopyOpacityCompositeOp
    )
    return compos

def remove_background(filepath):
    """ Remove the background of the image in 'filename' """

    img = pg.Image(filepath)
    transmask = trans_mask_sobel(img)
    img = alpha_composite(transmask, img)
    img.trim()
    path = 'out.png'
    img.write(path)
    return path

def backgroundsub():
    cap = cv2.VideoCapture("highway.mp4")
    _, first_frame = cap.read()
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    first_gray = cv2.GaussianBlur(first_gray, (5, 5), 0)

    while True:
        _, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

def threshold(imagepath):
    # blur and grayscale before thresholding
    image = cv2.imread(imagepath)
    blur = cv2.cvtColor(src = image, code = cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(src = blur, ksize = (7, 7), sigmaX = 0)

    # perform inverse binary thresholding 
    (t, maskLayer) = cv2.threshold(src = blur, 
        thresh = 0, maxval = 255, 
        type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # make a mask suitable for color images
    mask = cv2.merge(mv = [maskLayer, maskLayer, maskLayer])

    # display the mask image
    # cv2.namedWindow(winname = "mask", flags = cv2.WINDOW_NORMAL)
    # cv2.imshow(winname = "mask", mat = mask)
    # cv2.waitKey(delay = 0)

    # use the mask to select the "interesting" part of the image
    sel = cv2.bitwise_and(src1 = image, src2 = mask)

    return sel
    # cv2.waitKey(delay = 0)

def colorhistogram():
    # read original image, in full color, based on command
    # line argument
    filename = dm.getfile("Select your image")
    image = cv2.imread(filename)

    # display the image 
    cv2.namedWindow(winname = "Original Image", flags = cv2.WINDOW_NORMAL)
    cv2.imshow(winname = "Original Image", mat = image)
    cv2.waitKey(delay = 0)

    # split into channels
    channels = cv2.split(image)

    # tuple to select colors of each channel line
    colors = ("b", "g", "r") 

    # create the histogram plot, with three lines, one for
    # each color
    plt.xlim([0, 256])
    for(channel, c) in zip(channels, colors):
        histogram = cv2.calcHist(
            images = [channel], 
            channels = [0], 
            mask = None, 
            histSize = [256], 
            ranges = [0, 256])

        plt.plot(histogram, color = c)

    plt.xlabel("Color value")
    plt.ylabel("Pixels")

    plt.show()

def grayscalehistogram():
    # read image, based on command line filename argument;
    # read the image as grayscale from the outset
    filename = getfile("Select your image")
    image = cv2.imread(filename, flags = cv2.IMREAD_GRAYSCALE)

    # display the image
    cv2.namedWindow(winname = "Grayscale Image", flags = cv2.WINDOW_NORMAL)
    cv2.imshow(winname = "Grayscale Image", mat = image)
    cv2.waitKey(delay = 0)

    # create the histogram
    histogram = cv2.calcHist(images = [image], 
        channels = [0], 
        mask = None, 
        histSize = [256], 
        ranges = [0, 256])

    # configure and draw the histogram figure
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixels")
    plt.xlim([0, 256]) # <- named arguments do not work here

    plt.plot(histogram) # <- or here
    plt.show()

shape_paths = ['Boots Ankle', 'Boots Knee High', 'Boots Mid-Calf',
'Sandals', 'Shoes Flats', 'Shoes Heels', 'Shoes Loafers', 
'Shoes Oxfords', 'Shoes Sneakers and Athletic']

color_paths = ['Beige', 'Black', 'Blue', 'Brown', 'Gold', 'Gray',
 'Green', 'Multicolor', 'Orange', 'Pink', 'Purple', 'Red', 'Silver',
 'White', 'Yellow']

shape_source_path = 'D:\\AIProject\\shoe-dataset-shape\\'
color_source_path = 'D:\\AIProject\\shoe-dataset-color\\'

shape_save_path = 'D:\\AIProject\\shoe-dataset-shape-mod\\'
color_save_path = 'D:\\AIProject\\shoe-dataset-color-mod\\'

def loopoverimages(images, traintype, categorytype):
    for image in images:
        ## Your core processing code 
        res = threshold(image)
        ## rename and write back to the disk
        name = image[32:len(image) - 4]
        ext = image[len(image) - 4:len(image)]
        # if traintype == 'train':
            
        # elif traintype == 'validation':
        #     name = image[43:len(image) - 4]
        #     ext = image[len(image) - 4:len(image)]  
        print(name, ext)
        # name, ext = os.path.splitext(image)
        # imgname = name+"_res"+ext
        if categorytype == 'shape':
            imgname = shape_save_path + name + '_res' + ext
        elif categorytype == 'color':
            imgname = color_save_path + name + '_res' + ext
        print(imgname)
        # display the result
        cv2.imshow('current', mat = res)
        cv2.imwrite(imgname, res)

def processimages(path, traintype, categorytype):
    ## Get all the png image in the PATH_TO_IMAGES
    if categorytype == 'shape':
        pngs = sorted(glob.glob(shape_source_path + traintype + '\\' + path + '\\*.png'))
        jpgs = sorted(glob.glob(shape_source_path + traintype + '\\' + path + '\\*.jpg'))
    elif categorytype == 'color':
        pngs = sorted(glob.glob(color_source_path + traintype + '\\' + path + '\\*.png'))
        jpgs = sorted(glob.glob(color_source_path + traintype + '\\' + path + '\\*.jpg'))
    loopoverimages(pngs, traintype, categorytype)
    loopoverimages(jpgs, traintype, categorytype)

def processmultipleimage():
    for path in shape_paths:
        processimages(path, 'train', 'shape')
        processimages(path, 'validation', 'shape')
    for path in color_paths:
        processimages(path, 'train', 'color')
        processimages(path, 'validation', 'color')

processmultipleimage()
# remove_background(dm.getfile('Select an image'))
# threshold(dm.getfile('Select an image'))