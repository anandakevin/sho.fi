import cv2
import numpy as np
import imutils

def nothing(x):
    pass

# untuk video
cap = cv2.VideoCapture(0)

currwindow = cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

def getHSV(window):
    l_h = cv2.getTrackbarPos("LH", window)
    l_s = cv2.getTrackbarPos("LS", window)
    l_v = cv2.getTrackbarPos("LV", window)

    u_h = cv2.getTrackbarPos("UH", window)
    u_s = cv2.getTrackbarPos("US", window)
    u_v = cv2.getTrackbarPos("UV", window)

    return l_h, l_s, l_v, u_h, u_s, u_v

def drawline(mask, frame):
    # receive contour areas
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    countct = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        print(area)
        # draw the contours
        if area > 5000:
            cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
            countct += 1
    print("Total Objects: " + str(countct))


while True:
    # picture
    # frame = cv2.imread('hsv.jpg')
    # video
    _, frame = cap.read()
    filtered_frame = cv2.GaussianBlur(src = frame, ksize=(5, 5), sigmaX=0)
    hsv = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2HSV)
    thresh = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY)

    # get lower and upper bound for hsv value
    l_h, l_s, l_v, u_h, u_s, u_v = getHSV("Tracking")

    # l_b = np.array([110, 50, 50])
    # u_b = np.array([130, 255, 255])
    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])

    # perform inverse binary thresholding 
    (t, maskLayer) = cv2.threshold(src = thresh, thresh = 0, maxval = 255, type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # make a mask suitable for color images
    maskt = cv2.merge(mv = [maskLayer, maskLayer, maskLayer])
    maskhsv = cv2.inRange(hsv, l_b, u_b)

    reshsv = cv2.bitwise_and(frame, frame, mask=maskhsv)
    rest = cv2.bitwise_and(frame, maskt)

    drawline(maskLayer, rest)
    drawline(maskhsv, frame)

    # Display the results
    cv2.imshow("frame", frame)
    cv2.imshow("mask-threshold", maskt)
    cv2.imshow("mask-hsv", maskhsv)

    cv2.imshow("res-hsv", reshsv)
    cv2.imshow("res-threshold", rest)

    key = cv2.waitKey(1)
    if key == 27:
        cap.release()
        cv2.destroyAllWindows()
        break

