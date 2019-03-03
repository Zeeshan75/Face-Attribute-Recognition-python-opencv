import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os

from keras.models import load_model
from keras.preprocessing.image import img_to_array
from scipy.spatial import distance as dist
from imutils import face_utils
from imutils.face_utils import *

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
smodel = load_model("models/smile_model.h5")
gmodel = load_model("glass/glass_model.h5")
agmodel = load_model('age_gen_mw.h5')

def eye(image, gray, rect, Threshold):
    shape = predictor(gray, rect)
    shape = shape_to_np(shape)

    #simple hack ;)
    if (len(shape)==68):
        # extract the left and right eye (x, y)-coordinates
        (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
        (mStart, mEnd) = FACIAL_LANDMARKS_68_IDXS["mouth"]
    else:
        (lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]

    leftEyePts = shape[lStart:lEnd]
    rightEyePts = shape[rStart:rEnd]
    mouth= shape[mStart:mEnd]
    
    # Calculating the eyes aspect_ratio 
    A = dist.euclidean(leftEyePts[1], leftEyePts[5])
    B = dist.euclidean(leftEyePts[2], leftEyePts[4])
    C = dist.euclidean(leftEyePts[0], leftEyePts[3])
    leftEye_ar = (A + B) / (2.0 * C)

    D = dist.euclidean(rightEyePts[1], rightEyePts[5])
    E = dist.euclidean(rightEyePts[2], rightEyePts[4])
    F = dist.euclidean(rightEyePts[0], rightEyePts[3])
    rightEye_ar = (D + E) / (2.0 * F)
    
    if leftEye_ar > Threshold:
        if rightEye_ar > Threshold:
            statement1 = " Both eyes are open"
        else:
            statement1 = " Left Eye is open and right eye is closed"
    else:
        if rightEye_ar > Threshold:
            statement1 = " right Eye is open and left eye is closed"
        else:
            statement1 = " Both eyes are closed"

    return statement1

def smile(face):
    predicted_emotions = smodel.predict(face)[0]
    best_emotion = ' YES' if predicted_emotions[1] > predicted_emotions[0] else ' NO'
    return best_emotion

def glass(face):
    gresult = gmodel.predict(face)[0]
    res = ' NO' if gresult[1] > gresult[0] else ' YES'
    return res

def aggface(face):
    sface = cv2.resize(face, (64, 64)).reshape((1, 64, 64, 3))
    if len(sface) > 0:
        results = agmodel.predict(sface)
        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()
        age = int(predicted_ages[0])
        gender = "F" if predicted_genders[0][0] > 0.5 else "M"
        return age,gender

def predict_attributes(img):
	image = cv2.imread(img)
	image = imutils.resize(image, width=700)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 1)   # detect faces in the grayscale image

	for i,rect in enumerate(rects):
	    (x, y, w, h) = rect_to_bb(rect)
	    x = max(0,rect.left())
	    y = max(0,rect.top())
	    face1 = gray[y:y+h, x:x+w]
	    face2 = image[max(0,y-int((0.2*h))):y+h+int((0.2*h)), x-int(0.2*w):x+w+int(0.2*w)]
	    
	    ey = eye(image, gray, rect, 0.2)
	    
	    sface = cv2.resize(face1, (48, 48)).reshape((1, 48, 48, 1))
	    sm = smile(sface)
	    
	    gface = cv2.resize(face1, (48, 48))
	    cv2.imwrite('images/test.jpg',gface)
	    gface = cv2.imread('images/test.jpg')
	    gface = gface.reshape((1,48, 48, 3))
	    gl = glass(gface)
	    
	    ag,ge = aggface(face2)
	    print('face #',str(i),',\n Eyes:',ey,', Smiling:',sm,', Glasses:',gl,', Gender:',ge,', Age:',str(ag))