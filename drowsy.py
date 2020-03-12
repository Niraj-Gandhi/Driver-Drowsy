#Imports

import dlib
from pathlib import Path
import cv2
from  scipy.spatial import distance as dist
from imutils.video import VideoStream
from threading import Thread
import numpy as np
import playsound
from imutils import face_utils
import imutils
import time
import argparse

#audio file path
#D:\drowst detect\alarm.wav
audiopath="alarm.wav"
def alarm():
    playsound.playsound(audiopath)

#eye aspect ratio function
def eye_aspect_ratio(eye):
    X1=dist.euclidean(eye[1],eye[5])
    X2=dist.euclidean(eye[2],eye[4])
    Y=dist.euclidean(eye[0],eye[3])
    ear=(X1+X2)/(2.0*Y)
    return ear

eyethresh=0.3
eyecons=48

count=0
alarm_status=False

#shape_predictor_68_face_landmarks.dat
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lstart,lend)= face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rstart,rend)=face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
c=cv2.VideoCapture(0)
flag=0

while True:
    ret,frame=c.read()
    frame=imutils.resize(frame,width=450)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    obj= detector(gray,0)
    for s in obj:
        shape=predictor(gray,s)
        shape=face_utils.shape_to_np(shape)
        leftEye=shape[lstart:lend]
        rightEye=shape[rstart:rend]
        lear=eye_aspect_ratio(leftEye)
        rear=eye_aspect_ratio(rightEye)
        ear=(lear+rear)/2.0
        leftEyeHull=cv2.convexHull(leftEye)
        rightEyeHull=cv2.convexHull(rightEye)
        cv2.drawContours(frame,[leftEyeHull],-1,(255,255,255),1)
        cv2.drawContours(frame, [rightEyeHull], -1, (255, 255, 255), 1)
        if ear<eyethresh:
            count+=1
            print(count)

            if count>=eyecons:
                if not alarm_status:
                    alarm_status=True
                    t = Thread(target=alarm, daemon=True)
                    t.start()
                cv2.putText(frame,"*****************ALERT****************",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        else:
            count=0
            alarm_status=False

    cv2.imshow("Frame",frame)
    key=cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
cv2.destroyAllWindows()
c.stop()


