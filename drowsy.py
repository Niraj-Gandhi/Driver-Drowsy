#Imports
import tensorflow as tf
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

def lip_dist(shape):
    top_lip=shape[50:53]
    top_lip=np.concatenate((top_lip,shape[61:64]))
    lower_lip = shape[56:59]
    lower_lip = np.concatenate((lower_lip, shape[65:68]))

    mean_top=np.mean(top_lip,axis=0)
    mean_low=np.mean(lower_lip,axis=0)

    distance=abs(mean_top[1]-mean_low[1])
    return distance

def mouth_ear(mouth):
    m_ear=dist.euclidean(mouth[0],mouth[3])
    return m_ear
eyethresh=0.3
eyecons=48
eyecons1=150
yawn_thresh=25
count=0
alarm_status=False

#shape_predictor_68_face_landmarks.dat
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lstart,lend)= face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rstart,rend)=face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
#(mstart,mend)=face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

c=cv2.VideoCapture(0)
flag=0
c2=0
start=0.0
end=0.0
check = 0
eye_closure_duration = 0.0
once=1
drowsy_time=0


while True:
    ret,frame=c.read()
    frame=imutils.resize(frame,width=450)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    obj= detector(gray,0)
    if obj:
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
            '''
            mouth=shape[mstart:mend]
            h_mouth_ear=mouth_ear(mouth)
            print(f"horizontal lip distance=={h_mouth_ear}")
            '''
            shape_m=predictor(gray,s)
            x=shape_m.part(48).x
            y=shape_m.part(48).y
            #cv2.circle(frame,(x,y),3,(0,255,0),-1)
            x1= shape_m.part(54).x
            y1=shape_m.part(54).y
            #cv2.circle(frame, (x1, y1), 3, (0, 255, 0), -1)
            horizontal_mouth_dist=dist.euclidean((x,y),(x1,y1))
            lip_of_distance=lip_dist(shape)
            lip=shape[48:60]
            lip=cv2.convexHull(lip)
            cv2.drawContours(frame,[leftEyeHull],-1,(255,255,255),1)
            cv2.drawContours(frame, [rightEyeHull], -1, (255, 255, 255), 1)
            cv2.drawContours(frame,[lip],-1,(255,255,255),1)
            mod_ear=float(round(ear,2))
            cv2.putText(frame,f"EAR={mod_ear}",(0,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            print(f"ear=={ear}")

            if ear<eyethresh:
                check=1
                if once==1:
                    start = time.time()
                    print(f"start time=={start}")
                c2=0
                count+=1
                if count>=eyecons:
                    if not alarm_status:
                        alarm_status=True
                        t = Thread(target=alarm, daemon=True)
                        t.start()
                        print("*****************ALERT****************")
                    cv2.putText(frame,"*****************ALERT****************",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                once=0
            else:
                if check==1:
                    check=0
                    once=1
                    end=time.time()
                    print(f"end time=={end}")
                    eye_closure_duration = float(end-start)
                    print(f"Eye Closure duarion =={eye_closure_duration}")
                if eye_closure_duration>0.4 and eye_closure_duration<1.2:
                    print("are you feeling drowsy")
                    cv2.putText(frame, "are you feeling drowsy", (5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)
                count=0
                alarm_status=False

            if lip_of_distance>yawn_thresh and horizontal_mouth_dist<=41.2:
                cv2.putText(frame,"are you feeling drowsy and tired",(7,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                if not alarm_status:
                    alarm_status=True
                    t1=Thread(target=alarm,daemon=True)
                    t1.start()
            else:
                alarm_status=False




    else:
        count=0
        c2+=1
        if c2>=eyecons1:
            if not alarm_status:
                alarm_status = True
                t = Thread(target=alarm, daemon=True)
                t.start()
                print("Where are you looking")
            cv2.putText(frame, "Where are you looking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)
            c2=0
        alarm_status=False


    cv2.imshow("Frame",frame)
    key=cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
cv2.destroyAllWindows()
c.release()


