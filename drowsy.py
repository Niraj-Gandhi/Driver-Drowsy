#Imports
from os import path
import os
from statistics import mean
from xlwt import Workbook
import dlib
import cv2
from  scipy.spatial import distance as dist
from threading import Thread
import numpy as np
import playsound
from imutils import face_utils
import imutils
import time

#start: audio function
audiopath="alarm.wav"
def alarm():
    playsound.playsound(audiopath)
#end: audio function

#start:eye aspect ratio function
def eye_aspect_ratio(eye):
    X1=dist.euclidean(eye[1],eye[5])                                #calculating vertical euclidean distance between two eye points
    X2=dist.euclidean(eye[2],eye[4])
    mean_X=mean((X1,X2))
    Y=dist.euclidean(eye[0],eye[3])                                 #calculating horizontal euclidean distance between two eye points
    ear=(X1+X2)/(2.0*Y)
    return ear,mean_X

#start: yawn detection function
def lip_dist(shape):
    top_lip=shape[50:53]
    top_lip=np.concatenate((top_lip,shape[61:64]))
    lower_lip = shape[56:59]
    lower_lip = np.concatenate((lower_lip, shape[65:68]))

    mean_top=np.mean(top_lip,axis=0)
    mean_low=np.mean(lower_lip,axis=0)

    distance=abs(mean_top[1]-mean_low[1])
    return distance
#end :yawn detection function

#start: EAR function for mouth (for yawn detection)
def mouth_ear(mouth):
    m_ear=dist.euclidean(mouth[0],mouth[3])
    return m_ear
#end: EAR function for mouth (for yawn detection)

#start:function for gaze direction
def get_gaze_ratio(eye_points,landmark):
    #specifying the eye to be considered
    #the points specified indicate the eye in consideration
    eye_region = np.array([(landmark.part(eye_points[0]).x, landmark.part(eye_points[0]).y),
                                (landmark.part(eye_points[1]).x, landmark.part(eye_points[1]).y),
                                (landmark.part(eye_points[2]).x, landmark.part(eye_points[0]).y),
                                (landmark.part(eye_points[3]).x, landmark.part(eye_points[3]).y),
                                (landmark.part(eye_points[4]).x, landmark.part(eye_points[4]).y),
                                (landmark.part(eye_points[5]).x, landmark.part(eye_points[5]).y)], np.int32)
    height, width, _ = frame.shape                          #dimensions of the frame
    mask = np.zeros((height, width), np.uint8)              #to cover everything except the eye
    cv2.polylines(mask, [eye_region], True, 255, 2)         #to create a polygon of the size of the eye
    cv2.fillPoly(mask, [eye_region], 255)                   #filling the polygon at desired position
    eye = cv2.bitwise_and(gray, gray, mask=mask)
    min_x = np.min(eye_region[:, 0])                        #dimensions of the eye
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]              #eye region (black and white)
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    thresh_height, thresh_width = threshold_eye.shape
    left_threshold = threshold_eye[0:thresh_height, 0:int(thresh_width/2)]          #taking the left half of and eye
    left_white = cv2.countNonZero(left_threshold)                                   #taking the white region in the left half
    right_threshold = threshold_eye[0:thresh_height, int(thresh_width/2):width]     #taking the right half of an eye
    right_white = cv2.countNonZero(right_threshold)                                 #taking the white region in the right half

    if left_white == 0:
        gaze_ratio = 1
    elif right_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_white / right_white

    return gaze_ratio

#end:function for gaze direction

#initializations
#eye Threshold count
eyethresh=0.3
#threshold for alert in case eyes are closed
eyecons=48
#threshold for alert if driver is looking in a different direction
eyecons1=150
#to detect yawning
yawn_thresh=25
#to keep track for how much time eyes are closed
count=0
#alarm activation 
alarm_status=False
#camera video capture
c=cv2.VideoCapture(0)
#to keep track for how much time does the driver look in a different direction
c2=0
#alarm time calculation
start=0.0
end=0.0

check = 0
eye_closure_duration = 0.0
once=1
drowsy_time=0

max_eye=[]
#array to store perclos values
arr_perclos=[]
#array to store eye closure level values
arr_ecl=[]
ct=0
#log file insertion
log_x_perclos=1
log_x_ec=1
log_x_ecl=1
log_x_yawn=1
#perclos calculation
e_close = []
e_open = []
#yawn detection
yawn_cnt=0
yawn_check=0

#shape_predictor_68_face_landmarks.dat
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lstart,lend)= face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]                  #facial landmarks for left eye
(rstart,rend)=face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]                  #facial landmarks for right eye

#to remove logFile.xls file if exists previously
if path.exists("logFile.xls"):
    os.remove("logFile.xls")
#create a workbook to insert data into excel file
wb=Workbook()
#adding a sheet to workbook
drowsy_driver_log=wb.add_sheet('Sheet 1',cell_overwrite_ok=True)
drowsy_driver_log.write(0,0,'PERCLOS')
drowsy_driver_log.write(0,1,'Eye Closure Duration')
drowsy_driver_log.write(0,2,'Eye Closure Level')
drowsy_driver_log.write(0,3,'Yawning Count')
while True:
    ret,frame=c.read()                                                              #capture the current frame
    frame=imutils.resize(frame,width=450)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)                                     #convert to gray
    obj= detector(gray,0)                                                           #detect face in the frame

    if obj:
        for s in obj:

            shape=predictor(gray,s)                                                 #get shape from detected face
            shape=face_utils.shape_to_np(shape)
            leftEye=shape[lstart:lend]                                              #get left eye from frame
            rightEye=shape[rstart:rend]                                             #get right eye from frame
            (lear,mean_l)=eye_aspect_ratio(leftEye)                                 #calculate eye aspect ratio
            (rear,mean_r)=eye_aspect_ratio(rightEye)
            mean_eye=mean((mean_l,mean_r))
            max_eye.append(mean_eye)
            ear=(lear+rear)/2.0
            leftEyeHull=cv2.convexHull(leftEye)                                     #mark both the eyes
            rightEyeHull=cv2.convexHull(rightEye)
            shape_m=predictor(gray,s)
            x=shape_m.part(48).x
            y=shape_m.part(48).y
            x1= shape_m.part(54).x
            y1=shape_m.part(54).y
            horizontal_mouth_dist=dist.euclidean((x,y),(x1,y1))                     #get horizontal mouth width
            lip_of_distance=lip_dist(shape)                                         #calculate AR for mouth
            lip=shape[48:60]
            face_mark=shape[0:17]
            lip=cv2.convexHull(lip)                                                 #mark the mouth
            for f_cont in range(1,len(face_mark)):
                pta=tuple(face_mark[f_cont-1])
                ptb=tuple(face_mark[f_cont])
                cv2.line(frame,pta,ptb,(255,255,255),1)

            cv2.drawContours(frame,[leftEyeHull],-1,(255,255,255),1)                #outline the marked eyes
            cv2.drawContours(frame, [rightEyeHull], -1, (255, 255, 255), 1)
            cv2.drawContours(frame,[lip],-1,(255,255,255),1)                        #outline the marked mouth
            mod_ear=float(round(ear,2))
            cv2.putText(frame,f"EAR={mod_ear}",(0,300),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            print(f"ear=={ear}")
            if ear<=0.21:                                                           #if eyes are closed
                e_close.append(1)                                                   #add 1 and 0 to respective arrays
                e_open.append(0)
            if ear>0.21:                                                            #if eyes are open
                e_close.append(0)
                e_open.append(1)

            perclos=round((sum(e_close)/(sum(e_close)+sum(e_open)))*100,2)          #calculate PERCLOS values
            arr_perclos.append(perclos)                                             #stroe PERCLOS values
            print(e_open)
            print(e_close)
            arr_ecl.append(round((1 - (mean_eye / max(max_eye))) * 100, 2))         #calculate and append eye closure level values

            if ear<eyethresh:                                                       #if EAR is less than threshold value
                check=1
                if once==1:
                    start = time.time()                                             #start timer
                    print(f"start time=={start}")
                c2=0
                count+=1
                if count>=eyecons:                                                  #if time for which eyes are closed exceeds threshold values
                    if not alarm_status:
                        alarm_status=True                                           #set alarm activation true
                        t = Thread(target=alarm, daemon=True)                       #new Thread for alarm
                        t.start()                                                   #alarm
                        print("*****************ALERT****************")
                    cv2.putText(frame,"ALERT",(100,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                once=0
            else:
                if check==1:
                    check=0
                    once=1
                    end=time.time()
                    print(f"end time=={end}")
                    eye_closure_duration = float(end-start)                         #calculate time duration for which eyes were closed
                    print(f"Eye Closure duration =={eye_closure_duration}")
                    if eye_closure_duration>3.0:
                        drowsy_driver_log.write(log_x_ec,1,eye_closure_duration)
                        log_x_ec+=1
                if eye_closure_duration>0.4 and eye_closure_duration<2.2:
                    print("are you feeling drowsy")
                    cv2.putText(frame, "are you feeling drowsy", (5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)
                count=0
                alarm_status=False

            if lip_of_distance>yawn_thresh and horizontal_mouth_dist<=41.2:         #if the driver is yawning
                cv2.putText(frame,"are you feeling drowsy and tired",(7,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                yawn_check=1
            else:
                alarm_status=False

            gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], shape_m)         #call to get gaze ratio function
            gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], shape_m)
            gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2
            if gaze_ratio <= 1:                                                              #find the direction
                cv2.putText(frame, "RIGHT", (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif 1 < gaze_ratio < 1.7:
                cv2.putText(frame, "CENTER", (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "LEFT", (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print(ct)
            if ct%120==0:
                drowsy_driver_log.write(log_x_perclos,0,f"{max(arr_perclos)}%")
                log_x_perclos+=1
                eye_closure_level=max(arr_ecl)                             #round((1-(mean_eye/max(max_eye)))*100,2)
                drowsy_driver_log.write(log_x_ecl,2,f"{eye_closure_level}%")
                log_x_ecl+=1
                arr_ecl=[]
                if yawn_check==1:
                    yawn_cnt+=1
                    drowsy_driver_log.write(log_x_yawn,3,yawn_cnt)
            ct+=1

            wb.save("logFile.xls")


    else:
        count=0
        c2+=1
        if c2>=eyecons1:                                                    #if driver is looking in different direction for more than the threshold values
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


