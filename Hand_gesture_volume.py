#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volbar=400
volper=0
 
volMin,volMax = volume.GetVolumeRange()[:2]




mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils
model = load_model('mp_hand_gesture')
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    x, y, c = frame.shape
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)    
    className = ''
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
            # print(np.sqrt(((aa-cc)**2)+((bb-dd)**2))*10)
            # cv2.line(frame,(aa,bb),(cc,dd),(255,0,0),3)
            # prediction = model.predict([landmarks])
            # classID = np.argmax(prediction)
            # className = classNames[classID]
            className="Detected"
        aa=landmarks[4][0]
        bb=landmarks[4][1]
        cc=landmarks[8][0]
        dd=landmarks[8][1]
        length = hypot(cc-aa,dd-bb) #distance b/w 
        vol = np.interp(length,[30,350],[volMin,volMax]) 
        volbar=np.interp(length,[30,350],[400,200])
        volper=np.interp(length,[30,350],[0,100])
        # print(vol,int(length))
        volume.SetMasterVolumeLevel(vol, None)
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()

cv2.destroyAllWindows()


# In[ ]:




