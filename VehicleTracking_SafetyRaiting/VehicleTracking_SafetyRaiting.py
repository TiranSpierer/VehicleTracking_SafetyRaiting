import cv2
import numpy as np

cap = cv2.VideoCapture('Video/1.mp4')
object_detector = cv2.createBackgroundSubtractorMOG2(history = 100, varThreshold= 100)
tracker = EuclideanDistTracker()

while True:
    ret, frame = cap.read()
    if ret == False: break
        
    black = np.zeros_like(frame)
    mask = object_detector.apply(frame)
    contours,_ =cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detection = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20:
            cv2.drawContours(black,[cnt],-1,(255,255,255),1)
            x,y,w,h = cv2.boundingRect(cnt)
            detection.append([x,y,w,h])
    
    boxes_ids = tracker.update(detection)
    for box_id in boxes_ids:
        x,y,w,h,id = box_id
        cv2.putText(black,str(id), (x,y-15),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
        
    cv2.imshow('frame', frame)
    cv2.imshow('black', black)
    
    
    if cv2.waitKey(30)==ord('q'):
        break

    #cv2.imshow('frame', cv2.resize(frame,f(frame,0.7)))  
    #cv2.imshow('mask', cv2.resize(mask,f(mask,0.7)))
cv2.destroyAllWindows()    