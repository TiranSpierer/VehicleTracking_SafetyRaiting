import cv2
import numpy as np

cap = cv2.VideoCapture('Video/3.mp4')
ret,oldFrame = cap.read()
oldGrayFrame = cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY) 
blank2 = np.zeros(oldFrame.shape, np.uint8)

while ret:
    ret, newFrame = cap.read()
    if ret == False: break
    newFrame1 = newFrame.copy()
    
    newGrayFrame = cv2.cvtColor(newFrame, cv2.COLOR_BGR2GRAY) 
    result = cv2.absdiff(newGrayFrame,oldGrayFrame)
    blank1 = np.zeros(oldFrame.shape, np.uint8)
    if np.max(result)>10 :
        threshold = 30
        result[result>=threshold]=255
        result[result<threshold]=0
        result = cv2.GaussianBlur(result,(11,11),7)
    
        contours,_ = cv2.findContours(result,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            if cv2.contourArea(cnt)>250:
                cv2.drawContours(blank1,cnt,-1,(0,255,0),2)  
                cv2.drawContours(blank2,cnt,-1,(0,255,0),2)  
                cv2.drawContours(newFrame1,cnt,-1,(0,255,0),2)  
        cv2.imshow('result',cv2.resize(result,(640,400)))  
        cv2.imshow('newFrame',cv2.resize(newFrame,(640,400)))   
        cv2.imshow('blank1',cv2.resize(blank1,(640,400)))   
        cv2.imshow('blank2',cv2.resize(newFrame1,(640,400)))   
    
    oldGrayFrame = newGrayFrame
    
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()    








def buildRoi(cap):
    ret,oldFrame = cap.read()
    oldGrayFrame = cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY) 
    blank = np.zeros(oldFrame.shape, np.uint8)

    while ret:
        ret, newFrame = cap.read()
        if ret == False: break
    
        newGrayFrame = cv2.cvtColor(newFrame, cv2.COLOR_BGR2GRAY) 
        result = cv2.absdiff(newGrayFrame,oldGrayFrame)
        if np.max(result)>10 :
            threshold = 50
            result[result>=threshold]=255
            result[result<threshold]=0
            result = cv2.GaussianBlur(result,(11,11),5)
    
            contours,_ = cv2.findContours(result,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                if cv2.contourArea(cnt)>300:
                    cv2.drawContours(blank,cnt,-1,(0,255,0),2)  

        oldGrayFrame = newGrayFrame
    
    
    kernel = np.ones((7,7),np.uint8)
    opening = cv2.morphologyEx(blank, cv2.MORPH_OPEN, kernel)
    
    
    grayopening = cv2.cvtColor(opening,cv2.COLOR_BGR2GRAY)
    contours,_ = cv2.findContours(grayopening,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    areas = [cv2.contourArea(cnt) for cnt in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    return cv2.boundingRect(cnt)







    def timeGaussian(first,second,third):
    return first//4 + second//2 + third//4

def time(cap):
    _,one = cap.read()
    two = one.copy()
    _,three = cap.read()
    out = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 23, size)

    result = timeGaussian(one,one,three)
    cv2.imshow('4',cv2.resize(result,(960,600)))
    out.write(result)

    while True:
        one = two
        two = three
        ret,three = cap.read()
        if ret==False: break
        result = timeGaussian(one,two,three)
        cv2.imshow('4',cv2.resize(result,(960,600)))
        out.write(result)
 
        if cv2.waitKey(1)==ord('q'): break

    result = timeGaussian(one,two,two)
    cv2.imshow('4',cv2.resize(result,(960,600)))
    out.write(result)

    out.release()       
    cv2.destroyAllWindows()