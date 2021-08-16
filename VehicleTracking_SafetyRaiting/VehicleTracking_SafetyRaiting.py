import cv2
import numpy as np


path = 'Video/1.mp4'

buildRoi(path)
timeDerivative()
subtraction()


def buildRoi(path):
    cap = cv2.VideoCapture(path)
    
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
    x,y,w,h = cv2.boundingRect(cnt)
    
    
    #create the crop video
    hight,width=newGrayFrame.shape
    size = (width,hight)
    crop = cv2.VideoCapture(path)
    
    out = cv2.VideoWriter('cropVideo.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 23, (w,h))
    
    while True:
        ret,frame=crop.read()
        if ret==False: break
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        cv2.imshow('g',frame)
        roi = frame[y:y+h,x:x+w]
        cv2.imshow('f',roi)
        if cv2.waitKey(10)==27:break
            
        out.write(roi)
        
    cv2.destroyAllWindows()
    out.release()       






def timeGaussian(first,second,third):
    return first//4 + second//2 + third//4

def timeDerivative():
    path = 'cropVideo.avi'
    cap = cv2.VideoCapture(path)
    _,one = cap.read()
    two = one.copy()
    _,three = cap.read()
    h,w,_=one.shape
    size = (w,h)
    out = cv2.VideoWriter('cropGaussianVideo.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 23, size)

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





def subtraction():
    path = 'cropGaussianVideo.mp4'
    cap = cv2.VideoCapture(path)
    _,oldFrame = cap.read()
    h,w,_=oldFrame.shape

    oldGrayFrame = cv2.cvtColor(oldFrame,cv2.COLOR_BGR2GRAY)
    out1 = cv2.VideoWriter('cropGaussianMaskVideo.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 23, (w,h))
    out2 = cv2.VideoWriter('cropGaussianContoursVideo.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 23, (w,h))
    
    while True:
        ret, newFrame = cap.read()
        if ret == False: break
    
        newGrayFrame = cv2.cvtColor(newFrame, cv2.COLOR_BGR2GRAY) 
        result = cv2.absdiff(newGrayFrame,oldGrayFrame)
        
        threshold = 30
        result[result>=threshold]=255
        result[result<threshold]=0
        result = cv2.GaussianBlur(result,(11,11),7)
    
        contours,_ = cv2.findContours(result,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            if cv2.contourArea(cnt)>250:
                cv2.drawContours(newFrame,cnt,-1,(0,255,0),2)  
        out1.write(cv2.cvtColor(result,cv2.COLOR_GRAY2BGR))
        out2.write(newFrame)  
    
        oldGrayFrame = newGrayFrame
    
        if cv2.waitKey(1)==ord('q'): break
    out1.release()
    out2.release()
    cv2.destroyAllWindows()    