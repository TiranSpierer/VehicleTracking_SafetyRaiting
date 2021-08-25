import cv2
import numpy as np


def buildRoi(path):
    cap = cv2.VideoCapture(path)
    
    frameAmount = cap.get(7)   
    trainTestRatio = 0.1
    frameCounter = 1
    
    ret,oldFrame = cap.read()
    oldGrayFrame = cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY) 
    blank = np.zeros(oldFrame.shape, np.uint8)

    while ret and frameCounter < (frameAmount*trainTestRatio):
        frameCounter+=1
        
        ret, newFrame = cap.read()
        if ret == False: break

        newGrayFrame = cv2.cvtColor(newFrame, cv2.COLOR_BGR2GRAY) 
        result = cv2.absdiff(newGrayFrame,oldGrayFrame)

        if np.mean(result)!=0: 
            result = cv2.threshold(result, 30, 255, cv2.THRESH_BINARY)[1]
            result = cv2.GaussianBlur(result,(11,11),7)

            contours,_ = cv2.findContours(result,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                if cv2.contourArea(cnt)>150:
                    cv2.drawContours(blank,cnt,-1,(0,255,0),6)  
            cv2.imshow('blank',cv2.resize(blank,(640,400)))   

        oldGrayFrame = newGrayFrame

        if cv2.waitKey(1)==ord('q'): break
    input()
    cv2.destroyAllWindows()  
        
    #prefoem opening and closing operation for \\\"cleaning\\\" the image
    kernel = np.ones((9, 9),np.uint8)
    opening = cv2.morphologyEx(blank, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('f',cv2.resize(opening,(640,400)))
    cv2.imshow('g',cv2.resize(closing,(640,400)))
    cv2.waitKey(1)
    input()
    cv2.destroyAllWindows()
    
    #create photo of the Roi for later
    closing = cv2.cvtColor(closing,cv2.COLOR_BGR2GRAY)
    closing[closing!=0]=255
    cv2.imwrite("blackAndWhiteRoad.jpg", closing)   
    cv2.destroyAllWindows()
        
        
    cap.release()
    cv2.destroyAllWindows() 



    def frameDiffer(path):
    cap = cv2.VideoCapture(path)
    ret,oldFrame = cap.read()
    oldGrayFrame = cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY) 
    
    while ret:
        Roi = cv2.imread('blackAndWhiteRoad.jpg')
        ret, newFrame = cap.read()
        if ret == False: break
        newGrayFrame = cv2.cvtColor(newFrame, cv2.COLOR_BGR2GRAY) 
        result = cv2.absdiff(newGrayFrame,oldGrayFrame)
        if np.max(result)>10 :
            threshold = 30
            result[result>=threshold]=255
            result[result<threshold]=0
            result = cv2.GaussianBlur(result,(7,7),4)
    
            contours,_ = cv2.findContours(result,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                if cv2.contourArea(cnt)>100:
                    cv2.drawContours(Roi,cnt,-1,(0,255,0),2)  

            cv2.imshow('result',cv2.resize(Roi,(960,600)))  
            cv2.imshow('newFrame',cv2.resize(newFrame,(960,600)))   
    
        oldGrayFrame = newGrayFrame
    
        if cv2.waitKey(1)==ord('q'): break
    cap.release()
    cv2.destroyAllWindows()  




path = 'video/4.mp4'
#buildRoi(path)
frameDiffer(path)
cv2.destroyAllWindows() 