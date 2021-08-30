import cv2
import numpy as np

def resultPath(filePath):
    lastDot = filePath.rfind('.')
    return filePath[:lastDot] + 'Result' + filePath[lastDot:]

        

def showVideo(path):
    cap = cv2.VideoCapture(path)

    while True:     
        ret, frame = cap.read()
        
        if not ret: 
            break

        cv2.imshow("video", frame)
        if cv2.waitKey(50) == 27: 
            break

    cv2.destroyAllWindows()
    cap.release()


grayScaleRoad = "grayScaleRoad.jpg"

def preProcessing(path):
    cap = cv2.VideoCapture(path)
    ret,oldFrame = cap.read()
    oldGrayFrame = cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY) 
    blank = np.zeros(oldFrame.shape, np.uint8)
    
    #create Roi base on the 10% first frame
    for _ in range(int(cap.get(7)*0.1)): 
        ret, frame = cap.read()
        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #image processing
        result = cv2.threshold(result, 30, 255, cv2.THRESH_BINARY)[1]
        result = cv2.GaussianBlur(result,(11,11),7)
        
        #spread the contours on blank
        contours,_ = cv2.findContours(result,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            if cv2.contourArea(cnt)>150:
                cv2.drawContours(blank,cnt,-1,(255,255,255),20)
        #for continuesly
        oldGrayFrame = grayframe
    
    #more image processing on the result
    blank = cv2.cvtColor(blank,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((9, 9),np.uint8)
    opening = cv2.morphologyEx(blank, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    
    #save the outcome
    cv2.imwrite(grayScaleRoad, result)   
    cap.release()


def robustTimeDerivativeDistribution(path):
    Roi = cv2.imread(grayScaleRoad)
    cap = cv2.VideoCapture(path)
    
    #initialize the first loop parameters
    ret,one = cap.read()
    ret,two = cap.read()
    oneGray = cv2.cvtColor(one, cv2.COLOR_BGR2GRAY) 
    twoGray = cv2.cvtColor(two, cv2.COLOR_BGR2GRAY) 
    h, w = twoGray.shape
    size = (w, h)
    out = cv2.VideoWriter(resultPath(path), cv2.VideoWriter_fourcc(*'DIVX'), 23, size)
    
    while True:
        ret, three = cap.read()
        if ret == False: break
            
        threeGray = cv2.cvtColor(three, cv2.COLOR_BGR2GRAY) 
        result = cv2.absdiff(threeGray,oneGray)
        
        #image processing    
        result = cv2.GaussianBlur(result,(5,5),5)
        result = cv2.threshold(result, 10, 255, cv2.THRESH_BINARY)[1]
        result = cv2.GaussianBlur(result,(7,7),5)
        result = cv2.threshold(result, 10, 255, cv2.THRESH_BINARY)[1]
        
        kernel = np.ones((10,10),np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        
        #find moving objects
        contours,_ = cv2.findContours(result,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            if cv2.contourArea(cnt)>2000:
                x,y,w,h = cv2.boundingRect(cnt)
                cx = (x+x+w)//2
                cy = (y+y+h)//2
                if sum(Roi[cy,cx])!=0: 
                    cv2.rectangle(three,(x,y),(x+w,y+h),(0,255,0),3)
                else:
                    if cx<three.shape[1]//2:                
                        cv2.rectangle(three,(x,y),(x+w,y+h),(0,165,255),3)
                    else:
                        cv2.rectangle(three,(x,y),(x+w,y+h),(0,0,255),3)
                
        out.write(three)
        
        #show the output            
        cv2.imshow('f',three)
        cv2.waitKey(30)
        if cv2.waitKey(20) == 27: 
            break

        #for continuesly
        oneGray = twoGray
        twoGray = threeGray
        

def runProgram(path):
    preProcessing(path)
    robustTimeDerivativeDistribution(path)
    newPath = resultPath(path)
    #showVideo(newPath)

    roadDict(q, w, e, r, t):
        #..........

