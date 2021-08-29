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

def buildRoi(path):
    cap = cv2.VideoCapture(path)
    ret,oldFrame = cap.read()
    oldGrayFrame = cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY) 
    blank = np.zeros(oldFrame.shape, np.uint8)
    frames = []

    for _ in range(int(cap.get(7)*0.1)): 
        ret, frame = cap.read()
        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(grayframe)  
        result = cv2.absdiff(grayframe,oldGrayFrame)

        if np.mean(result)!=0: 
            result = cv2.threshold(result, 30, 255, cv2.THRESH_BINARY)[1]
            result = cv2.GaussianBlur(result,(11,11),7)
            contours,_ = cv2.findContours(result,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            for cnt in contours:
                if cv2.contourArea(cnt)>150:
                    cv2.drawContours(blank,cnt,-1,(255,255,255),20)

        oldGrayFrame = grayframe
        
    blank = cv2.cvtColor(blank,cv2.COLOR_BGR2GRAY)
    median = np.median(frames, axis=0).astype(dtype=np.uint8) 
    result = median*blank
    result = cv2.threshold(result, 120, 255, cv2.THRESH_BINARY)[1]
    result = cv2.GaussianBlur(result,(5,5),10)
    cv2.imwrite(grayScaleRoad, result)   
    cap.release()


def frameDiffer(path):
    cap = cv2.VideoCapture(path)
    ret,one = cap.read()
    ret,two = cap.read()
    oneGray = cv2.cvtColor(one, cv2.COLOR_BGR2GRAY) 
    twoGray = cv2.cvtColor(two, cv2.COLOR_BGR2GRAY) 
    h, w = twoGray.shape
    size = (w, h)
    out = cv2.VideoWriter(resultPath(path), cv2.VideoWriter_fourcc(*'DIVX'), 23, size)
    
    while True:
        Roi = cv2.imread(grayScaleRoad)

        ret, three = cap.read()
        if ret == False: 
            break
            
        threeGray = cv2.cvtColor(three, cv2.COLOR_BGR2GRAY) 
        result = cv2.absdiff(threeGray,oneGray)
        result = cv2.GaussianBlur(result,(5,5),5)
        result = cv2.threshold(result, 10, 255, cv2.THRESH_BINARY)[1]
        result = cv2.GaussianBlur(result,(7,7),5)
        result = cv2.threshold(result, 10, 255, cv2.THRESH_BINARY)[1]
        kernel = np.ones((10,10),np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        contours,_ = cv2.findContours(result,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            if cv2.contourArea(cnt)>200:
                x,y,w,h = cv2.boundingRect(cnt)
                cx = (x+x+w)//2
                cy = (y+y+h)//2
                if sum(Roi[cy,cx])==0: 
                    cv2.drawContours(three,cnt,-1,(0,0,255),2) 
                else:
                    cv2.drawContours(three,cnt,-1,(0,255,0),2) 
        
        out.write(three)
        oneGray = twoGray
        twoGray = threeGray
        

def runProgram(path):
    buildRoi(path)
    frameDiffer(path)
    newPath = resultPath(path)
    showVideo(newPath)