import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np

model = YOLO('best.pt')

def RGB(event, x, y,flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point=[x,y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB',RGB)

cap=cv2.VideoCapture('cv.avi')

my_file = open("coco.txt","r")
data=my_file.read()
class_list=data.split("\n")


count = 0
area1=[(69,136),(399,88),(544,163),(59,277)]
#area2=[(114,190),(429,148),(546,237),(86,322)]

while True:
    ret,frame=cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    count+=1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
   

#    print(px)
    list1=[]

    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        cx=int(x1+x2)//2
        cy=int(y1+y2)//2
        w,h=x2-x1,y2-y1
        result=cv2.pointPolygonTest(np.array(area1,np.int32),((cx,cy)),False)
        if result >= 0:
#        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),-1)
            cvzone.cornerRect(frame,(x1,y1,w,h),3,2)
            cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
            cvzone.putTextRect(frame,f'person',(x1,y1),1,1)
            list1.append(cx)

           
       
    ctr=len(list1)

    cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,255,0),2)
    #cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,0,255),2)
    cvzone.putTextRect(frame,f'Counter:- {ctr}',(50,60),1,1)

    if ctr > 8:
        print("Queue Full!")
        cvzone.putTextRect(frame,f'Queue Full! Raise Alarm!',(350,40),1,1)
    if ctr == 5:
        print("Queue is near to full!")
        cvzone.putTextRect(frame,f'Queue is Near to Getting Full!',(350,40),1,1)
    if ctr <= 3 or ctr==4:
        print("Queue is Almost Empty")
        cvzone.putTextRect(frame,f'Queue is Almost Empty!',(350,40),1,1)
        
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()