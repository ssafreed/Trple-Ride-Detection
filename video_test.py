import cv2
import numpy as np
from shapely.geometry import Polygon

net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')

classes =[]
with open('coco.names','r') as f:
    classes= f.read().splitlines()

#print(classes)

cap = cv2.VideoCapture('test.mp4')

while True:

    _,img = cap.read()
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), (0,0,0), swapRB=True, crop=False)
    #for b in blob:
    #    for n,img_blob in enumerate(b):
    #        cv2.imshow(str(n),img_blob)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence> 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size = (len(boxes), 3))


    bike_list=[]
    persons_list = []
    new_bike_list = []


    for i in indexes.flatten():
        x,y,w,h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i],2))
        color = colors[i]
        if label =='motorbike':
            new_bike_list.append([x,y,w,h])
            # cv2.rectangle(img, (x,y-40), (x+w,y+h), color,3)
            m1 = (x,y+h)
            m2 = (x+w,y+h)
            m3 = (x+w,y-40)
            m4 = (x,y-40)
            m_p = Polygon([m1,m2,m3,m4])
            bike_list.append(m_p)
            # cv2.putText(img, label+" "+confidence,(x,y+20), font, 2, (255,255,255), 2)
        else:
            if label == 'person':
                # cv2.rectangle(img, (x,y), (x+w,y+h), color,3)
                p1 = (x,y+h)
                p2 = (x+w,y+h)
                p3 = (x+w,y)
                p4 = (x,y)
                p_p = Polygon([p1,p2,p3,p4])
                persons_list.append(p_p)            
                # cv2.putText(img, label+" "+confidence,(x,y+20), font, 2, (255,255,255), 2)
            else:
                # cv2.rectangle(img, (x,y), (x+w,y+h), color,3)
                # cv2.putText(img, label+" "+confidence,(x,y+20), font, 2, (255,255,255), 2)
                pass
    for i in range(len(persons_list)-1):
        if persons_list[i].intersection(persons_list[i+1]).area>=0.8:
            # print(persons_list[i].intersection(persons_list[i+1]).area)
            persons_list.append(persons_list[i].intersection(persons_list[i+1]))

    bike_dict = {}
    for i in range(len(bike_list)):
        intersection_list = []
        for j in persons_list:
            intersect = bike_list[i].intersection(j).area
            intersection_list.append(intersect) 
        if 'bike_'+str(i) not in bike_dict:
            bike_dict['bike_'+str(i)] = intersection_list


    new_dict ={}
    for k,v in bike_dict.items():
        c=0
        for i in v:
            if i>=0.8:
                c+=1
            else:
                c=c
        if k not in new_dict:
            new_dict[k]=c


    for k in new_dict.keys():
        if new_dict[k]>=3:
            print(f'Triple Ride Detected at {k}')
        else:
            print(f'Triple Ride not Detected at {k}')


    print(new_dict.values())
    
    l=list(new_dict.values())
    for i in range(len(l)):       
        x,y,w,h = new_bike_list[i]
        if l[i]<=2:
            cv2.rectangle(img, (x,y-40), (x+w,y+h), (255,255,255),3)
            cv2.putText(img, "No Triple Ride",(x,y+20), font, 2, (255,255,255), 2)
        else:
            cv2.rectangle(img, (x,y-40), (x+w,y+h), (0,0,0),3)
            cv2.putText(img,"Triple Ride",(x,y-30), font, 2, (255,255,255), 2)

    cv2.imshow('Image',img)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
