import cv2
import socketio
import requests
from pytz import timezone
from datetime import datetime
from libraries.centroidtracker import CentroidTracker
from libraries.trackableobject import TrackableObject

classNames = []
classFile = "models/coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "models/frozen_inference_graph.pb"

PATH_CAM = "rtsp://admin:admin1234@192.168.8.101:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"
SOCKET_LINK = "http://116.193.191.157:8081" 
SAVED_COUNT = 'http://116.193.191.157:8081/database/history/HKCCTV_IM001_IN/guest_count/'
format_date = "%d_%m_%Y"

vanue = "Convention Hall - GOR Kadrie Oening (Pintu Masuk)"

tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)
sio = socketio.Client()

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def getObjects(img, thres, nms, draw=True, objects=[]):

    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    if len(objects) == 0: objects = classNames
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                if (draw):
                    detecbox.append(box)
    return img

def getHour():
    hours = datetime.now(timezone("Asia/Makassar")).strftime("%#H")
    return hours

def getDate():
    date_now = datetime.now(timezone('Asia/Makassar')).strftime(format_date)
    return date_now

if __name__ == "__main__":

    H = None
    W = None

    people_in = 0
    people_out = 0
    hour = 0
    starting_status = True
    error_status = False
    trackableObjects = {}

    prev_frame = 0
    new_frame = 0

    while True:
        try:

            if starting_status == True:
                response = requests.get(SAVED_COUNT+getDate()+"?key="+getHour())
                if response.status_code!=200:
                    people_in =  0
                    people_out = 0
                    response =  requests.post(SAVED_COUNT+getDate()+"?key="+getHour(), json={'name':vanue, 'in':people_in, 'out':people_out})
                else:
                    data_count = response.json()
                    people_in, people_out = data_count['in'], data_count['out']
                starting_status = False
            
            if error_status == True:
                date_day = getDate()
                hour = getHour()
                response =  requests.post(SAVED_COUNT+getDate()+"?key="+getHour(), json={'name':vanue, 'in':people_in, 'out':people_out})
                error_status = False

            if hour != getHour(): 
                if hour > getHour():
                    starting_status = True
                error_status = True

            success, img = cap.read()
            (H,W) = img.shape[:2]

            detecbox = []

            if(sio.connected != True):
                sio.connect(SOCKET_LINK)
            
            result = getObjects(img,0.5,0.2, objects=['person'])
            obj = tracker.update(detecbox)
            for(objId, centroid) in obj.items():
                to =trackableObjects.get(objId, None)

                if to is None:
                    to = TrackableObject(objId, centroid)
                else:
                    y = [c[0] for c in to.centroids]
                    to.centroids.append(centroid)

                    if not to.counted:
                        if y[0] > W//2 and centroid[0] < W//2:
                            people_out+=1
                            to.counted = True
                        elif y[0] < W//2 and centroid[0] > W//2:
                            people_in+=1
                            to.counted = True

                trackableObjects[objId] = to
            
            sio.emit("guest_count", {"in":str(people_in), "out":str(people_out)})

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        except Exception as e:
            cap = cv2.VideoCapture(PATH_CAM)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,0)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            print("Error : ", e)
            error_status = True
            continue

    sio.wait()
    cap.release()
    cv2.destroyAllWindows()