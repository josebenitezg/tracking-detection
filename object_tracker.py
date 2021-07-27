import cv2 as cv
import sys
import numpy as np

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image

tracker = cv.TrackerCSRT_create()

video_path = 'soccer-ball.mp4'
cap = cv.VideoCapture(video_path)

# Load names of classes, should be placed in a folder named model
classesFile = 'model/coco.names'
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "model/yolov3.cfg"
modelWeights = "model/yolov3.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def object_detector(frame):

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    labels = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        if classes[classIds[i]]=='sports ball':
            print('[INFO] {} class was founded'.format(classes[classIds[i]]))
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            boxes.append((left, top, width, height))
            return boxes[i]
        else:
            continue

bbox = None
outputFile = "tracking_out.avi"
#vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
new_bbox = None
while(True):
    ok, frame = cap.read()

    if not ok:
        print('Cannot read video file')
        sys.exit()

    bbox = object_detector(frame)
    print('The inicial bounding box is {}'.format(bbox))
    if(bbox):

        ok = tracker.init(frame, tuple(bbox))

        while True:
            # Read a new frame
            ok, frame = cap.read()
            if not ok:
                break

            ok, bbox = tracker.update(frame)

            # Draw bounding box
            if ok:
                print('[INFO] Tracking success')
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv.rectangle(frame, p1, p2, (0,255,0), 2, 1)

                cv.imshow("Tracking", frame)

                
            else :
                # Tracking failure
                print('[INFO] Tracking failed')
                cv.putText(frame, "Tracking failure detected, detecting object with Yolo", (20,120), cv.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2, cv.LINE_AA)
                print('[INFO] Updating bbox with object detection')
                new_bbox = object_detector(frame)
                print('[INFO] {} is the updated bbox from object detection'.format(new_bbox))
                if(new_bbox):
                    new_bbox = tuple(new_bbox)
                    
                    p1 = (int(new_bbox[0]), int(new_bbox[1]))
                    p2 = (int(new_bbox[0] + new_bbox[2]), int(new_bbox[1] + new_bbox[3]))
                    
                    cv.rectangle(frame, p1, p2, (0,0,255), 2, 1) 

                    cv.imshow("Tracking", frame)
                    tracker = cv.TrackerCSRT_create()
                    ok = tracker.init(frame, new_bbox)
                    print('[INFO] Setting new bbox to {}'.format(new_bbox))
                    print(ok)
                else:
                    cv.imshow("Tracking", frame)

            # Exit if ESC pressed
            k = cv.waitKey(1) & 0xff
            if k == 27 : break

        cap.release()