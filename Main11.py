import numpy as np
import time
import cv2
import os
import numpy as np



args = {"confidence": 0.8, "threshold": 0.5}
flag = False

labelsPath = "D:/BDU/PP/HelmetNumber/HelmetNumber/NumplateImageNewPy/helmet.names"
LABELS = open(labelsPath).read().strip().split("\n")
final_classes = ['Helmet']

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

weightsPath = os.path.abspath("D:/BDU/PP/HelmetNumber/HelmetNumber/NumplateImageNewPy/yolov3-helmet.weights")
configPath = os.path.abspath("D:/BDU/PP/HelmetNumber/HelmetNumber/NumplateImageNewPy/yolov3-helmet.cfg")

# print(configPath, "\n", weightsPath)

flagg1 = 0
flagg2 = 0
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print(type(ln), ln)


vs = cv2.VideoCapture(0)
writer = None
(W, H) = (None, None)

flag = True
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                            args["threshold"])

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            if (LABELS[classIDs[i]] in final_classes):


                # playsound('alarm.wav')
                if (flag):
                    # alert()
                    flag = False
                    # async_email(LABELS[classIDs[i]])
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                           confidences[i])
                cv2.putText(frame, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.imshow("Color", frame)
                flagg1 += 1
                # print(flag)

                if (flagg1 == 10):
                    print("Helmet Found!")

    else:
        flag = True
        print(0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for x, y, w, h in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.imshow('Color', frame)

        flagg2 += 1
        # print(flag)

        if (flagg2 == 10):
            flagg2 = 0
            print("Helmet Not  Found!")



    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
vs.release()
cv2.destroyAllWindows()


