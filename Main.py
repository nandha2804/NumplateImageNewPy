from flask import Flask, render_template, flash, request, session
from flask import render_template, redirect, url_for, request

import mysql.connector
import datetime
import time

app: Flask = Flask(__name__)
app.config['DEBUG']
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


@app.route("/")
def homepage():
    return render_template('index.html')


@app.route("/Home")
def Home():
    return render_template('index.html')


@app.route("/AdminLogin")
def AdminLogin():
    return render_template('AdminLogin.html')


@app.route("/UserLogin")
def UserLogin():
    return render_template('UserLogin.html')


@app.route("/NewUser")
def NewUser():
    return render_template('NewUser.html')


@app.route("/adminlogin", methods=['GET', 'POST'])
def adminlogin():
    error = None
    if request.method == 'POST':
        if request.form['uname'] == 'admin' or request.form['password'] == 'admin':
            conn = mysql.connector.connect(user='root', password='', host='localhost', database='1numberhelmetdb')
            cursor = conn.cursor()
            cur = conn.cursor()
            cur.execute("SELECT * FROM regtb")
            data = cur.fetchall()
            return render_template('AdminHome.html', data=data)

        else:
            return render_template('index.html', error=error)


@app.route("/AdminHome")
def AdminHome():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1numberhelmetdb')
    cursor = conn.cursor()
    cur = conn.cursor()
    cur.execute("SELECT * FROM regtb")
    data = cur.fetchall()
    return render_template('AdminHome.html', data=data)


@app.route("/remove")
def remove():
    did = request.args.get('did')

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1numberhelmetdb')
    cursor = conn.cursor()
    cursor.execute("delete from regtb  where Id='" + did + "' ")
    conn.commit()
    conn.close()

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1numberhelmetdb')
    # cursor = conn.cursor()
    cur = conn.cursor()
    cur.execute("SELECT * FROM regtb ")
    data = cur.fetchall()

    return render_template('AdminHome.html', data=data)


@app.route("/Report")
def Report():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1numberhelmetdb')

    cur = conn.cursor()
    cur.execute("SELECT * FROM entrytb")
    data = cur.fetchall()
    return render_template('AdminReport.html', data=data)


import cv2
import numpy as np
from skimage.filters import threshold_local
import tensorflow as tf
from skimage import measure
import imutils
import pytesseract
import re
import mysql.connector


def sort_cont(character_contours):
    """
    To sort contours from left to right
    """
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in character_contours]
    (character_contours, boundingBoxes) = zip(*sorted(zip(character_contours, boundingBoxes),
                                                      key=lambda b: b[1][i], reverse=False))
    return character_contours


def segment_chars(plate_img, fixed_width):
    """
    extract Value channel from the HSV format of image and apply adaptive thresholding
    to reveal the characters on the license plate
    """
    V = cv2.split(cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV))[2]

    T = threshold_local(V, 29, offset=15, method='gaussian')

    thresh = (V > T).astype('uint8') * 255

    thresh = cv2.bitwise_not(thresh)

    # resize the license plate region to a canoncial size
    plate_img = imutils.resize(plate_img, width=fixed_width)
    thresh = imutils.resize(thresh, width=fixed_width)
    bgr_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # perform a connected components analysis and initialize the mask to store the locations
    # of the character candidates
    labels = measure.label(thresh, neighbors=8, background=0)

    charCandidates = np.zeros(thresh.shape, dtype='uint8')

    # loop over the unique components
    characters = []
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask to display only connected components for the
        # current label, then find contours in the label mask
        labelMask = np.zeros(thresh.shape, dtype='uint8')
        labelMask[labels == label] = 255

        cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        # ensure at least one contour was found in the mask
        if len(cnts) > 0:

            # grab the largest contour which corresponds to the component in the mask, then
            # grab the bounding box for the contour
            c = max(cnts, key=cv2.contourArea)
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

            # compute the aspect ratio, solodity, and height ration for the component
            aspectRatio = boxW / float(boxH)
            solidity = cv2.contourArea(c) / float(boxW * boxH)
            heightRatio = boxH / float(plate_img.shape[0])

            # determine if the aspect ratio, solidity, and height of the contour pass
            # the rules tests
            keepAspectRatio = aspectRatio < 1.0
            keepSolidity = solidity > 0.15
            keepHeight = heightRatio > 0.5 and heightRatio < 0.95

            # check to see if the component passes all the tests
            if keepAspectRatio and keepSolidity and keepHeight and boxW > 14:
                # compute the convex hull of the contour and draw it on the character
                # candidates mask
                hull = cv2.convexHull(c)

                cv2.drawContours(charCandidates, [hull], -1, 255, -1)

    _, contours, hier = cv2.findContours(charCandidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sort_cont(contours)
        addPixel = 4  # value to be added to each dimension of the character
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            if y > addPixel:
                y = y - addPixel
            else:
                y = 0
            if x > addPixel:
                x = x - addPixel
            else:
                x = 0
            temp = bgr_thresh[y:y + h + (addPixel * 2), x:x + w + (addPixel * 2)]

            characters.append(temp)
        return characters
    else:
        return None


class PlateFinder:
    def __init__(self):
        self.min_area = 4500  # minimum area of the plate
        self.max_area = 30000  # maximum area of the plate

        self.element_structure = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(22, 3))

    def preprocess(self, input_img):
        imgBlurred = cv2.GaussianBlur(input_img, (7, 7), 0)  # old window was (5,5)
        gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)  # convert to gray
        sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)  # sobelX to get the vertical edges
        ret2, threshold_img = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        element = self.element_structure
        morph_n_thresholded_img = threshold_img.copy()
        cv2.morphologyEx(src=threshold_img, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_n_thresholded_img)
        return morph_n_thresholded_img

    def extract_contours(self, after_preprocess):
        _, contours, _ = cv2.findContours(after_preprocess, mode=cv2.RETR_EXTERNAL,
                                          method=cv2.CHAIN_APPROX_NONE)
        return contours

    def clean_plate(self, plate):
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        _, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)  # index of the largest contour in the area array

            max_cnt = contours[max_index]
            max_cntArea = areas[max_index]
            x, y, w, h = cv2.boundingRect(max_cnt)
            rect = cv2.minAreaRect(max_cnt)
            rotatedPlate = plate
            if not self.ratioCheck(max_cntArea, rotatedPlate.shape[1], rotatedPlate.shape[0]):
                return plate, False, None
            return rotatedPlate, True, [x, y, w, h]
        else:
            return plate, False, None

    def check_plate(self, input_img, contour):
        min_rect = cv2.minAreaRect(contour)
        if self.validateRatio(min_rect):
            x, y, w, h = cv2.boundingRect(contour)
            after_validation_img = input_img[y:y + h, x:x + w]
            after_clean_plate_img, plateFound, coordinates = self.clean_plate(after_validation_img)
            if plateFound:
                characters_on_plate = self.find_characters_on_plate(after_clean_plate_img)
                if (characters_on_plate is not None and len(characters_on_plate) == 10):
                    x1, y1, w1, h1 = coordinates
                    coordinates = x1 + x, y1 + y
                    after_check_plate_img = after_clean_plate_img
                    return after_check_plate_img, characters_on_plate, coordinates
        return None, None, None

    def find_possible_plates(self, input_img):
        """
        Finding all possible contours that can be plates
        """
        plates = []
        self.char_on_plate = []
        self.corresponding_area = []

        self.after_preprocess = self.preprocess(input_img)
        possible_plate_contours = self.extract_contours(self.after_preprocess)

        for cnts in possible_plate_contours:
            plate, characters_on_plate, coordinates = self.check_plate(input_img, cnts)
            if plate is not None:
                plates.append(plate)
                self.char_on_plate.append(characters_on_plate)
                self.corresponding_area.append(coordinates)

        if (len(plates) > 0):
            return plates
        else:
            return None

    def find_characters_on_plate(self, plate):

        charactersFound = segment_chars(plate, 400)
        if charactersFound:
            return charactersFound

    # PLATE FEATURES
    def ratioCheck(self, area, width, height):
        min = self.min_area
        max = self.max_area

        ratioMin = 3
        ratioMax = 6

        ratio = float(width) / float(height)
        if ratio < 1:
            ratio = 1 / ratio

        if (area < min or area > max) or (ratio < ratioMin or ratio > ratioMax):
            return False
        return True

    def preRatioCheck(self, area, width, height):
        min = self.min_area
        max = self.max_area

        ratioMin = 2.5
        ratioMax = 7

        ratio = float(width) / float(height)
        if ratio < 1:
            ratio = 1 / ratio

        if (area < min or area > max) or (ratio < ratioMin or ratio > ratioMax):
            return False
        return True

    def validateRatio(self, rect):
        (x, y), (width, height), rect_angle = rect

        if (width > height):
            angle = -rect_angle
        else:
            angle = 90 + rect_angle

        if angle > 15:
            return False
        if (height == 0 or width == 0):
            return False

        area = width * height
        if not self.preRatioCheck(area, width, height):
            return False
        else:
            return True


class NeuralNetwork:
    def __init__(self):
        self.model_file = "D:/BDU/PP/HelmetNumber/HelmetNumber/NumplateImageNewPy/model/binary_128_0.50_ver3.pb"
        self.label_file = "D:/BDU/PP/HelmetNumber/HelmetNumber/NumplateImageNewPy/model/binary_128_0.50_labels_ver2.txt"
        self.label = self.load_label(self.label_file)
        self.graph = self.load_graph(self.model_file)
        self.sess = tf.Session(graph=self.graph)

    def load_graph(self, modelFile):
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        with open(modelFile, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)
        return graph

    def load_label(self, labelFile):
        label = []
        proto_as_ascii_lines = tf.gfile.GFile(labelFile).readlines()
        for l in proto_as_ascii_lines:
            label.append(l.rstrip())
        return label

    def convert_tensor(self, image, imageSizeOuput):
        """
    takes an image and tranform it in tensor
    """
        image = cv2.resize(image, dsize=(imageSizeOuput, imageSizeOuput), interpolation=cv2.INTER_CUBIC)
        np_image_data = np.asarray(image)
        np_image_data = cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
        np_final = np.expand_dims(np_image_data, axis=0)
        return np_final

    def label_image(self, tensor):

        input_name = "import/input"
        output_name = "import/final_result"

        input_operation = self.graph.get_operation_by_name(input_name)
        output_operation = self.graph.get_operation_by_name(output_name)

        results = self.sess.run(output_operation.outputs[0],
                                {input_operation.outputs[0]: tensor})
        results = np.squeeze(results)
        labels = self.label
        top = results.argsort()[-1:][::-1]
        return labels[top[0]]

    def label_image_list(self, listImages, imageSizeOuput):
        plate = ""
        for img in listImages:
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            plate = plate + self.label_image(self.convert_tensor(img, imageSizeOuput))
        return plate, len(plate)


@app.route("/Verify")
def Verify():
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

    vs = cv2.VideoCapture(0)
    writer = None
    (W, H) = (None, None)

    flag = True
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
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

                    if (flagg1 == 30):
                        print("Helmet Found!")
                        vs.release()
                        cv2.destroyAllWindows()
                        # cur = conn.cursor()
                        # cur.execute("SELECT * FROM entrytb ")
                        # data = cur.fetchall()

                        return "Helmet Found!"

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

            if (flagg2 == 30):
                flagg2 = 0
                print("Helmet Not  Found!")
                vs.release()
                cv2.destroyAllWindows()
                return vv()

        if cv2.waitKey(1) == ord('q'):
            break

    # release the webcam and destroy all active windows
    vs.release()
    cv2.destroyAllWindows()


def vv():
    session['vno'] = ''
    findPlate = PlateFinder()

    # Initialize the Neural Network
    model = NeuralNetwork()

    count = 0

    cap = cv2.VideoCapture(0)
    while (True):

        ret, img = cap.read()
        if ret == True:
            count += 1
            print(count)

            if count < 500:
                cv2.imshow('original video', img)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

                possible_plates = findPlate.find_possible_plates(img)
                pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

                if possible_plates is not None:

                    for i, p in enumerate(possible_plates):

                        chars_on_plate = findPlate.char_on_plate[i]
                        recognized_plate, _ = model.label_image_list(chars_on_plate, imageSizeOuput=128)
                        print(recognized_plate)

                        cv2.imshow('plate', p)
                        predicted_result = pytesseract.image_to_string(p, lang='eng',
                                                                       config='--oem 3 --psm 6 -c tessedit_char_whitelist = ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                        print(predicted_result)

                        vno = re.sub(r"[^a-zA-Z0-9]", "", predicted_result)
                        print(vno)

                        ts = time.time()
                        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

                        conn = mysql.connector.connect(user='root', password='', host='localhost',
                                                       database='1numberhelmetdb')
                        cursor = conn.cursor()
                        cursor.execute("select * from regtb where VehicleNo='" + str(vno) + "' ")
                        data = cursor.fetchone()
                        if data is None:
                            print("VehilceNo Not Found")

                        else:
                            unm = data[8]
                            mob = data[6]
                            conn = mysql.connector.connect(user='root', password='', host='localhost',
                                                           database='1numberhelmetdb')
                            cursor = conn.cursor()
                            cursor.execute(
                                "select * from entrytb where Date='" + str(date) + "' and VehicleNo='" + str(vno) + "'")
                            data = cursor.fetchone()
                            if data is None:
                                conn = mysql.connector.connect(user='root', password='', host='localhost',
                                                               database='1numberhelmetdb')
                                cursor = conn.cursor()
                                cursor.execute(
                                    "insert into entrytb values('','" + str(vno) + "','" + unm + "','" + str(
                                        date) + "','" + str(
                                        timeStamp) + "','500','NotPaid')")
                                conn.commit()
                                conn.close()
                                print("Fine Amount Info Saved")

                                sendmsg(mob, " Fine Amount For Helmet Not Wearing  RS.500")

                                vnoo = vno
                                cap.release()
                                cv2.destroyAllWindows()

                                conn = mysql.connector.connect(user='root', password='', host='localhost',
                                                               database='1numberhelmetdb')
                                # cursor = conn.cursor()
                                cur = conn.cursor()
                                cur.execute("SELECT * FROM entrytb ")
                                data = cur.fetchall()

                                return render_template('AdminReport.html', data=data)


                            else:
                                cap.release()
                                cv2.destroyAllWindows()
                                return "Already Fine Amount Info Saved"

                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break

            else:

                cv2.imwrite("alert.jpg", img)
                sendmail()
                cap.release()
                cv2.destroyAllWindows()
                return "Numberplate  Not Found!"





        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    return render_template('AdminHome.html')

    # if(session['vno'] !=''):


def sendmail():
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders

    fromaddr = "sampletest685@gmail.com"
    toaddr = "mathansethupathy@gmail.com"

    # instance of MIMEMultipart
    msg = MIMEMultipart()

    # storing the senders email address
    msg['From'] = fromaddr

    # storing the receivers email address
    msg['To'] = toaddr

    # storing the subject
    msg['Subject'] = "Alert"

    # string to store the body of the mail
    body = "Without Number Plate Detection"

    # attach the body with the msg instance
    msg.attach(MIMEText(body, 'plain'))

    # open the file to be sent
    filename = "alert.jpg"
    attachment = open("alert.jpg", "rb")

    # instance of MIMEBase and named as p
    p = MIMEBase('application', 'octet-stream')

    # To change the payload into encoded form
    p.set_payload((attachment).read())

    # encode into base64
    encoders.encode_base64(p)

    p.add_header('Content-Disposition', "attachment; filename= %s" % filename)

    # attach the instance 'p' to instance 'msg'
    msg.attach(p)

    # creates SMTP session
    s = smtplib.SMTP('smtp.gmail.com', 587)

    # start TLS for security
    s.starttls()

    # Authentication
    s.login(fromaddr, "hneucvnontsuwgpj")

    # Converts the Multipart msg into a string
    text = msg.as_string()

    # sending the mail
    s.sendmail(fromaddr, toaddr, text)

    # terminating the session
    s.quit()


@app.route("/newsuer", methods=['GET', 'POST'])
def newsuer():
    if request.method == 'POST':
        vno = request.form['vno']
        name = request.form['name']
        gender = request.form['gender']
        Age = request.form['Age']
        email = request.form['email']
        pnumber = request.form['pnumber']
        address = request.form['address']
        uname = request.form['uname']
        password = request.form['password']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1numberhelmetdb')
        cursor = conn.cursor()
        cursor.execute(
            "insert into regtb values('','" + vno + "','" + name + "','" + gender + "','" + Age + "','" + email + "','" + pnumber + "','" + address + "','" + uname + "','" + password + "')")
        conn.commit()
        conn.close()

    return render_template("UserLogin.html")


@app.route("/AUserSearch", methods=['GET', 'POST'])
def AUserSearch():
    if request.method == 'POST':
        date = request.form['date']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1numberhelmetdb')
        cur = conn.cursor()
        cur.execute("SELECT * FROM entrytb where date='" + date + "'")
        data = cur.fetchall()

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1numberhelmetdb')
        cursor = conn.cursor()
        cursor.execute("SELECT  *  FROM entrytb where  date='" + date + "' and Status !='Paid' ")
        data1 = cursor.fetchall()

        for i in data1:
            uname = i[2]

            conn = mysql.connector.connect(user='root', password='', host='localhost', database='1numberhelmetdb')
            cursor = conn.cursor()
            cursor.execute("SELECT  *  FROM regtb where  username='" + uname + "'")
            data11 = cursor.fetchone()

            if data11:
                mobile = data11[6]
                print(mobile)
                sendmsg(mobile, "Licence's Blocked More Info visit RTO")

        return render_template('AdminReport.html', data=data)


global vno
global Email


def examvales1():
    vvno = session['vno']
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1numberhelmetdb')
    cursor = conn.cursor()
    cursor.execute("SELECT  *  FROM regtb where  VehicleNo	='" + vvno + "'")
    data = cursor.fetchone()

    if data:
        vno = data[0]
        Email = data[4]


    else:
        return 'Incorrect username / password !'
    return vno, Email


def sendmsg(targetno, message):
    import requests
    requests.post(
        "http://smsserver9.creativepoint.in/api.php?username=fantasy&password=596692&to=" + targetno + "&from=FSSMSS&message=Dear user  your msg is " + message + " Sent By FSMSG FSSMSS&PEID=1501563800000030506&templateid=1507162882948811640")


@app.route("/userlogin", methods=['GET', 'POST'])
def userlogin():
    error = None
    if request.method == 'POST':
        username = request.form['uname']
        password = request.form['password']
        session['uname'] = request.form['uname']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1numberhelmetdb')
        cursor = conn.cursor()
        cursor.execute("SELECT * from regtb where username='" + username + "' and Password='" + password + "'")
        data = cursor.fetchone()
        if data is None:
            alert = 'Username or Password is wrong'
            return render_template('goback.html', data=alert)
        else:

            conn = mysql.connector.connect(user='root', password='', host='localhost', database='1numberhelmetdb')
            # cursor = conn.cursor()
            cur = conn.cursor()
            cur.execute("SELECT * FROM regtb where username='" + username + "' and Password='" + password + "'")
            data = cur.fetchall()

            return render_template('UserHome.html', data=data)


@app.route("/UserHome")
def UserHome():
    uname = session['uname']

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1numberhelmetdb')
    # cursor = conn.cursor()
    cur = conn.cursor()
    cur.execute("SELECT * FROM regtb where username='" + uname + "' ")
    data = cur.fetchall()
    return render_template('UserHome.html', data=data)


@app.route("/UserReport")
def UserReport():
    uname = session['uname']

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1numberhelmetdb')
    # cursor = conn.cursor()
    cur = conn.cursor()
    cur.execute("SELECT * FROM  entrytb where username='" + uname + "' and Status !='Paid' ")
    data = cur.fetchall()
    return render_template('UserReport.html', data=data)


@app.route("/UserSearch", methods=['GET', 'POST'])
def UserSearch():
    if request.method == 'POST':
        date = request.form['date']
        uname = session['uname']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1numberhelmetdb')

        cur = conn.cursor()
        cur.execute("SELECT * FROM entrytb where UserName='" + uname + "' and  date='" + date + "'")
        data = cur.fetchall()

        return render_template('UserReport.html', data=data)


@app.route("/pay")
def pay():
    did = request.args.get('did')
    session['bid'] = did

    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1numberhelmetdb')
    cursor = conn.cursor()
    cursor.execute("SELECT  FineAmount  FROM  entrytb where  id	='" + did + "'")
    data = cursor.fetchone()

    if data:
        Amt = data[0]

    return render_template('Payment.html', Amount=Amt)


@app.route("/Payment", methods=['GET', 'POST'])
def Payment():
    did = session['bid']
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='1numberhelmetdb')
    cursor = conn.cursor()
    cursor.execute("Update  entrytb set Status='Paid' where id='" + did + "'")
    conn.commit()
    conn.close()

    alert = "Payment Sucessful! "
    return render_template('goback.html', data=alert)


@app.route("/numupdate", methods=['GET', 'POST'])
def numupdate():
    if request.method == 'POST':
        num = request.form['num']
        uname = session['uname']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1numberhelmetdb')
        cursor = conn.cursor()
        cursor.execute("Update regtb set VehicleNo='" + num + "' where Username='" + uname + "'")
        conn.commit()
        conn.close()

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='1numberhelmetdb')
        # cursor = conn.cursor()
        cur = conn.cursor()
        cur.execute("SELECT * FROM regtb where UserName='" + uname + "' ")
        data = cur.fetchall()

        return render_template('UserHome.html', data=data)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
