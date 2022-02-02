# YOLO object detection : Process video 
import cv2 as cv
import numpy as np
import time
import os

WHITE = (255, 255, 255)
img = None
img0 = None
outputs = None

# Load names of classes and get random colors
classes = open('coco/coco.names').read().strip().split('\n')
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

# Give the configuration and weight files for the model and load the network.
net = cv.dnn.readNetFromDarknet('yolo/yolov3.cfg', 'yolo/yolov3.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# determine the output layer
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def load_image(path, useVideo = False):
    global img, img0, outputs, ln

    if useVideo == False:
        img0 = cv.imread(path)

    img = img0.copy()
    
    blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    t0 = time.time()
    outputs = net.forward(ln)
    t = time.time() - t0

    # combine the 3 output groups into 1 (10647, 85)
    # large objects (507, 85)
    # medium objects (2028, 85)
    # small objects (8112, 85)
    outputs = np.vstack(outputs)

    post_process(img, outputs, 0.5)

    return img

def make_video_from_images(fileFormat,fileName,FPS=30.0,width=1280,height=720,displayImages=False):
    """
    The function converts set of images into video using ffmpeg
    :param fileFormat: directory of the image format -> "foo/foo%d.jpeg"
    :param fileName: output filename
    :param FPS: framerate of the output video
    :param width: image width
    :param height: image height
    :param displayImages: display the images while making video
    :return: None
    """
    commandForVideo = 'ffmpeg -f image2 -framerate {0} -i {1} -s {2}x{3} {4}'.format(FPS, fileFormat, width, height, fileName)
    print(commandForVideo)
    os.system(commandForVideo)

def post_process(img, outputs, conf):
    H, W = img.shape[:2]

    boxes = []
    confidences = []
    classIDs = []

    for output in outputs:
        scores = output[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > conf:
            x, y, w, h = output[:4] * np.array([W, H, W, H])
            p0 = int(x - w//2), int(y - h//2)
            p1 = int(x + w//2), int(y + h//2)
            boxes.append([*p0, int(w), int(h)])
            confidences.append(float(confidence))
            classIDs.append(classID)
            # cv.rectangle(img, p0, p1, WHITE, 1)

    indices = cv.dnn.NMSBoxes(boxes, confidences, conf, conf-0.1)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def trackbar(x):
    global img
    conf = x/100
    img = img0.copy()
    post_process(img, outputs, conf)
    #cv.displayOverlay('window', f'confidence level={conf}')
    cv.imshow('window', img)

cv.namedWindow('window')
cv.createTrackbar('confidence', 'window', 50, 100, trackbar)
# cap = cv.VideoCapture("video/20190802_PigeonGaze_trial02.2119571.20190802181457.avi")
cap = cv.VideoCapture("video/cockateal_2.mp4")
FPS = 30
totalFrames = cap.get(cv.CAP_PROP_FRAME_COUNT)
name = "video/output.mp4"
# fourcc = cv.VideoWriter_fourcc(*"VIDX")
# videoOut = cv.VideoWriter(name, fourcc, FPS, (imageWidth, imageHeight), True)  # 400,400 # 0x00000020
useVideo = True
frameCounter = 0

files = []

while(True):
    ret, frame = cap.read()
    if ret == False:
        print("Could not read the image")
        break

    img0 = frame.copy()

    imageHeight, imageWidth  = img0.shape[:2]
    result = load_image("images/barnbirds.jpg", useVideo)
    cv.imshow('window', result)
    imgPrefix = "images/output/"
    imgName = imgPrefix + str(frameCounter)+ ".jpg"
    cv.imwrite(imgName,img)
    files.append(imgName)
    # cv.displayOverlay('window', f'forward propagation time={t:.3}')
    k = cv.waitKey(10)

    #
    # if useVideo:
    #     videoOut.write(result)

    frameCounter += 1
    if k == ord('q'):
        break

    if k == ord("j"):  # Jumps 50 Frames
        nextFrame = cap.get(cv.CAP_PROP_POS_FRAMES)
        newFrame = nextFrame + 1000
        if newFrame < totalFrames:
            cap.set(cv.CAP_PROP_POS_FRAMES, newFrame)
        continue

if useVideo:
    fileName = "video/output.mp4"
    if os.path.isfile("video/output.mp4"):
        os.remove("video/output.mp4")
        print("Deleting old file")

    fileFormat = "images/output/%d.jpg"
    make_video_from_images(fileFormat, fileName, FPS= FPS, width= imageWidth, height= imageHeight,  displayImages=True)
    for f in files:
        os.remove(f)
    files.clear()
#
# if useVideo:
#     videoOut.release()
# load_image('images/barnbirds.jpg')
# load_image('images/barnbirds2.jpg')
# load_image('images/barnbirds3.jpg')
# load_image('images/image284.jpg')
# load_image('images/image286.jpg')
# load_image('images/image289.jpg')

cv.destroyAllWindows()