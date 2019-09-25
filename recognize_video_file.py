# USAGE
# python recognize_video.py --detector face_detection_model \
#	--embedding-model openface_nn4.small2.v1.t7 \
#	--recognizer output/recognizer.pickle \
#	--le output/le.pickle

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import datetime
import math

from align_faces_util import AlignFace


def show_image(name, image, verbose_delay=0):
    if image is None:
        return
    cv2.imshow(name, image)
    cv2.waitKey(verbose_delay)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True, help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True, help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True, help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True, help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-i", "--image", required=True, help="path to video")
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-a", "--align", action='store_true', help="do source face alignment before recognition")
ap.add_argument("-v", "--verbose", required=False, help="show debug info", action="count", default=0)
ap.add_argument("-sf", "--saveface", required=False, help="save detected face", default=False)

args = vars(ap.parse_args())

verbose = args["verbose"]
confidence_ = args["confidence"]
is_save_face = args["saveface"]

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
                              "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
# time.sleep(2.0)

cap = cv2.VideoCapture(args["image"])
frame_count = cap.get(7)
frame_rate = cap.get(5)  # frame rate
duration = frame_count / frame_rate
t = str(datetime.timedelta(seconds=duration))
print(t)

# start the FPS throughput estimator
# fps = FPS().start()
# loop over frames from the video file stream
# while True:


face_aligner = AlignFace(args["shape_predictor"])

seconds = 0
while cap.isOpened():
    frame_id = cap.get(1)  # current frame number
    ret, frame = cap.read()

    if not ret:
        break

    if 0 != frame_id % math.floor(frame_rate):
        continue

    seconds = seconds + 1

    sec = int(frame_id / frame_rate)
    t = str(datetime.timedelta(seconds=sec))

    # grab the frame from the threaded video stream
    # frame = vs.read()

    # resize the frame to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
    #frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    cnt = 0
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > confidence_:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 200 or fH < 200:
                continue

            do_align = args["align"]
            if do_align:
                faces = face_aligner.align(face)
                if faces and len(faces) > 0:
                    face = faces[0]
                else:
                    continue

            if verbose >= 2:
                show_image('face', face, 0)

            if is_save_face:
                filename = os.path.join(is_save_face,
                                   "face_" + t.replace(":", "_") + str(seconds) + "_" + str(i) + ".jpg")
                cv2.imwrite(filename, face)

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                             (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # draw the bounding box of the face along with the
            # associated probability
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            t = str(datetime.timedelta(seconds=seconds))
            print('{} - {}'.format(t, text))

    # update the FPS counter
    # fps.update()

    # show the output frame
    if verbose >= 2:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break


# stop the timer and display FPS information
# fps.stop()
# print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
# vs.stop()
