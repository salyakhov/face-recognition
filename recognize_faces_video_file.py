# USAGE
# python recognize_faces_video_file.py --encodings output/dlib_encodings_brighton.pickle --input input/final_20190630_sample_01.mp4 -y 0 -d hog -s 25 -o output/out.avi

# import the necessary packages
import face_recognition
import argparse
import imutils
import pickle
import time
import datetime
import cv2
import os
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
ap.add_argument("-v", "--verbose", required=False, help="show debug info", action="count", default=0)
ap.add_argument("-i", "--input", required=True, help="path to input video")
ap.add_argument("-o", "--output", type=str, help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1, help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="hog", help="face detection model to use: either `hog`, `cnn`, or `dnn`")
ap.add_argument("-m", "--detection-path", required=False, type=str, help="path to OpenCV's deep learning face detector")
ap.add_argument("-s", "--step", type=int, default=-1, help="step to read video. read every second of video by default. use 1 to read every frame")

args = vars(ap.parse_args())
verbose = args["verbose"]


def load_detector(path):
    print("[INFO] loading face detector...")
    proto_path = os.path.sep.join([path, "deploy.prototxt"])
    model_path = os.path.sep.join([path, "res10_300x300_ssd_iter_140000.caffemodel"])
    return cv2.dnn.readNetFromCaffe(proto_path, model_path)


def detect_faces(frame_):
    """
    TODO need to fix box coordinates
    :param frame_:
    :return: coordinates of detected faces
    """
    _boxes = []
    (h, w) = frame_.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(image=cv2.resize(frame_, (300, 300)),
                                      scalefactor=1.0,
                                      size=(300, 300),
                                      mean=(104.0, 177.0, 123.0),
                                      swapRB=False,
                                      crop=False)
    detector.setInput(imageBlob)
    detections = detector.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, endY, startY, endX) = box.astype("int")
            # (right, bottom, left, top) = css[3], css[0], css[1], css[2] -> (top, right, bottom, left)
            _boxes.append((startY, endY, startX, endX))
            # # extract the face ROI
            # face = frame[startY:endY, startX:endX]
            # (fH, fW) = face.shape[:2]

    return _boxes


detection_method = args["detection_method"]
detector = load_detector(args["detection_path"]) if detection_method == "dnn" else None

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# initialize the pointer to the video file and the video writer
print("[INFO] processing video...")
# stream = cv2.VideoCapture(args["input"])

UNKNOWN = "Unknown"

writer = None
ns_compare = 0
ns_matches = 0
ns_detects = 0
ns_encodes = 0
fm_cnt = 0

frame_id = 0
to_skip = 0

cap = cv2.VideoCapture(args["input"])
frame_count = cap.get(7)
frame_rate = cap.get(5)  # frame rate
duration = frame_count / frame_rate
print("Frames: {}, Frame rate: {}".format(frame_count, frame_rate))
# minutes = int(duration/60)
# seconds = duration%60
# print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))
print("Duration: {}".format(str(datetime.timedelta(seconds=duration))))

step = args["step"] if args["step"] > 0 else frame_rate


# loop over frames from the video file stream
while cap.isOpened():
    t0 = time.time()
    # grab the next frame
    frame_id += step + to_skip
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    sec = int(frame_id / frame_rate)
    t = str(datetime.timedelta(seconds=sec))

    (grabbed, frame) = cap.read()
    fm_cnt += 1

    # if the frame was not grabbed, then we have reached the
    # end of the stream
    if not grabbed:
        break

    # convert the input frame from BGR to RGB then resize it to have
    # a width of 750px (to speedup processing)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=750)
    r = frame.shape[1] / float(rgb.shape[1])

    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input frame, then compute
    # the facial embeddings for each face
    t1 = time.time()
    boxes = []
    if detection_method == "dnn":
        # TODO compare dnn against dlib
        boxes = detect_faces()

    else:
        boxes = face_recognition.face_locations(rgb, model=detection_method)

    #print("boxes={}".format(boxes))
    ns_detects = time.time() - t1
    t1 = time.time()
    encodings = face_recognition.face_encodings(rgb, boxes)
    ns_encodes = time.time() - t1
    names = []

    ns_compare = 0
    ns_matches = 0
    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        t1 = time.time()
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        ns_compare += (time.time() - t1)
        name = UNKNOWN

        # check to see if we have found a match
        t1 = time.time()
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            name = max(counts, key=counts.get)

        ns_matches += (time.time() - t1)
        # update the list of names
        names.append(name)
        if name != UNKNOWN:
            print("{} - {}".format(t, name))


    def format_float(value):
        return "{0:.5f}".format(value)


    if verbose >= 1:
        print("frame={}, t={}, t0={}, ns_detects={}, ns_encodes={}, ns_compare={}, ns_matches={}, "
              .format(fm_cnt, t, format_float(time.time() - t0), format_float(ns_detects), format_float(ns_encodes),
                      format_float(ns_compare), format_float(ns_matches)))

    # loop over the recognized faces
    if args["output"] is not None:
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # rescale the face coordinates
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)

            # draw the predicted face name on the image
            cv2.rectangle(frame, (left, top), (right, bottom),
                          (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)

    # if the video writer is None *AND* we are supposed to write
    # the output video to disk initialize the writer
    if writer is None and args["output"] is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 24,
                                 (frame.shape[1], frame.shape[0]), True)

    # if the writer is not None, write the frame with recognized
    # faces t odisk
    if writer is not None:
        writer.write(frame)

    # check to see if we are supposed to display the output frame to
    # the screen
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# close the video file pointers
cap.release()

# check to see if the video writer point needs to be released
if writer is not None:
    writer.release()
