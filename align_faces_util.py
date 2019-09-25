# USAGE
# python align_faces.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg

# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import time


class AlignFace:
	def __init__(self, predictor):
		# initialize dlib's face detector (HOG-based) and then create
		# the facial landmark predictor and the face aligner
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor(predictor)
		self.fa = FaceAligner(self.predictor, desiredFaceWidth=256)

	def align(self, image):
		# load the input image, resize it, and convert it to grayscale
		#image = imutils.resize(image, width=800)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		#cv2.imshow("gray", gray)
		#cv2.waitKey(0)

		rects = self.detector(gray, 2)

		res = []
		# loop over the face detections
		for rect in rects:
			# extract the ROI of the *original* face, then align the face
			# using facial landmarks
			#(x, y, w, h) = rect_to_bb(rect)
			#face_orig = imutils.resize(image[y:y + h, x:x + w], width=256)
			#cv2.imshow("Original", face_orig)
			#cv2.waitKey(0)
			face_aligned = self.fa.align(image, gray, rect)
			res.append(face_aligned)

			#import uuid
			#f = str(uuid.uuid4())
			#cv2.imwrite("foo/" + f + ".png", faceAligned)

			# display the output images
			#cv2.imshow("Original", faceOrig)
			#cv2.imshow("Aligned", face_aligned)
			#cv2.waitKey(0)

		return res


def main():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
	ap.add_argument("-i", "--image", required=True, help="path to input image")
	ap.add_argument("-o", "--output", required=False, help="path to output")
	args = vars(ap.parse_args())

	image = cv2.imread(args["image"])

	aligned_faces = AlignFace(args["shape_predictor"]).align(image)

	out = args["output"]
	if out:
		cnt = 0
		for face in aligned_faces:
			filename = out + "/image_" + str(int(time.time())) + "_" + cnt + ".jpg"
			cv2.imwrite(filename, face)
			print("Aligned image was creates at", filename)
			cnt += 1


if __name__ == '__main__':
	main()
