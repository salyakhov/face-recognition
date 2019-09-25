The face recognition project inspired by Mr. Bbreitenbikher  (aka Brightonbukher, Vtbikher, Bitnerbukher, Blandermixer, Brachenbuckstner,
 Bredlicuper, Barbershopper, and other gazillion spellings) in Russian intellectual TV-show "Chto? Gde? Kogda?" (What? When? Where?).

* https://www.instagram.com/chto_gde_kogda/
* https://www.youtube.com/watch?v=cvxChbkMFSY

# Option 1 - fast but accuracy is low
Base on tutorials from https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/

## Prepare dataset
Align an original photo for recognition:
```
$ python face-alignment/align_faces.py -p face-alignment/shape_predictor_68_face_landmarks.dat -i <some image from internet> -o dataset/bykova
```
Simple bash script to process multiple images:
```
$ for x in `ls dataset/tmp/face_*.jpg`; do echo $x; python face-alignment/align_faces.py -p face-alignment/shape_predictor_68_face_landmarks.dat -i $x -o dataset/tmp/; done
```

Get youtube video (let's use best quality for face extraction for further learning):
```bash
youtube-dl -f 22 https://www.youtube.com/watch?v=cvxChbkMFSY -o input/final_20190630.mp4
```

## Train model
```bash
rm -rf output/
workon ocv
python opencv-face-recognition/extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle --detector opencv-face-recognition/face_detection_model --embedding-model opencv-face-recognition/openface_nn4.small2.v1.t7
python opencv-face-recognition/train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle
```


# Option 2 - slow (need GPU) but accuracy is good
Based on tutorial from https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/

## Prepare dataset
```
python encode_faces.py --dataset dataset --encodings output/dlib_encodings_brighton.pickle
```
Test recognition
```bash
python recognize_faces_video_file.py --encodings encodings-chgk.pickle --input input/final_20190630.mp4
```