import argparse
import numpy as np
import cv2 as cv

parser = argparse.ArgumentParser()
parser.add_argument('--video', '-v', type=str, help='Path to the input video.')
parser.add_argument('--scale', '-sc', type=float, default=1.0, help='Scale factor used to resize input video frames.')
parser.add_argument('--face_detection_model', '-fd', type=str, default='dependencies/face_detection_yunet_2022mar.onnx', help='Path to the face detection model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet')
parser.add_argument('--face_recognition_model', '-fr', type=str, default='dependencies/face_recognition_sface_2021dec.onnx', help='Path to the face recognition model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface')
parser.add_argument('--score_threshold', type=float, default=0.9, help='Filtering out faces of score < score_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
args = parser.parse_args()

classToColor = {0: (255, 0, 0),
                1: (0, 255, 0),
                2: (0, 0, 255),
                3: (0, 255, 255),
                4: (255, 0, 255),
                5: (255, 255, 0)}
print(classToColor)

def visualize(input, faces, classes, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f} color {}'.format(idx, face[0], face[1], face[2], face[3], face[-1], classToColor[classes[idx]]))
            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), classToColor[classes[idx]], thickness)
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv.putText(input, 'FPS: {:.2f}'.format(12), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def detectClass(face, frame, classes):
    face1_align = recognizer.alignCrop(frame, face)
    face1_feature = recognizer.feature(face1_align)
    cosine_similarity_threshold = 0.363
    
    for key, face2_feature in classes.items():
        cosine_score = recognizer.match(face1_feature, face2_feature, cv.FaceRecognizerSF_FR_COSINE)
        
        if cosine_score >= cosine_similarity_threshold:
            return key, face1_feature
        
    return None, face1_feature

detector = cv.FaceDetectorYN.create(
    args.face_detection_model,
    "",
    (320, 320),
    args.score_threshold,
    args.nms_threshold,
    args.top_k
)

recognizer = cv.FaceRecognizerSF.create(
    args.face_recognition_model,
    ""
)
        
if args.video is not None:
    deviceId = args.video
else:
    deviceId = 0
    
cap = cv.VideoCapture(deviceId)
frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)*args.scale)
frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)*args.scale)
detector.setInputSize([frameWidth, frameHeight])

classes = dict()

while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        print('No frames grabbed!')
        break
    frame = cv.resize(frame, (frameWidth, frameHeight))
    faces = detector.detect(frame)
    colors = []
    
    for face in faces[1]:
        print(face)
        
        key, face_feature = detectClass(face, frame, classes)
        
        msg = 'the same identity'
        if key is None:
            msg = 'different identities'
            key = len(classes)
            classes[key] = face_feature
            
        colors.append(key)

    visualize(frame, faces, colors)
    cv.imshow('Live', frame)
            
cv.destroyAllWindows()