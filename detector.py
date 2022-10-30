import argparse
import numpy as np
import cv2 as cv
import time

class User:
    def __init__(self, id, name, place, time):
        self.place = place
        self.name = name
        self.time = time
        self.id = id
        self.face_feature = None        


class Detector:
    def __init__(self, face_detection_model, score_threshold, nms_threshold, top_k, face_recognition_model, deviceId, scale, place):
        self.detector = cv.FaceDetectorYN.create(
            face_detection_model,
            "",
            (320, 320),
            score_threshold,
            nms_threshold,
            top_k
        )
        
        self.recognizer = cv.FaceRecognizerSF.create(
            face_recognition_model,
            ""
        )
        
        self.deviceId = deviceId
        self.scale = scale
        self.place = place
        self.users = dict()
    
    def addUser(self, name, imagePath):
        id = len(self.users)
        user = User(id, name, self.place, None)
        
        frame = cv.imread(cv.samples.findFile(imagePath))
        
        frameWidth = int(frame.shape[1]*self.scale)
        frameHeight = int(frame.shape[0]*self.scale)
        frame = cv.resize(frame, (frameWidth, frameHeight))
        
        self.detector.setInputSize((frameWidth, frameHeight))
        
        faces = self.detector.detect(frame)
        print(faces)
        face = faces[1][0]
        face_align = self.recognizer.alignCrop(frame, face)
        face_feature = self.recognizer.feature(face_align)
        
        user.face_feature = face_feature
        
        
        
        self.users[id] = user
    
    def detectUser(self, face, frame):
        face_align = self.recognizer.alignCrop(frame, face)
        face_feature = self.recognizer.feature(face_align)
        cosine_similarity_threshold = 0.363        
        
        for key, user in self.users.items():
            cosine_score = self.recognizer.match(face_feature, user.face_feature, cv.FaceRecognizerSF_FR_COSINE)
            
            if cosine_score >= cosine_similarity_threshold:
                return key, face_feature
            
        return None, face_feature
    
    def visualize(self, input, faces, classes, thickness=2):
        if faces[1] is not None:
            for idx, face in enumerate(faces[1]):
                user = self.users[classes[idx]]
                
                print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f} color {}'.format(idx, face[0], face[1], face[2], face[3], face[-1], classToColor[classes[idx]]))
                coords = face[:-1].astype(np.int32)
                cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), classToColor[user.id], thickness)
                cv.putText(image, user.name, (coords[0], coords[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, classToColor[user.id], 2)
                
                cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
                cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
                cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
                cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
                cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
        cv.putText(input, 'FPS: {:.2f}'.format(12), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    def detection(self):        
        cap = cv.VideoCapture(self.deviceId)
        frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) * self.scale)
        frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) * self.scale)
        self.detector.setInputSize([frameWidth, frameHeight])

        while cv.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break
            frame = cv.resize(frame, (frameWidth, frameHeight))
            faces = self.detector.detect(frame)
            colors = []
            
            for face in faces[1]:
                print(face)
                
                key, face_feature = detectClass(face, frame)
                
                msg = 'the same identity'
                if key is None or key not in self.users:
                    msg = 'different identities'
                    key = len(self.users)
                    newUser = User(key, "User {}".format(key), self.place, time.time())
                    
                    
                    self.users[key] = newUser
                else:
                    self.users[key].place = self.place
                    self.users[key].time = time.time()
                    
                colors.append(key)

            self.visualize(frame, faces, colors)
            cv.imshow('Live', frame)
                    
        cv.destroyAllWindows()
        
    def lastSeen(self):
        return self.users
