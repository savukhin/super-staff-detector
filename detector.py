import argparse
import numpy as np
import cv2 as cv
import datetime
import sys
import os
import shutil

class User:
    def __init__(self, id, name, place, time, imagePathes):
        self.place = place
        self.name = name
        self.time = time
        self.id = id
        self.face_features = None
        self.imagePathes = imagePathes


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
        self.lastUserID = 1
    
    def addUser(self, name, imagePathes):
        id = len(self.users)
        user = User(id, name, self.place, None, imagePathes)
        face_features = []
        for path in imagePathes:
            print(path)
            f = open(path, "rb")
            chunk = f.read()
            chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
            frame = cv.imdecode(chunk_arr, cv.IMREAD_COLOR)
            # frame = cv.imread(cv.samples.findFile(path))
            
            frameWidth = int(frame.shape[1]*self.scale)
            frameHeight = int(frame.shape[0]*self.scale)
            frame = cv.resize(frame, (frameWidth, frameHeight))
            
            self.detector.setInputSize((frameWidth, frameHeight))
            
            faces = self.detector.detect(frame)
            face = faces[1][0]
            face_align = self.recognizer.alignCrop(frame, face)
            face_feature = self.recognizer.feature(face_align)
            
            face_features.append(face_feature)
            
        user.face_features = face_features
        
        self.users[id] = user
        self.lastUserID += 1
    
    # 0 - No user with this ID, 1 - Name not changed, (str, str) - Sucess replaced, 2 - Replaced only name
    def changeUser(self, oldId, newName):
        oldId = int(oldId)
        if oldId not in self.users:
            raise "No user with this ID"
        
        oldUser = self.users[oldId]
        
        if oldUser.name == newName:
            return 1
        
        for key, user in self.users.items():
            if user.name == newName:
                user.face_features += oldUser.face_features
                newFolder = os.path.dirname(os.path.abspath(user.imagePathes[0]))
                
                for path in oldUser.imagePathes:
                    oldFileName = os.path.basename(path)
                    newPath = newFolder + "/" + oldUser.name + "-" + oldFileName
                    
                    shutil.copy(path, newPath)
                    user.imagePathes.append(newPath)
                    
                # user.imagePathes += oldUser.imagePathes
                del self.users[oldId]
                return (oldUser.name, user.name)
        
        
        oldFolder = os.path.dirname(os.path.abspath(oldUser.imagePathes[0]))
        
        newFolder = os.path.join(
            os.path.abspath(os.path.join(oldFolder, os.pardir)),
            newName
        )
        
        os.rename(oldFolder, newFolder)
        
        oldPathes = self.users[oldId].imagePathes
        self.users[oldId].imagePathes = []
        
        
        for path in oldPathes:
            oldFileName = os.path.basename(path)
            
            self.users[oldId].imagePathes.append(
                os.path.join(newFolder, oldFileName)
            )
            
        self.users[oldId].name = newName
        return 2
    
    def detectUser(self, face, frame):
        face_align = self.recognizer.alignCrop(frame, face)
        face_feature = self.recognizer.feature(face_align)
        cosine_similarity_threshold = 0.363       
         
        for key, user in self.users.items():
            for face2_feature in user.face_features:
                cosine_score = self.recognizer.match(face_feature, face2_feature, cv.FaceRecognizerSF_FR_COSINE)
            
                if cosine_score >= cosine_similarity_threshold:
                    return key, face_feature
            
        return None, face_feature
    
    def visualize(self, input, faces, classes, thickness=2):
        classToColor = {0: (255, 0, 0),
                1: (0, 255, 0),
                2: (0, 0, 255),
                3: (0, 255, 255),
                4: (255, 0, 255),
                5: (255, 255, 0)}

        if faces[1] is not None:
            for idx, face in enumerate(faces[1]):
                user = self.users[classes[idx]]
                
                # print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f} color {}'.format(idx, face[0], face[1], face[2], face[3], face[-1], classToColor[user.id % 6]))
                coords = face[:-1].astype(np.int32)
                cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), classToColor[user.id % 6], thickness)
                cv.putText(input, user.name, (coords[0], coords[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, classToColor[user.id % 6], 2)
                
                cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
                cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
                cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
                cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
                cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
        cv.putText(input, 'FPS: {:.2f}'.format(12), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def detection(self):
        print ("Getting cam {}".format(self.deviceId))
        sys.stdout.flush()
        
        cap = cv.VideoCapture(self.deviceId)
        frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) * self.scale)
        frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) * self.scale)
        self.detector.setInputSize([frameWidth, frameHeight])
        
        print ("Starting detection", flush=True)
        sys.stdout.flush()

        while True:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print('No frames grabbed!', flush=True)
                break
            
            frame = cv.resize(frame, (frameWidth, frameHeight))
            faces = self.detector.detect(frame)
            colors = []
            
            if (faces[1] is None):
                # cv.imshow('Live', frame)
                continue
            
            for face in faces[1]:
                key, face_feature = self.detectUser(face, frame)
                
                msg = 'the same identity'
                if key is None or key not in self.users:
                    msg = 'different identities'
                    key = self.lastUserID
                    newName = "User {}".format(key)
                    
                    path = "images/" + newName
                    if not os.path.exists(path):
                        os.mkdir(path)
                    cv.imwrite(path + "/initial.jpg", frame)
                    
                    newUser = User(key, newName, self.place, datetime.datetime.now(), [path + "/initial.jpg"])
                    newUser.face_features = [face_feature]
                    
                    self.users[key] = newUser
                    self.lastUserID += 1
                else:
                    self.users[key].place = self.place
                    self.users[key].time = datetime.datetime.now()
                    # print ("username", self.users[key].name, "new place is", self.users[key].place, "new time is ", self.users[key].time, " new len is", len(self.users), "lastSeen len", len(await self.lastSeen()), flush=True)
                    
                colors.append(key)

            # self.visualize(frame, faces, colors)
            # cv.imshow('Live', frame)
                    
        # cv.destroyAllWindows()
        
    def lastSeen(self):
        return self.users
