import cv2
import os

video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Unable to access the camera")
else:
    print("Access to the camera was successfully obtained")

facedetect = cv2.CascadeClassifier()
count=0

nameID=str(input("Enter your name")).lower()
path = 'images/' + nameID

if not os.path.exists(path):
    os.makedirs(path)

while True:
    ret, frame = video.read()
    faces = facedetect.detectMultiScale(frame, 1.3, 5)
    for x, y, w, h in faces:
        count = count + 1
        name = 'images/' + nameID + '/' + str(count) + '.jpg'
        cv2.imwrite(name, frame[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.imshow("WindowFrame", frame)
    cv2.waitKey(1)
    if count > 500:
        break
video.release()
cv2.destroyAllWindows()