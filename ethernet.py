import cv2 as cv

# cap = cv.VideoCapture(0)
# cap = cv.VideoCapture('rtsp://admin:adminadmin@10.164.2.174/H264?ch=1&subtype=0')
# cap = cv.VideoCapture('http:///10.164.2.174:554/video')
cap = cv.VideoCapture('rtsp://10.164.2.174/live1.sdp')
# cap = cv.VideoCapture('http:///10.164.2.174:554/video1.mjpg')
# cap = cv.VideoCapture('rtsp://live1.sdp:554')
# cap = cv.VideoCapture('rtsp://admin:adminadmin@10.164.2.174/1')

if not cap.isOpened():
    print("Unable to access the camera")
else:
    print("Access to the camera was successfully obtained")
    
while cv.waitKey(1) < 0:
    ret, frame = cap.read()
    cv.imshow("WindowFrame", frame)
cap.release()
cv.destroyAllWindows()