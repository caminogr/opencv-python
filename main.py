import cv2
import numpy as np


def mosaic(img, alpha):
    w = img.shape[1]
    h = img.shape[0]

    img = cv2.resize(img, (int(w*alpha), int(h*alpha)))
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

    return img


face_cascade_path = '/usr/local/opt/opencv/share/'\
                    'opencv4/haarcascades/haarcascade_frontalface_default.xml'
eye_cascade_path = '/usr/local/opt/opencv/share/'\
                   'opencv4/haarcascades/haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

cap = cv2.VideoCapture(0)
# src = cv2.imread('data/src/lena_square.png')

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5)

    # ratio = 0.05
    # for x, y, w, h in faces:
    #     small = cv2.resize(src[y: y + h, x: x + w], None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    #     src[y: y + h, x: x + w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    # cv2.imwrite('data/dst/opencv_face_detect_mosaic.jpg', src)

    # for x, y, w, h in faces:
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #     face = img[y: y + h, x: x + w]
    #     face_gray = gray[y: y + h, x: x + w]
    #     eyes = eye_cascade.detectMultiScale(face_gray)
    #     for (ex, ey, ew, eh) in eyes:
    #         cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    for (x, y, w, h) in face:

        img[y:y+h, x:x+w] = mosaic(img[y:y+h, x:x+w], 0.05)
    cv2.imshow('video image', img)
    key = cv2.waitKey(10)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
