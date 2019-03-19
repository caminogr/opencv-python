# https://blanktar.jp/blog/2015/02/python-opencv-realtime-lauhgingman.html

import cv2

laugh = cv2.imread('./waraiotoko.png', -1)
mask = cv2.cvtColor(laugh[:,:,3], cv2.COLOR_GRAY2BGR)/255.0
# 笑い男からαチャンネルだけを抜き出して0から1までの値にする。あと3チャンネルにしておく。
laugh = laugh[:,:,:3]  

# print('overlayImage', overlayImage.shape)
# originalOverlayImage = cv2.imread('./waraiotoko.png', cv2.IMREAD_UNCHANGED)
# print(cv2.IMREAD_UNCHANGED)
# mask = originalOverlayImage[:,:,3]
# mask = cv2.cvtColor(mask, cv2.cv.CV_GRAY2BGR) 
# mask = mask / 255.0
# overlayImage = originalOverlayImage[:,:,:3]
# overlayImage = cv2.addWeighted(originalOverlayImage, 1, originalOverlayImage, 0.8, 0)
# print(dst.shape)
# resizedOverlayImage = cv2.resize(overlayImage, (400, 400))


# def mosaic(img, alpha):
#     w = img.shape[1]
#     h = img.shape[0]

#     img = cv2.resize(img, (int(w*alpha), int(h*alpha)))
#     img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

#     return img


face_cascade_path = '/usr/local/opt/opencv/share/'\
                    'opencv4/haarcascades/haarcascade_frontalface_default.xml'
# eye_cascade_path = '/usr/local/opt/opencv/share/'\ 'opencv4/haarcascades/haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier(face_cascade_path)
# eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

cap = cv2.VideoCapture(0)
# src = cv2.imread('data/src/lena_square.png')





# わらいおとこの色を変えたい




while True:
    ret, img = cap.read()
    # print('img', img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # imgWithAlpah = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    # print('rgba', rgba.shape)

    face = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5)
    # print(face)
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

    for rect in face:
        # for (x, y, w, h) in face:
        rect[0] -= min(60, rect[0])
        rect[1] -= min(90, rect[1])
        rect[2] += min(180, img.shape[1]-(rect[0]+rect[2]))
        rect[3] += min(180, img.shape[0]-(rect[1]+rect[3]))

        # print("x:", x)
        # print("y:", y)
        # print("w:", w)
        # print("h:", h)
        # img[y:y+h, x:x+w] = mosaic(img[y:y+h, x:x+w], 0.05)

        # back_im = overlayImage.copy()
        # back_im.paste(overlayImage, (x, y), overlayImage.split()[3])
        # overlayImage.cv2.imshow('video image', img)

    # 処理が重くならないように工夫
    # https://pycarnival.com/opencv_pil_resize_sample/

        # resizedOverlayImage = cv2.resize(overlayImage, tuple()

        # print("tuple(rect[2:])", tuple(rect[2:]))
        laugh2 = cv2.resize(laugh, tuple(rect[2:]))
        mask2 = cv2.resize(mask, tuple(rect[2:]))
        print('rect', rect)
        print('laugh2', laugh2)
        print('img', img)
        # mask2 = cv2.resize(mask, tuple(rect[2:]))

        img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = laugh2[:, :] * mask2 + img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] * (1.0 - mask2)
        # * \
        #     img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]



        # mask2 + img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] * (1.0 - mask2)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    # cv2.imshow('overlayImage', overlayImage)
    key = cv2.waitKey(10)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
