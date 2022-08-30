#==================================================================================================
#=== Face detection — using CV2 harr cascade ======================================================
# source https://github.com/shantnu/PyEng ===========================================================
#==================================================================================================
#!/usr/bin/python
import sys
import cv2
from pathlib import Path

def face_detect(img, show_img=False, draw_rect=False, cascasdepath = "haarcascade_frontalface_default.xml",
        scaleFactor = 1.19,
        minNeighbors = 5,
        minSize = (30,30)):

    if isinstance(img, str) or isinstance(img, Path):
        image = cv2.imread(img)
    else:
        image = img
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascasdepath)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = scaleFactor,
        minNeighbors = minNeighbors,
        minSize = minSize
        )

    print("The number of faces found = ", len(faces))

    if draw_rect:
        for (x,y,w,h) in faces:
            cv2.rectangle(image, (x,y), (x+h, y+h), (0, 255, 0), 2)

    if show_img:
        cv2.imshow("Faces found", image)
        cv2.waitKey(0)
    return faces, image

if __name__ == "__main__":
    face_detect('abba.png', show_img=True)
    face_detect('little_mix.jpg', show_img=True)
    #face_detect(sys.argv[1])