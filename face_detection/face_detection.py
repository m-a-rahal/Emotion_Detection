# see requirements.txt for installation
import cv2
import numpy as np
from face_detection.face_detection_methods.mtcnn import detect_faces as mtcnn_detect_faces
from face_detection.face_detection_methods.haarcascade import face_detect as harrcascade_detect_faces
from face_detection.utils.video_capture import VideCapture
from face_detection.utils.drawing import create_probabilites_text_image, add_text_under_box
from face_detection.utils.other import turn_to_square, run_as_thread, pad_to_square
import tensorflow as tf
import threading
import traceback

# --- constants --------------------------------------------------------------------------------------------------
HARR_CASCADE = 0
MTCNN = 1


# --------------------------------------------------------------------------------------------------


def extract_boxes_from_image(img, technique):
    if technique == MTCNN:
        boxes = []
        faces = mtcnn_detect_faces(img)
        for face in faces:
            boxes.append(face['box'])
    elif technique == HARR_CASCADE:
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        boxes, _ = harrcascade_detect_faces(img)
    return boxes


def face_detection_from_video(technique=HARR_CASCADE, color=(0, 220, 0),
                              logging=False, show_boxes=False, model=None, sort_labels=False,
                              resolution=(1000, 1000)):
    """
    Record video from webcam and detect the face box.
    detection_type: two modes HARR_CASCADE and MTCNN, harr method is fast but inaccurate, mtcnn is slow but much more accurate
    logging: if True, returns all the box images captured and their relative data
    """
    # To capture video from webcam. 
    with VideCapture(*resolution) as capture:
        log = []
        while True:
            frame = capture.read()
            img = np.array(frame)
            # extract boxes from faces
            boxes = extract_boxes_from_image(img, technique)
            # Draw the rectangle around each face
            square_images = []
            labels = []

            # make box a square instead
            for i in range(len(boxes)):
                boxes[i] = turn_to_square(*boxes[i])

            for (x, y, w, h) in boxes:
                cropped = img[y:y + h, x:x + w]
                # cropped = pad_to_square(cropped)
                try:
                    cropped = cv2.resize(cropped, (200, 200))
                    square_images.append(cropped)
                    labels.append(model.predict_label(model, cropped))  # predict_lable returns : prob, name, probs
                except Exception as e:
                    print(traceback.format_exc(e))

            if show_boxes:
                if len(square_images) > 0:
                    cv2.imshow('cropped', np.hstack(square_images))

            for i, (x, y, w, h) in enumerate(boxes):
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                text = labels[i][1] + ' ' + f'{labels[i][0]:.2f}'
                add_text_under_box(text, img, (x, y, w, h), color=color)
                margin_image = create_probabilites_text_image(labels[i][2], img, model, sort_labels=sort_labels)
                img = np.concatenate((img, margin_image), axis=1)
            # if logging is on, save data
            if logging and len(boxes) > 0:
                log.append(dict(image=img, boxes=boxes, labels=labels))
            # Display
            capture.show(img)
            # Stop if escape key is pressed
            k = cv2.waitKey(1) & 0xff
            if k in map(ord, '0q'): break  # press these keys to stop

    cv2.destroyAllWindows()
    return log


if __name__ == '__main__':
    face_detection_from_video(technique=MTCNN)
