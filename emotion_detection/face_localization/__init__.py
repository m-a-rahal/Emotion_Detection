import numpy

from emotion_detection.face_localization.face_detection_methods.haarcascade import \
    face_detect as harrcascade_detect_faces
from emotion_detection.face_localization.face_detection_methods.mtcnn import detect_faces as mtcnn_detect_faces


def face_boxes_detection(image: numpy.ndarray, method):
    """
    :param image: a numpy image
    :param method: 0 for HARR-CASCADE, 1 for MTCNN
    :return:
    """
    boxes = []
    if method == 1:
        faces = mtcnn_detect_faces(image)
        for face in faces:
            boxes.append(face['box'])
    elif method == 0:
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        boxes, _ = harrcascade_detect_faces(image)
    else:
        raise AttributeError(
            f'{0} is not a valid method for face detection, please use 0 for HARR_CASCADE or 1 for MTCNN')
    return boxes
