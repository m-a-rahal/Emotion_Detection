#==================================================================================================
#=== Face detection — mtcnn network ===============================================================
# source https://github.com/ipazc/mtcnn ===========================================================
#==================================================================================================
from pathlib import Path
from mtcnn.mtcnn import MTCNN
import cv2

def detect_faces(image_or_path):
    if isinstance(image_or_path, str) or isinstance(image_or_path, Path):
        image = cv2.imread(image_or_path)
    else:
        image = image_or_path
    detector = MTCNN()
    return detector.detect_faces(image)
''' return example
[
    {
        'box': [277, 90, 48, 63],
        'keypoints':
        {
            'nose': (303, 131),
            'mouth_right': (313, 141),
            'right_eye': (314, 114),
            'left_eye': (291, 117),
            'mouth_left': (296, 143)
        },
        'confidence': 0.99851983785629272
    },
        {
        'box': [120, 6, 52, 4],
        'keypoints':
        {
            'nose': (303, 131),
            'mouth_right': (313, 141),
            'right_eye': (314, 114),
            'left_eye': (291, 117),
            'mouth_left': (296, 143)
        },
        'confidence': 0.98851983785629272
    }
]
'''