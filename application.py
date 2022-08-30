from face_detection.face_detection import face_detection_from_video

# models
FER2013 = 10  # 69.93% accuracy on FER2013, terrible in real world applications
FER_PLUS = 11  # 83.--% accuracy on FER+, good overall not with some emotions like disguist
MOBILE_NET = 12  # 64.--% accuracy on AffectNet, very small and very precise, good for real world application
# face detection methods
HARR_CASCADE = 0  # fast but inaccurate
MTCNN = 1  # slow but accurate


# ==================================================================================================
# Model and parameters choice
# ==================================================================================================
def main():
    model = load_model(MOBILE_NET)
    face_detection_from_video(model=model, technique=HARR_CASCADE, show_boxes=False, resolution=(460, 259))


# ==================================================================================================
# Code (don't modify)
# ==================================================================================================

def load_model(model):
    if model == MOBILE_NET:
        return mobilenet('D:/Mohamed/PFE_Application/models/mobilenet/mobilenet_7.h5')
    elif model == FER_PLUS:
        return fer2013_to_ferplusCSV()
    elif model == FER2013:
        return fer2013()


if __name__ == '__main__':
    main()
