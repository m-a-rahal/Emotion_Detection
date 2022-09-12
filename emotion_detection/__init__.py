import PIL.Image
import numpy as np
from PIL import Image

from emotion_detection.video_capture import VideCapture
from emotion_detection.image_dict import ImageDict, load_image
from emotion_detection.face_localization import face_boxes_detection
from emotion_detection.image_dict import ImageDict
from emotion_detection.image_dict_list import ImageDictList
from emotion_detection.utils.cropping import make_square_images
from emotion_detection.utils.drawing import Drawer
from emotion_detection.utils.file_manager import create_path
from emotion_detection.models import batch_from_images, load_default_model
from emotion_detection.video_capture import VideCapture, FIRST_CAMERA, SECOND_CAMERA

# face detection methods ===========================================================
HARR_CASCADE = 0
MTCNN = 1


class ImageEmotionDetector:
    def __init__(self, model=None, face_detection_method=0, min_confidence=0.9, verbose=0):
        """
        detects faces from an image, and for each face, it detects the emotions of the face
        NOTES:
        - the model needs to have two extra parameters:
            - preprocessing : preprocessing layers or function that takes a batch of images as input
            - emotions : emotion names
        :param model: the model that predicts emotions, leave None to use default model
        :param face_detection_method: two options, 0 for HARR_CASCADE (default) and 1 for MTCNN
        :param min_confidence: min confidence for face localization (to exclude low confidence boxes)
        :param verbose: show model prediction progress bar : model.predict(verbose = verbose)
        """
        self.min_confidence = min_confidence
        if model is None:
            model = load_default_model()  # use default model
        self.model = model
        self.verbose = verbose
        # if the face detection method is callable, use it
        self.face_detection_method = face_detection_method
        if callable(face_detection_method):
            self.box_localization = lambda image, min_conf: face_detection_method(image, min_conf)
        else:
            self.box_localization = lambda image, min_conf: face_boxes_detection(image, self.face_detection_method,
                                                                                 min_conf)

    def detect(self, image: Image) -> ImageDict:
        """
        detect emotions from image,
        NOTE: images and boxes returned are numpy images
        :param image: an PIL or numpy (RGB) image or an image file (use ImageEmotionDetector.load_image to read images)
        :return: a dictionary with results (boxes, predictions ... etc.), or an empty dictionary if nothing is detected
        """
        # 1. assert that the model has right needed components =================================================
        assert hasattr(self.model, 'preprocessing'), "the model needs to have the parameter 'preprocessing' (" \
                                                     "preprocessing layers or function that takes a batch of images " \
                                                     "as input), \nplease set model.preprocessing = your_function "
        assert hasattr(self.model, 'emotions'), "the model needs to have the parameter 'emotions' (emotion names) " \
                                                "\nplease set model.emotions = [emotion names here]"
        # 2. detect face positions ==============================================================================
        if isinstance(image, str):
            image = load_image(image)
        if isinstance(image, np.ndarray):
            np_image = image
            image = Image.fromarray(image)
        elif isinstance(image, PIL.Image.Image):
            np_image = np.array(image)
        else:
            raise AttributeError('unsupported type of image, please use PIL image, or numpy image with RGB colors ('
                                 'not BGR!)')
        boxes = self.box_localization(np_image, self.min_confidence)
        # 3. make square images from boxes ======================================================================
        boxes, square_images = make_square_images(image, boxes)
        # if no good squares are present, return an empty dictionary
        if len(boxes) == 0:
            return ImageDict()
        # 4. make batch of images and pass them to the model, this will use the input shape of the model
        batch = batch_from_images(square_images, target_size=self.model.input.shape[1:3])
        batch = self.model.preprocessing(batch)
        predictions = self.model.predict(batch, verbose=self.verbose)
        # 6. return the results ==================================================================================
        result = ImageDict(image=np_image, predictions=predictions, boxes=boxes, emotions=self.model.emotions)
        return result

    __call__ = detect


class VideoEmotionDetector:
    def __init__(self, image_emotion_detector: ImageEmotionDetector = None,
                 logging=False, save_file=None, drawer: Drawer = None):
        """
        :param image_emotion_detector:
        :param logging:
        :param save_file:
        :param drawer:
        """
        if drawer is None:
            drawer = Drawer(image_emotion_detector.model.emotions)  # default drawer
        if image_emotion_detector is None:
            image_emotion_detector = ImageEmotionDetector()
        self.drawer = drawer
        self.save_file = save_file
        self.logging = logging
        self.emotion_image_detector = image_emotion_detector

    def detect(self, video: VideCapture = None):
        """
        :param video: ed.VideoCapture object (ed is the emotion_detection module)
        :return: log of all detection results, or nothing if logging = False
        """
        if video is None:
            video = VideCapture(0)
        with video as capture:
            log = ImageDictList(emotions=self.emotion_image_detector.model.emotions, drawer=self.drawer)
            while capture.cap.isOpened():
                # read frame, and stop if no frames are available
                try:
                    image = capture.read()
                except VideCapture.LastFrameException:
                    break
                # predict labels
                results = self.emotion_image_detector(image)
                # if no faces have been detected, continue
                if not results:
                    # show image
                    if not capture.show(image):
                        break
                    continue
                # get boxes and predictions
                boxes = results.boxes
                predictions = results.predictions
                results.drawer = self.drawer  # add same drawer to the result
                # draw boxes and margin
                np_image = self.drawer.draw_boxes_and_margin(image, predictions, boxes)
                # append results to log
                log.append(results)
                # show image
                if not capture.show(np_image):
                    break

        return log

    __call__ = detect
