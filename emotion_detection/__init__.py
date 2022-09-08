import time

import PIL.Image
from PIL import Image
import cv2
import numpy as np

from emotion_detection.face_localization import face_boxes_detection
from emotion_detection.models import load_default_model, get_max_emotion
from emotion_detection.utils.cropping import make_square_images
from emotion_detection.utils.drawing import add_text_under_box, create_probabilities_text_image, Drawer
from emotion_detection.utils.file_manager import create_path
from tensorflow import cast as tf_cast
from tensorflow import float32 as tf_float32
from tensorflow.keras.utils import load_img

# face detection methods ===========================================================
HARR_CASCADE = 0
MTCNN = 1
FIRST_CAMERA = 0
SECOND_CAMERA = 1


class ImageDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def draw_boxes(self, drawer: Drawer = None):
        if drawer is None:
            drawer = Drawer(self.emotions)
        return drawer.draw_boxes(self.image, self.predictions, self.boxes)

    def show_image(self):
        return Image.fromarray(self.image)

    def show_image_with_boxes(self, drawer: Drawer = None):
        return Image.fromarray(self.draw_boxes(drawer))

    def show_boxes(self, drawer: Drawer = None):
        if drawer is None:
            drawer = Drawer(self.emotions)
        return drawer.draw_boxes(self.image, self.predictions, self.boxes)

    def show_square_images(self, drawer: Drawer = None):
        if drawer is None:
            drawer = Drawer(self.emotions)
        return drawer.show_square_images(self.square_images, self.predictions)

    def show_dataframe(self, drawer: Drawer = None):
        if drawer is None:
            drawer = Drawer(self.emotions)
        return drawer.show_as_dataframe(self.predictions)

    def draw_boxes_and_margin(self, drawer: Drawer = None, selected_idx=0):
        if drawer is None:
            drawer = Drawer(self.emotions)
        return drawer.draw_boxes_and_margin(self.image, self.predictions, self.boxes, selected_idx=selected_idx)

    def show_boxes_and_margin(self, drawer: Drawer = None, selected_idx=0):
        return Image.fromarray(self.draw_boxes_and_margin(drawer, selected_idx))


class ImageDictList(list):
    def __init__(self, *args, emotions=None, drawer: Drawer = None):
        super().__init__(*args)
        self.emotions = emotions
        self.drawer = drawer

    def scroll_through_results(self, rotate_scroll=True):
        """
        :param rotate_scroll: rewind list of results when the index is out of bounds
        """
        print('press 4 to show previous image')
        print('press 6 to show next image')
        print('press 8 and 2 to scroll through faces')
        print('press 0 to exit')
        i = -1
        j = -1
        action = ord('r')
        while True:
            if action == ord('2'):
                j -= 1
            if action == ord('8'):
                j += 1
            if action == ord('6'):
                i += 1
            elif action == ord('4'):
                i -= 1
            if rotate_scroll:
                i = i % len(self)
            else:
                i = min(len(self) - 1, i)
                i = max(0, i)
            res: ImageDict = self[i]
            image = res.draw_boxes_and_margin(selected_idx=j % len(self[i].boxes))
            # res.show_square_images()
            cv2.imshow('image', image[:, :, ::-1])
            action = cv2.waitKey(0)
            if action in map(ord, '0q'):
                break
        cv2.destroyAllWindows()


def batch_from_images(square_images, target_size):
    resized = [cv2.resize(square, target_size) for square in square_images]
    nd_array = np.array(resized)
    return tf_cast(nd_array, tf_float32)


class ImageEmotionDetector:
    def __init__(self, model=None, face_detection_method=0, min_confidence=0.9,
                 return_image=True, return_square_images=True,
                 verbose=0):
        """
        detects faces from an image, and for each face, it detects the emotions of the face
        NOTES:
        - the model needs to have two extra parameters:
            - preprocessing : preprocessing layers or function that takes a batch of images as input
            - emotions : emotion names
        :param model: the model that predicts emotions, leave None to use default model
        :param face_detection_method: two options, 0 for HARR_CASCADE (default) and 1 for MTCNN
        :param min_confidence: min confidence for face localization (to exclude low confidence boxes)
        :param return_image: Return the image with the bounding boxes drawn on it
        :param return_square_images: return the square images
        :param verbose: show model prediction progress bar : model.predict(verbose = verbose)
        """
        self.min_confidence = min_confidence
        if model is None:
            model = load_default_model()  # use default model
        self.model = model
        self.verbose = verbose
        self.return_square_images = return_square_images
        self.return_image = return_image
        # if the face detection method is callable, use it
        self.face_detection_method = face_detection_method
        if callable(face_detection_method):
            self.box_localization = lambda image, min_conf: face_detection_method(image, min_conf)
        else:
            self.box_localization = lambda image, min_conf: face_boxes_detection(image, self.face_detection_method, min_conf)

    def detect(self, image: Image) -> ImageDict:
        """
        detect emotions from image,
        NOTE: images and boxes returned are numpy images
        :param image: an PIL or numpy (RGB) image or an image file (use ImageEmotionDetector.load_image to read images)
        :return: a dictionary {boxes, predictions [, image] [, square_images]}, or an empty dictionary if nothing is detected
        """
        # 1. assert that the model has right needed components =================================================
        assert hasattr(self.model, 'preprocessing'), "the model needs to have the parameter 'preprocessing' (" \
                                                     "preprocessing layers or function that takes a batch of images " \
                                                     "as input), \nplease set model.preprocessing = your_function "
        assert hasattr(self.model, 'emotions'), "the model needs to have the parameter 'emotions' (emotion names) " \
                                                "\nplease set model.emotions = [emotion names here]"
        # 2. detect face positions ==============================================================================
        if isinstance(image, str):
            image = self.load_image(image)
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
        result = ImageDict(predictions=predictions, boxes=boxes, emotions=self.model.emotions)
        if self.return_image:
            result.image = np_image
        if self.return_square_images:
            result.square_images = square_images
        return result

    __call__ = detect

    @staticmethod
    def load_image(file_path):
        return load_img(file_path)


class VideCapture:
    def __init__(self, video_src, width=480, height=640, resize_factor=1.0,
                 window='capture', save_to_file=None,
                 fps=1000, video_speed=1.0):
        """
        A wrapper for cv2.VideoCapture
        :param fps: frames per second
        :param video_src: 0 for front camera, 1 for back camera, or filepath for video saved on disk
        :param width: attempt to load with this width (doesn't work with saved videos - use resize_factor instead)
        :param height: attempt to load with this height (doesn't work with saved videos - use resize_factor instead)
        :param resize_factor: resizes the image according to factor (default = 1.0 — no change)
        :param window: name of the window
        :param save_to_file: save video recording to specified file
        :param video_speed: accelerate video, should be > 1.0, only works with saved videos
        """
        self.resize_factor = resize_factor
        self.cap: cv2.VideCapture = None
        self.video_src = video_src
        self.window = window
        self.shape = (width, height)
        self.fps = max(int(fps), 1)  # make sure it's an integer and that it's bigger than 0
        self.save_to_file = save_to_file
        self.last_read_time = None  # use to calculate time delay
        self.video_speed = max(video_speed, 1.0)

    def __enter__(self):
        self.last_read_time = None
        self.cap = cv2.VideoCapture(self.video_src)
        self.set_resolution(*self.shape)
        if self.save_to_file is not None:
            create_path(self.save_to_file, is_file=True)
        return self

    def __exit__(self, _type, value, traceback):
        self.cap.release()
        cv2.destroyWindow(self.window)

    def show(self, image):
        cv2.imshow(self.window, image[:, :, ::-1])
        # Stop if escape key is pressed
        k = cv2.waitKey(self.wait_time()) & 0xff
        if k in map(ord, '0q'):
            return False  # press these keys to stop
        return True

    def wait_time(self, base=False):
        assert self.cap is not None, "capture has not been initialized yet"
        if self.is_live_capture():
            wait_time = int(1000 / self.fps)
            return max(wait_time, 1)
        elif base:
            wait_time = int(1000 / self.cap.get(cv2.CAP_PROP_FPS) - 1)
            return max(wait_time, 1)
        else:
            wait_time = int(1000 / self.cap.get(cv2.CAP_PROP_FPS) + 1000 / self.fps - 1)
            return max(wait_time, 1)

    def read(self):
        if self.last_read_time is None:  # first read
            self.last_read_time = time.time()
            delay = 0.0
        else:
            delay = time.time() - self.last_read_time
            self.last_read_time = time.time()
        success, frame = self.cap.read()
        if not success:
            raise VideCapture.LastFrameException("last frame reached")
        # if is live video, skip the right amount of frames according to delay
        if not self.is_live_capture():
            for i in range(self.frames_to_skip(delay)):
                self.cap.read()
        if self.resize_factor != 1.0:
            w_old, h_old, _ = frame.shape
            w_new, h_new = round(w_old * self.resize_factor), round(h_old * self.resize_factor)
            frame = cv2.resize(frame, (h_new, w_new))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # cv2 uses BGR be default, so convert to RGB
        return frame

    def set_resolution(self, width, height):
        assert self.cap is not None, "capture has not been initialized yet"
        if width is None or height is None:
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def is_live_capture(self):
        return self.video_src in [FIRST_CAMERA, SECOND_CAMERA]

    def frames_to_skip(self, delay):
        # skip according to delay in respect to base frame rate
        n_frames = int(1000 * delay / self.wait_time(base=True) * self.video_speed)
        # print(n_frames, "skipped frames")
        return n_frames

    def play(self):
        """
        play the video as it is
        """
        with self as capture:
            while True:
                image = capture.read()
                if not capture.show(image):
                    break

    class LastFrameException(Exception):
        pass


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

    def detect(self, video_capture: VideCapture = None):
        """
        :param video_capture: ed.VideoCapture object (ed is the emotion_detection module)
        :return: log of all
        """
        if video_capture is None:
            video_capture = VideCapture(0)
        with video_capture as capture:
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
                boxes = results['boxes']
                predictions = results['predictions']
                results.drawer = self.drawer  # add same drawer to the result
                # draw boxes and margin
                np_image = self.drawer.draw_boxes_and_margin(image, predictions, boxes)
                # append results to log
                log.append(results)
                # show image
                if not capture.show(np_image):
                    break

        cv2.destroyAllWindows()
        return log

    __call__ = detect
