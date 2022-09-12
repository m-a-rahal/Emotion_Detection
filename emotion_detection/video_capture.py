import time
import cv2
from emotion_detection.utils.file_manager import create_path

FIRST_CAMERA = 0
SECOND_CAMERA = 1


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
