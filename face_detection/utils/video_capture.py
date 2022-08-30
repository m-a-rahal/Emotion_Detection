import cv2
class VideCapture:
    def __init__(self, width=None, height=None, window='capture'):
        self.cap = None
        self.window = window
        self.shape = (width, height)
    
    def __enter__(self):
        self.cap = cv2.VideoCapture(0)
        self.set_resolution(*self.shape)
        return self
    
    def __exit__(self,type, value, traceback):
        self.cap.release()
        cv2.destroyWindow(self.window)
    
    def show(self, image):
        cv2.imshow(self.window, image)
        
    def read(self):
        _,frame = self.cap.read()
        return frame
    def set_resolution(self, width, height):
        if width is None or height is None:
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)