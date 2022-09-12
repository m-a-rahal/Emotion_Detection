from PIL import Image
from emotion_detection.utils.drawing import Drawer
from emotion_detection.utils.cropping import make_square_images
from tensorflow.keras.utils import load_img


class ImageDict:
    def __init__(self, image=None, boxes=None, predictions=None, emotions=None, square_images=None):
        self.image = image
        self._square_images = square_images
        self.emotions = emotions
        self.predictions = predictions
        self.boxes = boxes

    def __len__(self):
        if self.boxes:
            return len(self.boxes)
        return 0

    def draw_boxes(self, drawer: Drawer = None):
        if drawer is None:
            drawer = Drawer(self.emotions)
        return drawer.draw_boxes(self.image, self.predictions, self.boxes)

    def save(self, folder=None, writer=None, ignore_squares=False, ignore_image=False):
        """
        Saves the dictionary to a folder containing images and predictions
        :param folder:
        :param ignore_squares:
        :param ignore_image:
        :param writer:
        :return:
        """
        pass

    @staticmethod
    def load(file):
        raise Exception('not implemented')
        # return ImageDict(**loaded_dict)

    def show_image(self):
        return Image.fromarray(self.image)

    def show_image_with_boxes(self, drawer: Drawer = None):
        return Image.fromarray(self.draw_boxes(drawer))

    def show_box(self, box_idx):
        return Image.fromarray(self.square_images[box_idx])

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

    def draw_boxes_and_margin(self, selected_idx=0, drawer: Drawer = None):
        if drawer is None:
            drawer = Drawer(self.emotions)
        return drawer.draw_boxes_and_margin(self.image, self.predictions, self.boxes, selected_idx=selected_idx)

    def show_boxes_and_margin(self, selected_idx=0, drawer: Drawer = None):
        return Image.fromarray(self.draw_boxes_and_margin(selected_idx, drawer))

    def show_cam_heatmaps(self, model, argmax_cam=True, update_predictions=True, drawer: Drawer = None):
        if drawer is None:
            drawer = Drawer(self.emotions)
        heatmap_squares, new_predictions = drawer.apply_CAM_heatmaps(model, self.square_images, argmax_cam=argmax_cam)
        if update_predictions:
            self.predictions = new_predictions
        return drawer.show_square_images(heatmap_squares, self.predictions)

    @property
    def square_images(self):
        if self._square_images is None:
            _, self._square_images = make_square_images(self.image, self.boxes)
        return self._square_images

    @square_images.setter
    def square_images(self, value):
        self._square_images = value

    @square_images.deleter
    def square_images(self):
        del self._square_images


def load_image(file_path):
    return load_img(file_path)
