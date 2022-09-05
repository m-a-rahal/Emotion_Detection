import cv2
import numpy as np
from emotion_detection.models import get_max_emotion
from emotion_detection.utils.drawing_functions import add_text_under_box, create_probabilities_text_image, \
    show_dataframe, show_images


class Drawer:
    def __init__(self, emotions, box_color=(0, 255, 0), box_text_color=(0, 0, 0),
                 selected_box_color=(255, 150, 0), box_thickness=2,
                 emotions_text_color=(200, 200, 200), correct_emotion_text_color=(220, 0, 0),
                 text_size=0.6, text_thickness=3.4):
        """
        All the colors used here are RGB colors (Red, Green, Blue)
        :param emotions: list of emotion names used by the model (in same order)
        :param box_color: color of the box drawn around the face
        :param selected_box_color: when many boxes in an image, this color will highlight the one selected
        :param box_text_color: color if the text drawn above the face
        :param emotions_text_color: color of emotions with ZERO probability displayed on margin/side
        :param correct_emotion_text_color: color of emotions of 100% probability displayed on margin/side
        :param text_size: size of the text
        :param text_thickness: thickness of text
        """
        self.selected_box_color = selected_box_color
        self.text_size = text_size
        self.text_thickness = text_thickness
        self.emotions = emotions
        self.correct_emotion_text_color = correct_emotion_text_color
        self.emotions_text_color = emotions_text_color
        self.box_text_color = box_text_color
        self.box_color = box_color
        self.box_thickness = box_thickness

    def draw_box(self, image, prediction, box):
        (x, y, w, h) = box
        cv2.rectangle(image, (x, y), (x + w, y + h), self.selected_box_color, self.box_thickness)
        prob, emotion = get_max_emotion(self.emotions, prediction)
        text = emotion + ' ' + f'{prob:.2f}'
        add_text_under_box(text, image, (x, y, w, h), color=self.selected_box_color, text_color=self.box_text_color,
                           text_size=self.text_size, text_thickness=self.text_thickness, box_thickness=self.box_thickness)
        return image

    def draw_boxes(self, image, predictions, boxes):
        # make a copy of the image to not override it
        image = np.array(image)
        for i, (x, y, w, h) in enumerate(boxes):
            prediction = predictions[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), self.box_color, self.box_thickness)
            prob, emotion = get_max_emotion(self.emotions, prediction)
            text = emotion + ' ' + f'{prob:.2f}'
            add_text_under_box(text, image, (x, y, w, h), color=self.box_color, text_color=self.box_text_color,
                               text_size=self.text_size, text_thickness=self.text_thickness, box_thickness=self.box_thickness)
        return image
    
    def show_square_images(self, images, predictions, figsize=(12, 12), font={'family': 'monospace', 'weight': 'bold', 'size': 16}):
        # assert emotions is not None, "please introduce emotion names, eg. : emotions=['happy', 'sad', ...]"
        return show_images(images, predictions, self.emotions, figsize, font)

    def add_margin_text(self, image, prediction):
        margin_image = create_probabilities_text_image(prediction, image, self.emotions,
                                                       text_color=self.emotions_text_color,
                                                       correct_text_color=self.correct_emotion_text_color,
                                                       text_size=self.text_size,
                                                       text_thickness=self.text_thickness)
        image = np.concatenate((image, margin_image), axis=1)
        return image

    def show_as_dataframe(self, predictions, true_predictions=None):
        return show_dataframe(predictions, true_predictions, self.emotions)

    def draw_boxes_and_margin(self, image, predictions, boxes, selected_idx=0):
        # Draw boxes onto image
        np_image = np.array(image)  # get copy from image
        # draw all boxes
        np_image = self.draw_boxes(np_image, predictions, boxes)
        # highlight which box is intended by margin
        if len(boxes) > 1:
            np_image = self.draw_box(np_image, predictions[selected_idx], boxes[selected_idx])
        # draw margin
        return self.add_margin_text(np_image, predictions[selected_idx])


