import numpy as np
from PIL import Image


def make_square_images(image: Image, boxes):
    good_boxes = []
    square_images = []
    for box in boxes:
        try:
            # try to capture a square around the face
            square_box = turn_to_square(*box)
            cropped = crop(square_box, image)
            square_images.append(np.array(cropped))
            good_boxes.append(square_box)
        except InvalidCroppedImageException:  # if box is of invalid shape
            # try again without turning into box
            try:
                cropped = crop(box, image)
                square_images.append(np.array(cropped))
                good_boxes.append(box)
            except InvalidCroppedImageException:
                continue
            continue

    return good_boxes, square_images


def crop(box, image: Image):
    x, y, w, h = box
    try:
        cropped = image.crop((x, y, x + w, y + h))
        return cropped
    except Exception:
        raise InvalidCroppedImageException()


class InvalidCroppedImageException(Exception):
    pass


def turn_to_square(x, y, w, h):
    if w != h:
        m = max(w, h)
        center = (x + w // 2, y + h // 2)
        x = center[0] - m // 2
        y = center[1] - m // 2
        w = h = m
    return x, y, w, h
