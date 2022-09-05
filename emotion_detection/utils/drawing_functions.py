import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from emotion_detection.models import get_max_emotion


# images ============================================================================================
def show_images(images, predictions, emotions, figsize=(12, 12),
                            font={'family': 'monospace', 'weight': 'bold', 'size': 16}):
    # assert emotions is not None, "please intorduce emotion names, eg. : emotions=['happy', 'sad', ...]"
    plt.figure(figsize=figsize)
    n = int(len(images) ** 0.5)
    ncols = n
    nrows = n + math.ceil((len(images) - n * n) / n)
    for i, (image, pred) in enumerate(zip(images, predictions)):
        plt.subplot(nrows, ncols, i + 1)
        image = image.astype(np.uint8)
        # remove pseudo grayscale channel (if there is one)
        if image.shape[-1] == 1:
            image = np.squeeze(image, axis=-1)
        # make image RGB if grayscale
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)
        # show image
        plt.imshow(image)
        # title of the image
        prob, emotion = get_max_emotion(emotions, pred)
        percentage = f'{prob:.1f}%'
        plt.title(f'({i + 1}) {emotion} {percentage}', color='black', **font)
        plt.axis("off")
    plt.show()


def show_images_truth(x, y_pred, y_true, emotions=None, figsize=(12, 12),
                font={'family': 'monospace', 'weight': 'bold', 'size': 16}, normalize=True):
    assert emotions is not None, "please intorduce emotion names, eg. : emotions=['happy', 'sad', ...]"
    plt.figure(figsize=figsize)
    N = x.shape[0]
    n = int(N ** 0.5)
    ncols = n
    nrows = n + math.ceil((N - n * n) / n)
    for i, (image, label, true_label) in enumerate(zip(x, y_pred, y_true)):
        ax = plt.subplot(ncols, nrows, i + 1)
        # normalize values
        if normalize:
            image = 255 * np.linalg.norm(image, keepdims=True, axis=-1)
        image = image.astype(np.uint8)
        # remove pseudo grayscale channel (if there is one)
        if image.shape[-1] == 1:
            image = np.squeeze(image, axis=-1)
        # make image RGB if grayscale
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)
        # show image
        plt.imshow(image)
        # title of the image
        true_emotion = emotions[int(true_label.argmax())]
        pred_emotion = emotions[int(label.argmax())]
        precentage = f'{100 * label.max():.1f}%'
        if pred_emotion == true_emotion:
            plt.title(f'{pred_emotion} {precentage}', color='darkgreen', **font)
        else:
            plt.title(f'{pred_emotion} {precentage} ({true_emotion})', color='red', **font)
        plt.axis("off")
    plt.show()


# dataframes ========================================================================================

aliases = {'angry': 'anger', 'sad': 'sadness', 'happy': 'happiness'}


def color_map(color, max_alpha, min_alpha):
    max_alpha = int(max_alpha, 16) / 255.0  # convert to float between 0 and 1
    min_alpha = int(min_alpha, 16) / 255.0  # convert to float between 0 and 1

    def map_(value):
        alpha = max(255 * value * max_alpha, 255 * min_alpha)
        alpha = '%02x' % round(alpha)  # value is between 00 and FF
        return f'background-color : {color}{alpha}'

    return map_


def table_styler(emotions, colors={}, highlight_color='', max_alpha='55', min_alpha='09'):
    default_colors = {'anger': '#AA0000',
                      'fear': '#AA00AA',
                      'neutral': '#555599',
                      'sadness': '#0000AA',
                      'surprise': '#887711',
                      'happiness': '#AA5500',
                      'disgust': '#00AA00',
                      'contempt': '#440044',
                      'unknown': '#00CCFF'}
    default_colors.update({emotion: default_colors[aliases[emotion]] for emotion in aliases})
    default_colors.update(colors)
    colors = default_colors

    def styler(style):
        # make values look like percentages
        style.format(lambda value: f'{100 * round(value, 2):.02f}%')
        # give color code to columns the columns
        for emotion in emotions:
            style.applymap(color_map(colors[emotion], max_alpha, min_alpha), subset=[emotion])
        style.highlight_max(color='', axis=1, props=f'font-weight: bold; color: {highlight_color}')
        return style

    return styler


def show_dataframe(y_pred, y_true=None, emotions=[], style_formatter=table_styler):
    assert len(emotions) > 0, 'please input emotions list'
    image_number = y_pred.shape[0]
    lines = []
    for i in range(image_number):
        if y_true:
            lines.append(y_true[i])
        lines.append(y_pred[i])
    if y_true:
        results = pd.DataFrame(data=np.concatenate((y_true, y_pred), axis=1),
                               columns=emotions,
                               index=pd.MultiIndex.from_product(
                                   [[f'image {i + 1}' for i in range(image_number)], ['true', 'pred']]))
    else:
        results = pd.DataFrame(data=np.stack(lines, axis=0),
                               columns=emotions,
                               # index=pd.MultiIndex.from_product([[f'image {i+1}' for i in range(image_number)],
                               # ['pred']]))
                               index=[f'image {i + 1}' for i in range(image_number)])
    return results.style.pipe(style_formatter(emotions))


# text and boxes ============================================================================================
def add_text_under_box(text, image, box, color=(0, 255, 0), text_color=(0, 0, 0), text_size=0.6,
                       text_thickness=3.4, box_thickness=2):
    (x, y, w, h) = box
    # Prints the text.
    margin = int(20 * text_size)
    cv2.rectangle(image, (x, y - 2 * margin), (x + w, y), color, -1)
    cv2.rectangle(image, (x, y - 2 * margin), (x + w, y), color, box_thickness)
    text_thickness = max(1, int(text_thickness * text_size))
    cv2.putText(image, text, (x + 5, y - margin + 5),
                cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, text_thickness)


def create_probabilities_text_image(probs, image: np.ndarray, emotions, text_color, correct_text_color, text_size=0.6,
                                    text_thickness=3.4):
    probs = 100 * probs
    w, h, c = image.shape
    margin_image = np.ones((w, 190, c), image.dtype) * 255
    # fill area with
    # cv2.rectangle(margin_image, (0,0), margin_image.shape[:2], color, -1)
    x, y = 5, 20
    increment = int(40 * text_size)
    labels = {}
    for i in range(len(emotions)):
        labels[i] = probs[i]

    items = labels.items()

    correct_text_color = np.array(correct_text_color)
    text_thickness = max(1, int(text_thickness * text_size))
    for i, v in items:
        text = f'{emotions[i]:10} {probs[i]:05.2f}%'
        p = (probs[i] / 100) ** 0.5
        txt_clr = correct_text_color * p + np.array(text_color, dtype=np.float32) * (1 - p)
        # text_thickness = max(1, int(text_thickness * text_size))
        cv2.putText(margin_image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, text_size, txt_clr, text_thickness)
        y += increment

    return margin_image
