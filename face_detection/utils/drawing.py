import cv2
import numpy as np
import tensorflow as tf

white = np.array((255,255,255), dtype=np.float32)
'''gray = (100, 100, 100)
yellow = (255, 174, 0)
red = (255, 0, 0)
blue = (20, 0, 217)
white = (230, 230, 230)
black = (0, 0, 0)
green = (0, 128, 0)

emotion_colors = {
    'neutral'   : (gray, black),
    'happiness' : (yellow, black),
    'surprise'  : (white, black),
    'sadness'   : (blue, white),
    'anger'     : (red, white),
    'disgust'   : (green, white),
    'fear'      : (brown, white),
    'contempt'  : (purple, white),
    'unknown'   : (black, white)
}'''

def add_text_under_box(text, image, box, color=(0,255,0),text_color=(255,255,255), text_size=0.6):
    (x, y, w, h) = box
    # Prints the text.
    margin = int(20 * text_size)
    cv2.rectangle(image, (x, y - 2*margin), (x + w, y), color, -1)
    cv2.rectangle(image, (x, y - 2*margin), (x + w, y), color, 2)
    cv2.putText(image, text, (x+5, y - margin+5),
                        cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, 2)

def create_probabilites_text_image(probs, image : np.ndarray, model,
        color=(50, 50, 50),text_color=(255,255,255), correct_text_color=(0,255,0), text_size=0.6, sort_labels=False):
    probs = 100*probs[0]
    w,h,c = image.shape
    margin_image = np.zeros((w,190,c), image.dtype)
    # fill area with
    cv2.rectangle(margin_image, (0,0), margin_image.shape[:2], color, -1)
    x,y = 5,20
    increment = int(40*text_size)
    label = tf.argmax(probs)
    labels = {}
    for i in range(len(model.emotion_names)):
        labels[i] = probs[i]

    items = labels.items()
    if sort_labels:
        sort(items, key = lambda x : x[1], reverse=True)
    
    correct_text_color = np.array(correct_text_color)
    for i,v in items:
        text = f'{model.emotion_names[i]:10} {probs[i]:05.2f}%'
        p = (probs[i] / 100)**0.5
        txt_clr = correct_text_color * p + white * (1-p)
        cv2.putText(margin_image, text, (x,y),cv2.FONT_HERSHEY_SIMPLEX, text_size, txt_clr, 2)
        y += increment

    return margin_image