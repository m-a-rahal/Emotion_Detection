from tensorflow.keras.layers import Resizing, Rescaling, Lambda, Input
from tensorflow.keras.models import load_model, Sequential
import tensorflow as tf

# model names
PRESET_MobileNet_FER2013 = 0
PRESET_MobileNet_FERplus = 1
# model paths
MobileNet_FER2013_path = 'emotion_detection/models/mobilenet_fer2013.h5'
MobileNet_FERplus_path = 'emotion_detection/models/mobilnet_ferplus.h5'
# emotion names
MobileNet_FER2013_emotions = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
MobileNet_FERplus_emotions = MobileNet_FER2013_emotions + ['contempt', 'unknown']
# preprocessing
MobileNet_preprocessing = Sequential([
    # Input((224,224, 3)),
    Resizing(224, 224)
])
RCA_preprocessing = Sequential([
    Lambda(tf.image.rgb_to_grayscale),  # turn image to grayscale
    Rescaling(scale=1.0 / 255),  # normalize to [0, 1] range
    Resizing(48, 48)
])


def load_default_model(preset=0):
    if preset == PRESET_MobileNet_FER2013:
        model = load_model(MobileNet_FER2013_path)
        model_preset(model, preset)
        return model
    elif preset == PRESET_MobileNet_FERplus:
        model = load_model(MobileNet_FERplus_path)
        model_preset(model, preset)
        return model
    return None


def model_preset(model, preset):
    if preset == PRESET_MobileNet_FER2013:
        # add preprocessing
        model.preprocessing = MobileNet_preprocessing
        # add emotions
        model.emotions = MobileNet_FER2013_emotions
    elif preset == PRESET_MobileNet_FERplus:
        # add preprocessing
        model.preprocessing = MobileNet_preprocessing
        # add emotions
        model.emotions = MobileNet_FERplus_emotions


def get_max_emotion(emotions, prediction):
    """ :returns : probability, emotion"""
    label = tf.argmax(prediction)
    return tf.reduce_max(prediction) * 100, emotions[int(label)]
