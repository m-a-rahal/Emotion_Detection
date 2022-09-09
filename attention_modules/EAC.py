from keras.models import Model
from keras.layers import Layer
from keras.metrics import Mean, mean_squared_error
import tensorflow as tf
import numpy as np
import os
from pathlib import Path


class EACModelWrapper(Model):
    def __init__(self, model: Model, last_features: Layer, gap_layer: Layer, output_dense_layer: Layer,
                 lambda_factor=5.0, ignore_sample_weight_for_eac_loss=True,
                 preprocessing_layers=None):
        """ implementation of the paper :
         « Learn From All: Erasing Attention Consistency for Noisy Label Facial Expression Recognition »
         by Y. Zhang, C. Wang, X. Ling, et W. Deng,
         NOTE: to save model during training, please use the "SaveBest" callback defined below
         :param lambda_factor: loss = classification_loss + λ * consistency_loss (default=5.0)
         :param last_features: the model's last features before the GlobalAveragePooling layer
         :param gap_layer: the model's GlobalAveragePooling layer (GAP)
         :param output_dense_layer: the model's output neurons that come straight after GAP layer
         :param ignore_sample_weight_for_eac_loss: turning this to True might cause loss spikes when sample weights are high
        """
        super().__init__(name=model.name)
        self.ignore_sample_weight_for_eac_loss = ignore_sample_weight_for_eac_loss
        self.preprocessing_layers = preprocessing_layers
        self.lambda_factor = lambda_factor
        self.model = model
        # define sub-model that ends right before GAP
        self.cnn = Model(model.input, last_features.output, name='CNN')
        self.fc = Model(gap_layer.input, model.output)
        self.output_layer = output_dense_layer
        self.mean_eac_loss = Mean(name='eac_loss')

    def eac_loss(self, x1, x2, y_true, sample_weight):
        # generate features
        features1 = self.cnn(x1, training=True)
        features2 = self.cnn(x2, training=True)
        # only pass non-flipped features to get predictions
        y_pred = self.fc(features1, training=True)
        # compute classification loss
        if self.ignore_sample_weight_for_eac_loss:
            sample_weight = None
        classif_loss = self.compiled_loss(y_true, y_pred, regularization_losses=self.losses,
                                          sample_weight=sample_weight)
        # get weights of last layer
        weights = self.output_layer.trainable_weights[0]
        # compute CAM maps form features
        cam1 = features1 @ weights
        cam2 = features2 @ weights
        # flip cam2
        cam2 = cam2[:, :, ::-1, :]
        # calculate consistency loss, mean-squared difference between CAM maps
        consist_loss = mean_squared_error(cam1, cam2)
        # final loss
        loss = classif_loss + self.lambda_factor * consist_loss
        return loss, y_pred

    def compute_cam(self, tensor, training=False):
        """
        computes CAM map as described in « Learning Deep Features for Discriminative Localization », 2015,
        by B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, et A. Torralba, doi: 10.48550/ARXIV.1512.04150.
        :param tensor: input tensor
        :param training:
        :return: returns the CAM map
        """
        # generate features
        features = self.cnn(tensor, training=training)
        # get weights of last layer
        weights = self.output_layer.trainable_weights[0]
        # compute CAM maps form features
        return features @ weights

    # see this tutorial for more details about custom training loops :
    # “Customize what happens in Model.fit” https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
    def train_step(self, data):
        if len(data) == 3:
            x1, y_true, sample_weight = data
        else:
            sample_weight = None
            x1, y_true = data
        # data would be already cropped with the help of the ImageDataGenerator
        if self.preprocessing_layers:
            x1 = self.preprocessing_layers(x1, training=True)
        # generate flipped image
        x2 = x1[:, :, ::-1, :]
        with tf.GradientTape() as tape:
            loss, y_pred = self.eac_loss(x1, x2, y_true, sample_weight)

        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y_true, y_pred, sample_weight=sample_weight)
        self.mean_eac_loss.update_state(loss)
        # Return a dict mapping metric names to current value
        res = {m.name: m.result() for m in self.metrics}
        res.update({"eac_loss": self.mean_eac_loss.result()})
        return res

    def reset_metrics(self):
        super(EACModelWrapper, self).reset_metrics()
        self.mean_eac_loss.reset_state()

    def call(self, inputs, training=None, mask=None):
        if self.preprocessing_layers:
            inputs = self.preprocessing_layers(inputs, training=training)
        return self.model.call(inputs, training, mask)

    def save_weights(self, *args, **kwargs):
        return self.model.save_weights(*args, **kwargs)

    def save(self, *args, **kwargs):
        return self.model.save(*args, **kwargs)

    @property
    def layers(self):
        return self.model.layers


# random eraser implementation
# source : https://github.com/yu4u/cutout-random-erasing


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1 / 0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        if input_img.ndim == 3:
            img_h, img_w, img_c = input_img.shape
        elif input_img.ndim == 2:
            img_h, img_w = input_img.shape

        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w - w)  # modified here
            top = np.random.randint(0, img_h - h)  # modified here

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            if input_img.ndim == 3:
                c = np.random.uniform(v_l, v_h, (h, w, img_c))
            if input_img.ndim == 2:
                c = np.random.uniform(v_l, v_h, (h, w))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w] = c

        return input_img

    return eraser


class SaveBest(tf.keras.callbacks.Callback):
    def __init__(self, monitored_metric, mode, save_file, threshold=0.0,
                 save_best_only=True, save_weights_only=False, include_optimizer=False):
        """
        :param threshold: threshold of improvement to consider, default 0.0 will accept any improvement
        :param monitored_metric: str name of the metric to monitor form training log
        :param mode: 'max' to maximize the metric, 'min' to minimized
        """
        super(SaveBest, self).__init__()
        self.monitored_metric = monitored_metric
        self.best_value = None  # monitor metric best value
        assert mode in ['max', 'min'], 'mode must be either "max" or "min"'
        self.mode = mode
        self.threshold = abs(threshold)
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        assert os.path.exists(Path(save_file).parent), f"the folder {Path(save_file).parent} does not exist! please " \
                                                       f"create it "
        self.save_file = save_file
        self.include_optimizer = include_optimizer

    def improved(self, old_value, new_value):
        if old_value is None:
            return True
        if self.mode == 'max':
            return new_value > old_value + self.threshold
        else:
            return new_value < old_value - self.threshold

    def save_model(self):
        if self.save_weights_only:
            self.model.save_weights(self.save_file)
        else:
            self.model.save(self.save_file, include_optimizer=self.include_optimizer)

    def on_epoch_end(self, epoch, logs=None):
        new_value = logs[self.monitored_metric]
        if self.improved(self.best_value, new_value) or not self.save_best_only:
            self.save_model()
            self.best_value = new_value


class OnEpochEndCallback(tf.keras.callbacks.Callback):
    def __init__(self, behavior):
        """
        This is a generic callback that takes a callable object describing what to do at end of epoch
        :param behavior: a callable object that takes 3 parameters: epoch, logs and model, and performs changes
        """
        self.behavior = behavior

    def on_epoch_end(self, epoch, logs=None):
        self.behavior(epoch, logs, self.model)
