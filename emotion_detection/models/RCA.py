from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Input, \
    BatchNormalization, Dropout, Dense, Activation
from tensorflow.keras.models import Model
from attention_modules.CBAM import CBAM


# ==================================================================================================
# === layer and block naming functions ==================================================================================================
# ==================================================================================================

class Counter(dict):
    def __getitem__(self, name):
        self[name] = value = self.get(name, 0) + 1
        return value


class Layer_namer(Counter):
    def __init__(self, block_number=None, block_name='block'):
        super().__init__()
        self.block_name = block_name
        self.block_number = block_number

    def __call__(self, layer_name):
        text = f'{layer_name}{self[layer_name]}'  # ex: block1_conv2, block10_fc1
        if self.block_number:
            text = f'{self.block_name}{self.block_number}_{text}'
        return text


class Block_namer(Layer_namer):
    def __init__(self, block_name='block', start_number=1):
        super().__init__(block_name=block_name)
        self.block_number = start_number - 1

    def __next__(self):
        self.block_number += 1
        return super().__call__


def architectureRCA(n_classes=7):
    block_namer = Block_namer()
    cbam = CBAM()
    inputs = x = Input(shape=(48, 48, 1), name='input')
    x = double_conv_block(x, 96, (5, 5), cbam=cbam, namer=next(block_namer))
    x = double_conv_block(x, 192, (5, 5), cbam=cbam, namer=next(block_namer))
    x = double_conv_block(x, 768, (3, 3), cbam=cbam, namer=next(block_namer))
    x = double_conv_block(x, 768, (3, 3), cbam=cbam, namer=next(block_namer))

    # Fully connected layers
    # Fully connected layers
    namer = Layer_namer()
    x = Flatten()(x)
    x = dense_block(x, 512, namer=namer)
    x = dense_block(x, 1024, namer=namer)
    outputs = Dense(n_classes, name='output')(x)
    outputs = Activation('softmax')(outputs)
    model = Model(inputs, outputs, name='RCA_fer2013')
    return model


# CAB block
def double_conv_block(x, filters, conv_kernel=(3, 3), activation='relu', dropout_rate=0.25, dropout_multiplier=1,
                      # conv params
                      cbam=None, regularizer=None, *, namer):  # attention params
    """
    :param dropout_multiplier: used multiply filters and dropout together so that dropout can increase without affecting the desired performance
    :param namer: use 'block_namer'
    """
    dropout_rate = dropout_multiplier * dropout_rate
    filters = int(dropout_multiplier * filters)
    # Conv 1
    x = Conv2D(filters, conv_kernel, padding='same', kernel_regularizer=regularizer, name=namer('conv'))(x)
    x = BatchNormalization(name=namer('batch_norm'))(x)
    x = Activation(activation, name=namer('activation'))(x)
    # Attention block (CBAM)
    if cbam:
        x = cbam.next_block(x, kernel_shape=7)(x)
    # Conv 2
    x = Conv2D(filters, conv_kernel, padding='same', kernel_regularizer=regularizer, name=namer('conv'))(x)
    x = BatchNormalization(name=namer('batch_norm'))(x)
    x = Activation(activation, name=namer('activation'))(x)
    # MaxPooling
    x = MaxPooling2D(pool_size=(2, 2), name=namer('maxpool'))(x)
    # Dropout
    if dropout_rate > 0:
        x = Dropout(dropout_rate, name=namer('dropout'))(x)
    return x


# FC block
def dense_block(x, units, activation='relu', dropout_rate=0.5, regularizer=None, *, namer):
    x = Dense(units, kernel_regularizer=regularizer, name=namer('fc'))(x)
    x = BatchNormalization(name=namer('batch_norm'))(x)
    x = Activation(activation, name=namer('activation'))(x)
    x = Dropout(dropout_rate, name=namer('dropout'))(x)
    return x
