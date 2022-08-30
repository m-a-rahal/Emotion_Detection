# code implementation of Woo et al. paper "CBAM: Convolutional Block Attention Module":  
# Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). Cbam: Convolutional block attention module. In Proceedings of the European conference on computer vision (ECCV) (pp. 3-19).
from tensorflow.keras.layers import Conv2D,Flatten,MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Input,BatchNormalization,Dropout,Dense,Concatenate, Add, Activation, Multiply, Reshape
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf

#==================================================================================================
#=== Channel Attention ==================================================================================================
#==================================================================================================

def channel_attention_mask(tensor, ratio=8):
    _,w,h,c = tensor.shape
    assert c >= ratio, "there should be at least {ratio} channels in the input of channel_attention_mask (equal to ratio)"
    shared_mlp = Sequential([
        Dense(c//ratio,activation='relu', kernel_initializer='he_normal', bias_initializer='zeros'),
        Dense(c, kernel_initializer='he_normal', bias_initializer='zeros')
    ])
    max_pool = shared_mlp(GlobalAveragePooling2D()(tensor))
    avg_pool = shared_mlp(GlobalMaxPooling2D()(tensor))
    add_pool = Add()([max_pool, avg_pool])
    sigmoid  = Activation('sigmoid')(add_pool)
    att_mask = Reshape((1,1,c))(sigmoid)
    return att_mask


#==================================================================================================
#=== Spacial Attention ==================================================================================================
#==================================================================================================

def spacial_attention_mask(tensor, kernel_shape=None):
    ''' the paper Woo et al. recomended a 7x7 kernel_shape,
    however, you can leave kernel_shape = None to take the same spacial dimensions as input (tensor), 
    i.e. if tensor is shaped (_,w,h,c), the kernel shape will be (w,h)'''
    # getting shapes
    _,w,h,c = tensor.shape
    if kernel_shape is None: kernel_shape = (w,h)
    # calculate attention mask
    mask_conv = Conv2D(1, kernel_shape ,padding='same',activation='sigmoid',kernel_initializer='he_normal', use_bias=False)
    max_pool  = tf.reduce_max(tensor, axis=3, keepdims=True)
    avg_pool  = tf.reduce_mean(tensor, axis=3, keepdims=True)
    combined_pool = Concatenate(axis=3)([max_pool, avg_pool])
    att_mask  = mask_conv(combined_pool)
    return att_mask

#===============================================================================================
#=== CBAM BLOCK ================================================================================
#===============================================================================================

def cbam_attention(tensor, ratio=8, kernel_shape=None):
    # apply masks in order : 
    # - channel attention
    # - spacial attention
    tensor = Multiply()([tensor, channel_attention_mask(tensor, ratio=ratio)])
    tensor = Multiply()([tensor, spacial_attention_mask(tensor, kernel_shape=kernel_shape)])
    return tensor


#===============================================================================================
#=== Example of usage ==========================================================================
#===============================================================================================
#--- 🎁 TIP ------------------------------------------------------------------------------------
# if you want to hide the inner layers of these blocks, use keras functional API like this :

if __name__ == '__main__':
    class CBAM(object):
        def __init__(self):
            self.count = 1
        def next_block(self,tensor, name='CBAM_block', ratio=8, kernel_shape=None):
            block =  Model(tensor, cbam_attention(tensor, ratio=ratio, kernel_shape=kernel_shape), name=name+f'_{self.count}')
            self.count += 1
            return block

    # Model definition
    cbam = CBAM()
    inputs = x = Input((224,224,3))
    x = Conv2D(32, 3, padding='same')(x)
    x = Conv2D(32, 3, padding='same')(x)
    x = cbam.next_block(x)(x)
    x = MaxPooling2D()(x)
    x = Conv2D(64, 3, padding='same')(x)
    x = Conv2D(64, 3, padding='same')(x)
    x = cbam.next_block(x)(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, 3, padding='same')(x)
    x = Conv2D(128, 3, padding='same')(x)
    x = cbam.next_block(x)(x)
    x = MaxPooling2D()(x)

    """ summary
    ____________________________________________________________________________________________________
     Layer (type)                                Output Shape                            Param #        
    ====================================================================================================
     input_1 (InputLayer)                        [(None, 48, 48, 3)]                     0              
                                                                                                        
     conv2d_1 (Conv2D)                           (None, 48, 48, 32)                      896            
                                                                                                        
     conv2d_2 (Conv2D)                           (None, 48, 48, 32)                      9248           
                                                                                                        
     CBAM_block_1 (Functional)                   (None, 48, 48, 32)                      4900           
                                                                                                        
     max_pooling2d (MaxPooling2D)                (None, 24, 24, 32)                      0              
                                                                                                        
     conv2d_4 (Conv2D)                           (None, 24, 24, 64)                      18496          
                                                                                                        
     conv2d_5 (Conv2D)                           (None, 24, 24, 64)                      36928          
                                                                                                        
     CBAM_block_2 (Functional)                   (None, 24, 24, 64)                      2248           
                                                                                                        
     max_pooling2d_1 (MaxPooling2D)              (None, 12, 12, 64)                      0              
                                                                                                        
     conv2d_7 (Conv2D)                           (None, 12, 12, 128)                     73856          
                                                                                                        
     conv2d_8 (Conv2D)                           (None, 12, 12, 128)                     147584         
                                                                                                        
     CBAM_block_3 (Functional)                   (None, 12, 12, 128)                     4528           
                                                                                                        
     max_pooling2d_2 (MaxPooling2D)              (None, 6, 6, 128)                       0              
                                                                                                        
    ====================================================================================================
    Total params: 298,684
    Trainable params: 298,684
    Non-trainable params: 0
    ____________________________________________________________________________________________________
    """