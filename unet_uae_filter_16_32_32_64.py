from layers import *

from keras import backend as K
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, concatenate, TimeDistributed, RepeatVector, ConvLSTM2D
from convolutional_recurrent_3d import ConvLSTM3D
#from zoo.pipeline.api.keras.layers.convolutional_recurrent import ConvLSTM3D
from keras.models import Model

import numpy as np


def create_vae(input_shape, depth):
    # Encoder
    input = Input(shape=input_shape, name='image')

    enc1 = conv_bn_relu(16, 3, 3, 3, stride=(2, 2, 2))(input)
    time_enc1 = RepeatConv(depth)(enc1)
    enc2 = conv_bn_relu(32, 3, 3, 3, stride=(1, 1, 1))(enc1)
    time_enc2 = RepeatConv(depth)(enc2)
    enc3 = conv_bn_relu(32, 3, 3, 3, stride=(2, 2, 2))(enc2)
    time_enc3 = RepeatConv(depth)(enc3)
    #enc4 = conv_bn_relu(128, 3, 3, 3, stride=(1, 1, 1))(enc3)
    enc4 = conv_bn_relu(64, 3, 3, 3, stride=(1, 1, 1))(enc3)
    time_enc4 = RepeatConv(depth)(enc4)

    # x = res_conv(128, 3, 3, 3)(enc4)
    # x = res_conv(128, 3, 3, 3)(x)
    #x = res_conv(128, 3, 3, 3)(x)
    x = res_conv(64, 3, 3, 3)(enc4)
    x = res_conv(64, 3, 3, 3)(x)
    encoder = Model(input, x, name='encoder')

    x = RepeatConv(depth)(enc4)
    x = ConvLSTM3D(64, (3, 3, 3), strides=(1, 1, 1), padding = 'same', activation='relu', return_sequences = True)(x)

   
    #x = ConvLSTM2D(64, (3, 3), strides=(1, 1), padding = 'same', activation='relu', return_sequences = True)(x)
    #x = ConvLSTM2D(128, (3, 3), strides=(1, 1), padding = 'same', activation='relu', return_sequences = True)(x)


    #x = time_res_conv(128, 3, 3, 3)(x)
    x = time_res_conv(64, 3, 3, 3)(x)
    dec4 = time_res_conv(64, 3, 3, 3)(x)

    merge4 = concatenate([time_enc4, dec4], axis = -1)
    dec3 = time_dconv_bn_nolinear(64, 3, 3, 3, stride=(1, 1, 1))(merge4)
    merge3 = concatenate([time_enc3, dec3], axis = -1)
    dec2 = time_dconv_bn_nolinear(32, 3, 3, 3, stride=(2, 2, 2))(merge3)
    merge2 = concatenate([time_enc2, dec2], axis = -1)
    dec1 = time_dconv_bn_nolinear(32, 3, 3, 3, stride=(1, 1, 1))(merge2)
    
    merge1 = concatenate([time_enc1, dec1], axis = -1)
    dec0 = time_dconv_bn_nolinear(16, 3, 3, 3, stride=(2, 2, 2))(merge1)

    output = TimeDistributed(Conv3D(1, (3, 3, 3), padding='same', activation=None))(dec0)
    
    #print('output shape is ', K.int_shape(output))
    # Full net
    full_model = Model(input, output)

    return full_model, encoder