from keras import backend as K
from keras.engine.topology import Layer
from keras.layers.merge import add
from keras.engine import InputSpec
from keras.layers.core import Activation
from keras.layers.convolutional import Conv3D, UpSampling3D
from keras.layers import BatchNormalization, TimeDistributed, Reshape, RepeatVector
from keras import regularizers

import tensorflow as tf
reg_weights = 0.00001

def conv_bn_relu(nb_filter, depth, height, width, stride = (1, 1, 1)):
    def conv_func(x):
        x = Conv3D(nb_filter, (depth, height, width), strides=stride, padding='same', kernel_regularizer=regularizers.l2(reg_weights))(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    return conv_func


def time_conv_bn_relu(nb_filter, depth, height, width, stride = (1, 1, 1)):
    def conv_func(x):
        x = TimeDistributed(Conv3D(nb_filter, (depth, height, width), strides=stride, padding='same', kernel_regularizer=regularizers.l2(reg_weights)))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Activation("relu"))(x)
        return x

    return conv_func


def res_conv(nb_filter, depth, height, width, stride=(1, 1, 1)):
    def _res_func(x):
        identity = x

        a = Conv3D(nb_filter, (depth, height, width), strides=stride, padding='same', kernel_regularizer=regularizers.l2(reg_weights))(x)
        a = BatchNormalization()(a)
        a = Activation("relu")(a)
        a = Conv3D(nb_filter, (depth, height, width), strides=stride, padding='same', kernel_regularizer=regularizers.l2(reg_weights))(a)
        y = BatchNormalization()(a)

        return add([identity, y])

    return _res_func


def time_res_conv(nb_filter, depth, height, width, stride=(1, 1, 1)):
    def _res_func(x):
        identity = x

        a = TimeDistributed(Conv3D(nb_filter, (depth, height, width), strides=stride, padding='same', kernel_regularizer=regularizers.l2(reg_weights)))(x)
        a = TimeDistributed(BatchNormalization())(a)
        a = TimeDistributed(Activation("relu"))(a)
        a = TimeDistributed(Conv3D(nb_filter, (depth, height, width), strides=stride, padding='same', kernel_regularizer=regularizers.l2(reg_weights)))(a)
        y =TimeDistributed(BatchNormalization())(a)

        return add([identity, y])

    return _res_func


# def dconv_bn_nolinear(nb_filter, depth, height, width, stride=(2, 2, 2), activation="relu"):
#     def _dconv_bn(x):
#         x = UnPooling3D(size=stride)(x)
#         x = ReflectionPadding3D(padding=(int(depth/2), int(height/2), int(width/2)))(x)
#         x = Conv3D(nb_filter, (depth, height, width), padding='valid', kernel_regularizer=regularizers.l2(reg_weights))(x)
#         x = BatchNormalization()(x)
#         x = Activation(activation)(x)
#         return x

#     return _dconv_bn

def dconv_bn_nolinear(nb_filter, depth, height, width, stride=(2, 2, 2), activation="relu"):
    def _dconv_bn(x):
        print('size is ', stride)
        
        
        
        x = UnSampling3D(size=stride)(x)
        x = ReflectionPadding3D(padding=(int(depth/2), int(height/2), int(width/2)))(x)
        x = Conv3D(nb_filter, (depth, height, width), padding='valid', kernel_regularizer=regularizers.l2(reg_weights))(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        return x

    return _dconv_bn

# def time_dconv_bn_nolinear(nb_filter, depth, height, width, stride=(2, 2, 2), activation="relu"):
#     def _dconv_bn(x):
#         x = TimeDistributed(UnPooling3D(size=stride))(x)
#         x = TimeDistributed(ReflectionPadding3D(padding=(int(depth/2), int(height/2), int(width/2))))(x)
#         x = TimeDistributed(Conv3D(nb_filter, (depth, height, width), padding='valid', kernel_regularizer=regularizers.l2(reg_weights)))(x)
#         x = TimeDistributed(BatchNormalization())(x)
#         x = TimeDistributed(Activation(activation))(x)
#         return x

#     return _dconv_bn


def time_dconv_bn_nolinear(nb_filter, depth, height, width, stride=(2, 2, 2), activation="relu"):
    def _dconv_bn(x):
        x = TimeDistributed(UpSampling3D(size=stride))(x)
        x = TimeDistributed(ReflectionPadding3D(padding=(int(depth/2), int(height/2), int(width/2))))(x)
        x = TimeDistributed(Conv3D(nb_filter, (depth, height, width), padding='valid', kernel_regularizer=regularizers.l2(reg_weights)))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Activation(activation))(x)
        return x

    return _dconv_bn


class ReflectionPadding3D(Layer):
    def __init__(self, padding=(1, 1, 1), dim_ordering='default', **kwargs):
        super(ReflectionPadding3D, self).__init__(**kwargs)

        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()

        self.padding = padding
        if isinstance(padding, dict):
            if set(padding.keys()) <= {'far_pad', 'near_pad', 'top_pad', 'bottom_pad', 'left_pad', 'right_pad'}:
                
                self.far_pad = padding.get('far_pad', 0)  # depth far
                self.near_pad = padding.get('near_pad', 0) # depth near
                self.top_pad = padding.get('top_pad', 0)
                self.bottom_pad = padding.get('bottom_pad', 0)
                self.left_pad = padding.get('left_pad', 0)
                self.right_pad = padding.get('right_pad', 0)
            else:
                raise ValueError('Unexpected key found in `padding` dictionary. '
                                 'Keys have to be in {"top_pad", "bottom_pad", '
                                 '"left_pad", "right_pad"}.'
                                 'Found: ' + str(padding.keys()))
        else:
            padding = tuple(padding)
            if len(padding) == 3:
                self.far_pad = padding[0]
                self.near_pad = padding[0]
                self.top_pad = padding[1]
                self.bottom_pad = padding[1]
                self.left_pad = padding[2]
                self.right_pad = padding[2]
            elif len(padding) == 6:
                self.far_pad = padding[0]
                self.near_pad = padding[1]
                self.top_pad = padding[2]
                self.bottom_pad = padding[3]
                self.left_pad = padding[4]
                self.right_pad = padding[5]
            else:
                raise TypeError('`padding` should be tuple of int '
                                'of length 2 or 4, or dict. '
                                'Found: ' + str(padding))

        if dim_ordering not in {'tf'}:
            raise ValueError('dim_ordering must be in {tf}.')
        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=5)]

    def call(self, x, mask=None):
        far_pad = self.far_pad
        near_pad = self.near_pad
        top_pad = self.top_pad
        bottom_pad = self.bottom_pad
        left_pad = self.left_pad
        right_pad = self.right_pad

        paddings = [[0, 0], [far_pad, near_pad], [left_pad, right_pad], [top_pad, bottom_pad], [0, 0]]

        return tf.pad(x, paddings, mode='REFLECT', name=None)

    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'tf':
            depths = input_shape[1] + self.far_pad + self.near_pad if input_shape[1] is not None else None
            rows = input_shape[2] + self.top_pad + self.bottom_pad if input_shape[2] is not None else None
            cols = input_shape[3] + self.left_pad + self.right_pad if input_shape[3] is not None else None

            return (input_shape[0],
                    depths,
                    rows,
                    cols,
                    input_shape[4])
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)

    def get_config(self):
        config = {'padding': self.padding}
        base_config = super(ReflectionPadding3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class UnPooling3D(UpSampling3D):
    def __init__(self, size=(2, 2, 2)):
        super(UnPooling3D, self).__init__(size)

    def call(self, x, mask=None):
        shapes = x.get_shape().as_list()
        d = self.size[0] * shapes[1]
        w = self.size[1] * shapes[2]
        h = self.size[2] * shapes[3]
        return tf.image.resize_nearest_neighbor(x, (d, w, h))


class InstanceNormalize(Layer):
    def __init__(self, **kwargs):
        super(InstanceNormalize, self).__init__(**kwargs)
        self.epsilon = 1e-3

    def call(self, x, mask=None):
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, self.epsilon)))

    def compute_output_shape(self, input_shape):
        return input_shape
    
    
class RepeatConv(Layer):
    """Repeats the input n times.
    # Example
    ```python
        model = Sequential()
        model.add(Dense(32, input_dim=32))
        # now: model.output_shape == (None, 32)
        # note: `None` is the batch dimension
        model.add(RepeatVector(3))
        # now: model.output_shape == (None, 3, 32)
    ```
    # Arguments
        n: integer, repetition factor.
    # Input shape
        4D tensor of shape `(num_samples, w, h, c)`.
    # Output shape
        5D tensor of shape `(num_samples, n, w, h, c)`.
    """

    def __init__(self, n, **kwargs):
        super(RepeatConv, self).__init__(**kwargs)
        self.n = n
        self.input_spec = InputSpec(ndim=5)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n, input_shape[1], input_shape[2], input_shape[3], input_shape[4])

    def call(self, inputs):
           
        x = K.expand_dims(inputs, 1)
        pattern = tf.stack([1, self.n, 1, 1, 1, 1])
        return K.tile(x, pattern)

    def get_config(self):
        config = {'n': self.n}
        base_config = super(RepeatConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
