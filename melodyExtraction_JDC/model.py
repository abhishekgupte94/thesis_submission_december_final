
import os

os.environ['KERAS_BACKEND'] = "tensorflow"
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
# import keras.backend as KK
# from keras import backend as K
from keras import layers
from keras import Model
from keras import regularizers
# import tensorflow as tf
import torchvision
import torchaudio
import torch.nn
# from keras.layers.advanced_activations import LeakyReLU
# from keras.models import Model
# from keras.layers.core import Dense, Dropout, Activation
# from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,Dropout,\
#                         LSTM,Reshape,Bidirectional,TimeDistributed,Input,add,concatenate,Lambda

import math

def ResNet_Block(input,block_id,filterNum):
    ''' Create a ResNet block
    Args:
        input: input tensor
        filterNum: number of output filters
    Returns: a keras tensor
    '''
    x = layers.BatchNormalization()(input)
    x = layers.LeakyReLU(0.01)(x)
    x = layers.MaxPooling2D((1, 4))(x)

    init = layers.Conv2D(filterNum, (1, 1), name='conv'+str(block_id)+'_1x1', padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    x = layers.Conv2D(filterNum, (3, 3), name='conv'+str(block_id)+'_1',padding='same',kernel_initializer='he_normal',use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.01)(x)
    x = layers.Conv2D(filterNum, (3, 3),  name='conv'+str(block_id)+'_2',padding='same',kernel_initializer='he_normal',use_bias=False)(x)

    x = layers.add([init, x])
    return x


def melody_ResNet_joint_add(options):
    num_output = int(45 * 2 ** (math.log(options.resolution, 2)) + 2)
    input = layers.Input(shape=(options.input_size, options.num_spec, 1))

    block_1 = layers.Conv2D(64, (3, 3), name='conv1_1', padding='same', kernel_initializer='he_normal', use_bias=False,
                     kernel_regularizer=regularizers.l2(1e-5))(input)
    block_1 = layers.BatchNormalization()(block_1)
    block_1 = layers.LeakyReLU(0.01)(block_1)
    block_1 = layers.Conv2D(64, (3, 3), name='conv1_2', padding='same', kernel_initializer='he_normal', use_bias=False,
                     kernel_regularizer=regularizers.l2(1e-5))(block_1)

    block_2 = ResNet_Block(input=block_1, block_id=2, filterNum=128)
    block_3 = ResNet_Block(input=block_2, block_id=3, filterNum=192)
    block_4 = ResNet_Block(input=block_3, block_id=4, filterNum=256)

    block_4 = layers.BatchNormalization()(block_4)
    block_4 = layers.LeakyReLU(0.01)(block_4)
    block_4 = layers.MaxPooling2D((1, 4))(block_4)
    block_4 = layers.Dropout(0.5)(block_4)

    numOutput_P = 2 * block_4.shape[-1]
    output = layers.Reshape((options.input_size, numOutput_P))(block_4)

    output = layers.Bidirectional(layers.LSTM(256, return_sequences=True, recurrent_dropout=0.3, dropout=0.3))(output)
    output = layers.TimeDistributed(layers.Dense(num_output))(output)
    output = layers.TimeDistributed(layers.Activation("softmax"), name='output')(output)

    block_1 = layers.MaxPooling2D((1, 4 ** 4))(block_1)
    block_2 = layers.MaxPooling2D((1, 4 ** 3))(block_2)
    block_3 = layers.MaxPooling2D((1, 4 ** 2))(block_3)

    joint = layers.concatenate([block_1, block_2, block_3, block_4])
    joint = layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', use_bias=False,
                   kernel_regularizer=regularizers.l2(1e-5))(joint)
    joint = layers.BatchNormalization()(joint)
    joint = layers.LeakyReLU(0.01)(joint)
    joint = layers.Dropout(0.5)(joint)

    num_V = joint.shape[-1] * 2
    output_V = layers.Reshape((options.input_size, num_V))(joint)

    output_V = layers.Bidirectional(layers.LSTM(32, return_sequences=True, stateful=False, recurrent_dropout=0.3, dropout=0.3))(
        output_V)
    output_V = layers.TimeDistributed(layers.Dense(2))(output_V)
    output_V = layers.TimeDistributed(layers.Activation("softmax"))(output_V)

    output_NS = layers.Lambda(lambda x: x[:, :, 0])(output)
    output_NS = layers.Reshape((options.input_size, 1))(output_NS)
    output_S = layers.Lambda(lambda x: 1 - x[:, :, 0])(output)
    output_S = layers.Reshape((options.input_size, 1))(output_S)
    output_VV = layers.concatenate([output_NS, output_S])

    output_V = layers.add([output_V, output_VV])
    output_V = layers.TimeDistributed(layers.Activation("softmax"), name='output_V')(output_V)

    model = Model(inputs=input, outputs=[output, output_V])
    return model

