#  Moddel_2_118 Design
# Add Dropout!!!

from keras.models import Model
from keras.layers.convolutional import Conv3D, ZeroPadding3D
from keras.layers.convolutional import MaxPooling3D, AveragePooling3D
from keras.layers import Dropout, Input
from keras.layers import Flatten, add
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization  # batch Normalization for managing internal covariant shift.
from keras.layers import Activation
from keras.utils import plot_model


def Conv3d_BN(x, nb_filter, kernel_size, strides=1, padding='same', name=None):
    x = Conv3D(nb_filter, kernel_size, padding=padding, data_format='channels_first', strides=strides,
               activation='relu')(x)
    x = BatchNormalization()(x)
    return x


def identity_Block(inpt, nb_filter, kernel_size, strides=1, with_conv_shortcut=False):
    x = Conv3d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv3d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv3d_BN(inpt, nb_filter=nb_filter, strides=strides,
                             kernel_size=kernel_size)
        x = Dropout(0.2)(x)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x


# def bottlneck_Block(inpt, nb_filter, strides=1, with_conv_shortcut=False):
#     k1, k2, k3 = nb_filter
#     x = Conv3d_BN(inpt, nb_filter=k1, kernel_size=1, strides=strides, padding='same')
#     x = Conv3d_BN(x, nb_filter=k2, kernel_size=3, padding='same')
#     x = Conv3d_BN(x, nb_filter=k3, kernel_size=1, padding='same')
#     if with_conv_shortcut:
#         shortcut = Conv3D(inpt, nb_filter=k3, data_format='channels_first', strides=strides, kernel_size=1)
#         x = add([x, shortcut])
#         return x
#     else:
#         x = add([x, inpt])
#         return x


def resnet(shape, classes):
    inpt = Input(shape=shape)
    x = ZeroPadding3D((1, 1, 1), data_format='channels_first')(inpt)

    # conv1
    x = Conv3d_BN(x, nb_filter=16, kernel_size=(6, 6, 6), strides=1, padding='valid')
    x = MaxPooling3D(pool_size=(3, 3, 3), strides=2, data_format='channels_first')(x)

    # conv2_x
    x = identity_Block(x, nb_filter=32, kernel_size=(2, 2, 2), strides=1, with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=32, kernel_size=(2, 2, 2))
#     x = identity_Block(x, nb_filter=64, kernel_size=(3, 3, 3))

    # conv3_x
    x = identity_Block(x, nb_filter=64, kernel_size=(2, 2, 2), strides=1, with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=64, kernel_size=(2, 2, 2))
#     x = identity_Block(x, nb_filter=128, kernel_size=(3, 3, 3))
#     x = identity_Block(x, nb_filter=128, kernel_size=(3, 3, 3))

#     # conv4_x
#     x = identity_Block(x, nb_filter=256, kernel_size=(3, 3, 3), strides=2, with_conv_shortcut=True)
#     x = identity_Block(x, nb_filter=256, kernel_size=(3, 3, 3))
#     x = identity_Block(x, nb_filter=256, kernel_size=(3, 3, 3))

    x = AveragePooling3D(pool_size=(2, 2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=inpt, outputs=x)
    return model
