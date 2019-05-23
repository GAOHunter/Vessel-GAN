import os

from keras import backend as K
from keras import objectives
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, GlobalAveragePooling2D, Dropout, add, AtrousConvolution2D, Conv2DTranspose
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Activation, Flatten
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard

from groupnormalization import GroupNormalization

os.environ['KERAS_BACKEND'] = 'tensorflow'
K.set_image_dim_ordering('tf')

def ResidualBlock(x, n_filters):

    res = x

    out = Conv2D(filters=n_filters, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    out = GroupNormalization(groups=32, axis=3, scale=False)(out)
    out= Activation('relu')(out)
    out = Conv2D(filters=n_filters, kernel_size=(3, 3), strides=(1,1), padding='same')(out)
    out = GroupNormalization(groups=32, axis=3, scale=False)(out)
    out = Activation('relu')(out)

    res = Conv2D(filters=n_filters, kernel_size=(1,1), strides=(1,1), padding='same')(x)

    out = add([res, out])

    return out

def DACblock(x, n_filters):
    feature1 = AtrousConvolution2D(nb_filter=n_filters, nb_row=3, nb_col=3, atrous_rate=(1,1), border_mode='same')(x)

    feature2 = AtrousConvolution2D(nb_filter=n_filters, nb_row=3, nb_col=3, atrous_rate=(3,3), border_mode='same')(x)
    feature2 = AtrousConvolution2D(nb_filter=n_filters, nb_row=1, nb_col=1, atrous_rate=(1,1), border_mode='same')(feature2)

    feature3 = AtrousConvolution2D(nb_filter=n_filters, nb_row=3, nb_col=3, atrous_rate=(3,3), border_mode='same')(feature1)
    feature4 = AtrousConvolution2D(nb_filter=n_filters, nb_row=3, nb_col=3, atrous_rate=(5, 5), border_mode='same')(feature3)

    feature3 = AtrousConvolution2D(nb_filter=n_filters, nb_row=1, nb_col=1, atrous_rate=(1, 1), border_mode='same')(feature3)
    feature4 = AtrousConvolution2D(nb_filter=n_filters, nb_row=1, nb_col=1, atrous_rate=(1, 1), border_mode='same')(feature4)

    return add([x, feature1, feature2, feature3, feature4])

def RMPblock(x, n_filters,datasets):

    if datasets == 'STARE':
        feature1 = MaxPooling2D(pool_size=(3,3), strides=3)(x)
        feature2 = MaxPooling2D(pool_size=(5,5), strides=5)(x)
        feature3 = MaxPooling2D(pool_size=(9,9), strides=9)(x)
        feature4 = MaxPooling2D(pool_size=(15,15), strides=15)(x)
    elif datasets == 'DRIVE':
        feature1 = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        feature2 = MaxPooling2D(pool_size=(4, 4), strides=4)(x)
        feature3 = MaxPooling2D(pool_size=(5, 5), strides=5)(x)
        feature4 = MaxPooling2D(pool_size=(8, 8), strides=8)(x)

    size1 = x.get_shape()[1:3]

    feature1 = Conv2D(filters=1, kernel_size=(1,1), strides=(1,1))(feature1)
    size2 = feature1.get_shape()[1:3]
    size = size1[0].value / size2[0].value, size1[1].value / size2[1].value
    feature1 = UpSampling2D(size=size, interpolation='bilinear')(feature1)

    feature2 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1))(feature2)
    size2 = feature2.get_shape()[1:3]
    size = size1[0].value / size2[0].value, size1[1].value / size2[1].value
    feature2 = UpSampling2D(size=size, interpolation='bilinear')(feature2)

    feature3 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1))(feature3)
    size2 = feature3.get_shape()[1:3]
    size = size1[0].value / size2[0].value, size1[1].value / size2[1].value
    feature3 = UpSampling2D(size=size, interpolation='bilinear')(feature3)

    feature4 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1))(feature4)
    size2 = feature4.get_shape()[1:3]
    size = size1[0].value / size2[0].value, size1[1].value / size2[1].value
    feature4 = UpSampling2D(size=size, interpolation='bilinear')(feature4)

    return Concatenate(axis=3)([x, feature1, feature2, feature3, feature4])


def generator(img_size, n_filters, tensorboard, name='g'):
    """
    generate network based on unet
    """

    # set image specifics
    k = 3  # kernel size
    s = 2  # stride
    img_ch = 3  # image channels
    out_ch = 1  # output channel
    group = 32  # num of groups
    img_height, img_width = img_size[0], img_size[1]
    padding = 'same'

    inputs = Input((img_height, img_width, img_ch))

    conv1 = Conv2D(n_filters, kernel_size=(k, k), padding='same')(inputs)
    conv1 = GroupNormalization(groups=group, axis=3, scale=False)(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)

    conv2 = ResidualBlock(pool1, 2 * n_filters)
    pool2 = MaxPooling2D(pool_size=(s, s))(conv2)

    conv3 = ResidualBlock(pool2, 4 * n_filters)
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)

    conv4 = ResidualBlock(pool3, 8 * n_filters)
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)

    conv5 = ResidualBlock(pool4, 16 * n_filters)

    across = DACblock(conv5, 16*n_filters)

    # dataset = 'STARE'
    dataset = 'DRIVE'
    up1 = RMPblock(across, 16*n_filters,datasets=dataset)

    up2 = UpSampling2D(size=(s,s))(up1)
    conv6 = Conv2D(8 * n_filters, (k, k), padding=padding)(up2)
    conv6 = GroupNormalization(groups=group, axis=3, scale=False)(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(8 * n_filters, (k, k), padding=padding)(conv6)
    conv6 = GroupNormalization(groups=group, axis=3, scale=False)(conv6)
    conv6 = Activation('relu')(conv6)

    up3 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv6), conv3])
    conv7 = Conv2D(4 * n_filters, (k, k), padding=padding)(up3)
    conv7 = GroupNormalization(groups=group, axis=3, scale=False)(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(4 * n_filters, (k, k), padding=padding)(conv7)
    conv7 = GroupNormalization(groups=group, axis=3, scale=False)(conv7)
    conv7 = Activation('relu')(conv7)

    up4 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv7), conv2])
    conv8 = Conv2D(2 * n_filters, (k, k), padding=padding)(up4)
    conv8 = GroupNormalization(groups=group, axis=3, scale=False)(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(2 * n_filters, (k, k), padding=padding)(conv8)
    conv8 = GroupNormalization(groups=group, axis=3, scale=False)(conv8)
    conv8 = Activation('relu')(conv8)

    up5 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv8), conv1])
    conv9 = Conv2D(n_filters, (k, k), padding=padding)(up5)
    conv9 = GroupNormalization(groups=group, axis=3, scale=False)(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(n_filters, (k, k), padding=padding)(conv9)
    conv9 = GroupNormalization(groups=group, axis=3, scale=False)(conv9)
    conv9 = Activation('relu')(conv9)

    outputs = Conv2D(out_ch, (1, 1), padding=padding, activation='sigmoid')(conv9)

    g = Model(inputs, outputs, name=name)
    tensorboard.set_model(g)

    return g

def extractor(x, n_filters):

    feature1 = Conv2D(n_filters, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    feature2 = Conv2D(n_filters, kernel_size=(3,3), strides=(1,1), padding='same')(feature1)
    feature3 = Conv2D(n_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(feature2)
    feature4 = Conv2D(n_filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(feature3)

    feature = Concatenate(axis=3)([feature1, feature2, feature3, feature4])
    return feature

def discriminator(img_size, n_filters, init_lr, tensorboard, name='d'):
    """
    discriminator network (pixel GAN)
    """

    # set image specifics
    k = 3  # kernel size
    img_ch = 3  # image channels
    out_ch = 1  # output channel
    img_height, img_width = img_size[0], img_size[1]

    inputs = Input((img_height, img_width, img_ch + out_ch))

    conv1 = extractor(inputs, n_filters)
    conv1 = LeakyReLU(0.2)(conv1)
    conv2 = extractor(conv1, n_filters)
    conv2 = LeakyReLU(0.2)(conv2)
    conv3 = extractor(conv2, n_filters)
    conv3 = LeakyReLU(0.2)(conv3)
    outputs = Concatenate(axis=3)([inputs, conv1, conv2, conv3])
    outputs = Conv2D(out_ch, kernel_size=(1, 1), padding="same")(outputs)
    outputs = Activation('sigmoid')(outputs)

    d = Model(inputs, outputs, name=name)

    def d_loss(y_true, y_pred):
        L = objectives.binary_crossentropy(K.batch_flatten(y_true),
                                           K.batch_flatten(y_pred))
        return L

    def dice_coef(y_true, y_pred, smooth=1):
        """
        Dice = (2*|X & Y|)/ (|X|+ |Y|)
             =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
        ref: https://arxiv.org/pdf/1606.04797v1.pdf
        """
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)

    def dice_coef_loss(y_true, y_pred):
        return 1 - dice_coef(y_true, y_pred)

    d.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss=d_loss, metrics=['accuracy'])
    # d.compile(optimizer=RMSprop(lr=init_lr), loss=d_loss, metrics=['accuracy'])
    tensorboard.set_model(d)

    return d, d.layers[-1].output_shape[1:]


def GAN(g, d, img_size, n_filters_g, n_filters_d, alpha_recip, init_lr, tensorboard, name='gan'):
    """
    GAN (that binds generator and discriminator)
    """
    img_h, img_w = img_size[0], img_size[1]

    img_ch = 3
    seg_ch = 1

    fundus = Input((img_h, img_w, img_ch))
    vessel = Input((img_h, img_w, seg_ch))

    fake_vessel = g(fundus)
    fake_pair = Concatenate(axis=3)([fundus, fake_vessel])

    gan = Model([fundus, vessel], d(fake_pair), name=name)

    def dice_coef(y_true, y_pred, smooth=1):
        """
        Dice = (2*|X & Y|)/ (|X|+ |Y|)
             =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
        ref: https://arxiv.org/pdf/1606.04797v1.pdf
        """
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)

    def dice_coef_loss(y_true, y_pred):
        return 1-dice_coef(y_true, y_pred)

    def weighted_bce_loss(y_true, y_pred, weight):
        # avoiding overflow
        epsilon = 1e-7
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        logit_y_pred = K.log(y_pred / (1. - y_pred))

        # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
        loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * (
                    K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
        return K.sum(loss) / K.sum(weight)

    def weighted_dice_loss(y_true, y_pred, weight):
        smooth = 1.
        w, m1, m2 = weight * weight, y_true, y_pred
        intersection = (m1 * m2)
        score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
        loss = 1. - K.sum(score)
        return loss

    def weighted_bce_dice_loss(y_true, y_pred):
        y_true = K.cast(y_true, 'float32')
        y_pred = K.cast(y_pred, 'float32')
        # if we want to get same size of output, kernel size must be odd number
        # averaged_mask = K.pool2d(y_true, pool_size=(11, 11), strides=(1, 1), padding='same', pool_mode='avg')

        # while imageGAN training, use this averaged_mask
        averaged_mask = y_true

        border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
        weight = K.ones_like(averaged_mask)
        w0 = K.sum(weight)
        weight += border * 2
        w1 = K.sum(weight)
        weight *= (w0 / w1)
        loss = weighted_bce_loss(y_true, y_pred, weight) + weighted_dice_loss(y_true, y_pred, weight)
        return loss

    def gan_loss(y_true, y_pred):
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)

        # L_adv = weighted_bce_dice_loss(y_true, y_pred)
        L_adv = objectives.binary_crossentropy(y_true_flat, y_pred_flat)
        # L_adv = objectives.mean_squared_error(y_true_flat, y_pred_flat)

        vessel_flat = K.batch_flatten(vessel)
        fake_vessel_flat = K.batch_flatten(fake_vessel)
        # L_seg = weighted_bce_dice_loss(vessel, fake_vessel)
        L_seg = objectives.binary_crossentropy(vessel_flat, fake_vessel_flat)
        # L_seg = objectives.mean_absolute_error(vessel_flat, fake_vessel_flat)

        # L_adv = dice_coef_loss(y_true_flat, y_pred_flat)
        # L_seg = dice_coef_loss(vessel_flat, fake_vessel_flat)

        # L_adv = boundary_loss(y_true, y_pred)
        # L_seg = boundary_loss(vessel, fake_vessel)

        # L_adv = wasserstein_loss(y_true, y_pred)
        # L_seg = wasserstein_loss(vessel, fake_vessel)

        return L_adv + alpha_recip*L_seg

    gan.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss=gan_loss, metrics=[dice_coef])
    # gan.compile(optimizer=RMSprop(lr=init_lr), loss=gan_loss, metrics=['accuracy'])
    tensorboard.set_model(gan)

    return gan

