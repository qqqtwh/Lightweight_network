from tensorflow.keras.layers import *
from tensorflow.keras import Model


def conv_bn_relu(inputs,filters,ker_sizes,strides=1,name=None):
    x = Conv2D(filters, kernel_size=ker_sizes, strides=strides, use_bias=False, name=name)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x

def separ_residual(inputs,filters,relu_first=False,name=None):
    # inputs shape: h,w,c
    # h,w,c -- h/2,w/2,filters

    res = Conv2D(filters, 1, 2, padding='same', use_bias=False,name=name)(inputs)
    res = BatchNormalization()(res)

    x = inputs
    if relu_first:
        x = ReLU()(x)
    x = SeparableConv2D(filters=filters, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = SeparableConv2D(filters=filters, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), (2, 2), padding='same')(x)
    x = Add()([x, res])
    return x

def build_Xception(n_classes=1000,height=299, width=299):
    img_input = Input(shape=(height, width, 3))

    # 1.Entry flow
    # Entry_block1: 299,299,3 -- 149,149,64
    x = conv_bn_relu(img_input,32,3,2,name='Entry_block1_1')
    x = conv_bn_relu(x,64,3,1,name='Entry_block1_2')
    # Entry_block2: 149,149,64 -- 75,75,128
    x = separ_residual(x,128,name='Entry_block2')
    # Entry_block3: 75,75,128 -- 38,38,256
    x = separ_residual(x, 256,relu_first=True,name='Entry_block3')
    # Entry_block4: 38,38,256 -- 19,19,728
    x = separ_residual(x, 728,relu_first=True,name='Entry_block4')

    # 2.Middle flow: 19,19,728 -- 19,19,728
    for i in range(1,9):
        res = x
        x = ReLU(name='Middle_block'+str(i))(x)
        x = SeparableConv2D(728,(3,3),padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = SeparableConv2D(728, (3, 3), padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = SeparableConv2D(728, (3, 3), padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)

        x = Add()([x,res])

    # 3.Exit flow
    # Exit_block1: 19,19,728 --10,10,1024
    res = Conv2D(1024, 1, 2, padding='same', use_bias=False, name='Exit_block1')(x)
    res = BatchNormalization()(res)
    x = ReLU()(x)
    x = SeparableConv2D(filters=728, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = SeparableConv2D(filters=1024, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), (2, 2), padding='same')(x)
    x = Add()([x, res])

    # Exit_block2: 10,10,1024 -- 10,10,1536 --10,10,2048
    x = SeparableConv2D(filters=1536, kernel_size=(3, 3), padding='same', use_bias=False,name='Exit_block2')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = SeparableConv2D(filters=2048, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # 10,10,2048 -- 2048
    x = GlobalAveragePooling2D()(x)
    x = Dense(n_classes,activation='softmax')(x)

    return Model(img_input,x)


if __name__ == '__main__':
    build_Xception(1000).summary()