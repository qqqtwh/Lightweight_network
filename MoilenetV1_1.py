from tensorflow.keras.layers import *
from tensorflow.keras import Model

def conv_block(inputs, filters, kernel, strides):
    '''
    :param inputs: 输入的 tensor
    :param filters: 卷积核数量
    :param kernel:  卷积核大小
    :param strides: 卷积步长
    :return:
    '''
    x = ZeroPadding2D(1)(inputs)
    x = Conv2D(filters, kernel, strides, padding='valid', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)
    return x

def dw_pw_block(inputs, dw_strides, pw_filters, name):
    '''
    :param inputs:      输入的tensor
    :param dw_strides:  深度卷积的步长
    :param pw_filters:  逐点卷积的卷积核数量
    :param name:
    :return:
    '''
    x = ZeroPadding2D(1)(inputs)
    # dw
    x = DepthwiseConv2D((3, 3), dw_strides, padding='valid', use_bias=False, name=name)(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)
    # pw
    x = Conv2D(pw_filters, (1, 1), 1, padding='valid', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)
    return x

def build_MobilenetV1_1(n_classes=1000,height=416, width=416):
    img_input = Input(shape=(height, width, 3))
    # block1:con1 + dw_pw_1
    # 416,416,3 -- 208,208,32 -- 208,208,64
    x = conv_block(img_input, 32, (3, 3), (2, 2))
    x = dw_pw_block(x, 1, 64, 'dw_pw_1')

    # block2:dw_pw_2
    # 208,208,64 -- 104,104,128
    x = dw_pw_block(x, 2, 128, 'dw_pw_2_1')
    x = dw_pw_block(x, 1, 128, 'dw_pw_2_2')

    # block3:dw_pw_3
    # 104,104,128 -- 52,52,256
    x = dw_pw_block(x, 2, 256, 'dw_pw_3_1')
    x = dw_pw_block(x, 1, 256, 'dw_pw_3_2')

    # block4:dw_pw_4
    # 52,52,256 -- 26,26,512
    x = dw_pw_block(x, 2, 512, 'dw_pw_4_1')
    for i in range(5):
        x = dw_pw_block(x, 1, 512, 'dw_pw_4_' + str(i + 2))

    # block5:dw_pw_5
    # 26,26,512 -- 13,13,1024
    x = dw_pw_block(x, 2, 1024, 'dw_pw_5_1')
    x = dw_pw_block(x, 1, 1024, 'dw_pw_5_2')

    # 13,13,1024 -- 1,1,1024
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1,1,1024))(x)
    x = Dropout(0.5)(x)
    # 1,1,1024 -- 1,1,n_classes
    x = Conv2D(n_classes,1,padding='same')(x)
    x = Softmax()(x)
    x = Reshape((n_classes,))(x)
    return Model(img_input,x)

if __name__ == '__main__':
    build_MobilenetV1_1(10).summary()