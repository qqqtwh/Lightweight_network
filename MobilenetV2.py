from tensorflow.keras.layers import *
from tensorflow.keras import Model

# 倒残差结构
def inv_res_block(inputs,filters,strides,expansion,is_add,block_id=1,rate=1):
    '''
    :param inputs: 输入的 tensor
    :param filters: 深度可分离卷积卷积核数量
    :param strides: 深度可分离卷积步长
    :param expansion:  倒残差通道扩张的倍数
    :param is_add: 是否进行残差相加
    :param rate: 空洞卷积扩张率
    :param block_id:
    :return:
    '''
    in_channels = inputs.shape[-1]
    x = inputs
    # 如果是第0个倒残差块，不进行通道扩张
    if block_id:
        x = Conv2D(in_channels*expansion,kernel_size=1,padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU(max_value=6)(x)
    # 深度可分离卷积提取特征
    x = DepthwiseConv2D(kernel_size=3,strides=strides,padding='same',use_bias=False,dilation_rate=(rate,rate))(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)
    # 使用 1x1 卷积进行通道缩小
    x = Conv2D(filters,kernel_size=1,padding='same',use_bias=False)(x)
    x = BatchNormalization()(x)

    if is_add:
        return Add()([inputs,x])

    return x
# img_input=size,residual_1 = size/4,x = size/8
def build_MobilenetV2(n_classes=1000,height=416, width=416):
    img_input = Input(shape=(height, width, 3))
    # 416,416,3 -- 208,208,32
    x = Conv2D(32,3,2,padding='same',use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)

    # 208,208,32 -- 208,208,16;首个倒残差块内部不进行通道先扩张后缩小
    x = inv_res_block(x,filters=16,strides=1,expansion=1,is_add=False,block_id=0)

    # 208,208,16 -- 104,104,24
    x = inv_res_block(x,filters=24,strides=2,expansion=6,is_add=False)
    x = inv_res_block(x,filters=24,strides=1,expansion=6,is_add=True)
    residual_1 = x

    # 104,104,24 -- 52,52,32
    x = inv_res_block(x, filters=32, strides=2, expansion=6, is_add=False)
    x = inv_res_block(x, filters=32, strides=1, expansion=6, is_add=True)
    x = inv_res_block(x, filters=32, strides=1, expansion=6, is_add=True)

    # 52,52,32 -- 52,52,64
    x = inv_res_block(x, filters=64, strides=1, expansion=6, is_add=False)
    x = inv_res_block(x, filters=64, strides=1, expansion=6, is_add=True,rate=2)
    x = inv_res_block(x, filters=64, strides=1, expansion=6, is_add=True,rate=2)
    x = inv_res_block(x, filters=64, strides=1, expansion=6, is_add=True,rate=2)

    # 52,52,64 -- 52,52,96
    x = inv_res_block(x, filters=96, strides=1, expansion=6, is_add=False, rate=2)
    x = inv_res_block(x, filters=96, strides=1, expansion=6, is_add=True, rate=2)
    x = inv_res_block(x, filters=96, strides=1, expansion=6, is_add=True, rate=2)

    # 52,52,96 -- 52,52,160
    x = inv_res_block(x, filters=160, strides=1, expansion=6, is_add=False, rate=2)
    x = inv_res_block(x, filters=160, strides=1, expansion=6, is_add=True, rate=4)
    x = inv_res_block(x, filters=160, strides=1, expansion=6, is_add=True, rate=4)

    # 52,52,160 -- 52,52,320
    x = inv_res_block(x, filters=320, strides=1, expansion=6, is_add=False, rate=4)

    # 52,52,320 -- 52,52,1280
    x = Conv2D(1280,kernel_size=1,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)
    # 52,52,1280 -- 1280
    x = GlobalAveragePooling2D()(x)
    # 1280 -- n_classes
    x = Dense(n_classes,activation='softmax')(x)

    return Model(img_input,x)


if __name__ == '__main__':
    build_MobilenetV2(10).summary()