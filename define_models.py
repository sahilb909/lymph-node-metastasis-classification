def createResNet():
    from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Add, AveragePooling2D, Flatten, Dense
    from keras.models import Model
    from keras.regularizers import l2

    def convolutional_block(input_tensor, filters, kernel_size, strides=(2, 2)):
        filters1, filters2, filters3 = filters

        x = Conv2D(filters1, (1, 1), strides=strides, kernel_regularizer=l2(0.01))(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)

        shortcut = Conv2D(filters3, (1, 1), strides=strides, kernel_regularizer=l2(0.01))(input_tensor)
        shortcut = BatchNormalization()(shortcut)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x

    def identity_block(input_tensor, filters, kernel_size):
        filters1, filters2, filters3 = filters

        x = Conv2D(filters1, (1, 1), kernel_regularizer=l2(0.01))(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)

        x = Add()([x, input_tensor])
        x = Activation('relu')(x)
        return x

    input_layer = Input(shape=(96, 96, 3))

    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', kernel_regularizer=l2(0.01))(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = convolutional_block(x, filters=[64, 64, 256], kernel_size=3, strides=(1, 1))
    x = identity_block(x, filters=[64, 64, 256], kernel_size=3)
    x = identity_block(x, filters=[64, 64, 256], kernel_size=3)

    x = convolutional_block(x, filters=[128, 128, 512], kernel_size=3)
    x = identity_block(x, filters=[128, 128, 512], kernel_size=3)
    x = identity_block(x, filters=[128, 128, 512], kernel_size=3)
    x = identity_block(x, filters=[128, 128, 512], kernel_size=3)

    x = AveragePooling2D(pool_size=(7, 7))(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)
    output_layer = Dense(2, activation='sigmoid', kernel_regularizer=l2(0.01))(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

def createVGG16():
    from keras.applications import VGG16
    from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Add, AveragePooling2D, Flatten, Dense
    from keras.models import Model, Sequential
    from keras.regularizers import l2

    vgg = VGG16(input_shape=(96, 96, 3), weights='imagenet', include_top=False, pooling='avg')
    vgg.trainable = False

    model = Sequential()
    model.add(vgg)
    model.add(Dense(512, activation=('relu')))
    model.add(Dense(256, activation=('relu')))
    model.add(Dense(2, activation=('sigmoid')))

    return model