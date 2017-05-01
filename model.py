from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Conv3D,ZeroPadding3D,Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, History

def model_FCN9_base():
    #create model similar to BNafterReLU, change to HE initialization
    model = Sequential()

    model.add(ZeroPadding3D((1,1,1),input_shape=(24,24,24,1)))
    model.add(Conv3D(32,(3,3,3), use_bias = False, kernel_initializer = 'he_normal'))
    model.add(BatchNormalization()) #would perform the correct convolutional batch normalization
    model.add(Activation('relu'))

    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(32,(3,3,3), use_bias = False, kernel_initializer = 'he_normal'))
    model.add(BatchNormalization()) #would perform the correct convolutional batch normalization
    model.add(Activation('relu'))


    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(32,(3,3,3), use_bias = False, kernel_initializer = 'he_normal'))
    model.add(BatchNormalization()) #would perform the correct convolutional batch normalization
    model.add(Activation('relu'))

    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(64,(3,3,3), use_bias = False, kernel_initializer = 'he_normal'))
    model.add(BatchNormalization()) #would perform the correct convolutional batch normalization
    model.add(Activation('relu'))

    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(64,(3,3,3), use_bias = False, kernel_initializer = 'he_normal'))
    model.add(BatchNormalization()) #would perform the correct convolutional batch normalization
    model.add(Activation('relu'))

    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(64,(3,3,3), use_bias = False, kernel_initializer = 'he_normal'))
    model.add(BatchNormalization()) #would perform the correct convolutional batch normalization
    model.add(Activation('relu'))

    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(32,(3,3,3), use_bias = False, kernel_initializer = 'he_normal'))
    model.add(BatchNormalization()) #would perform the correct convolutional batch normalization
    model.add(Activation('relu'))

    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(32,(3,3,3), use_bias = False, kernel_initializer = 'he_normal'))
    model.add(BatchNormalization()) #would perform the correct convolutional batch normalization
    model.add(Activation('relu'))

    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(32,(3,3,3), use_bias = False, kernel_initializer = 'he_normal'))
    model.add(BatchNormalization()) #would perform the correct convolutional batch normalization
    model.add(Activation('relu'))

    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(1,(3,3,3), use_bias = False, kernel_initializer = 'he_normal'))

    return model
def model_FCN9_xavier():
    #create model
    model = Sequential()

    model.add(ZeroPadding3D((1,1,1),input_shape=(24,24,24,1)))
    model.add(Conv3D(32,(3,3,3), use_bias = False, kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization()) #would perform the correct convolutional batch normalization
    model.add(Activation('relu'))

    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(32,(3,3,3), use_bias = False, kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization()) #would perform the correct convolutional batch normalization
    model.add(Activation('relu'))

    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(32,(3,3,3), use_bias = False, kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization()) #would perform the correct convolutional batch normalization
    model.add(Activation('relu'))

    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(64,(3,3,3), use_bias = False, kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization()) #would perform the correct convolutional batch normalization
    model.add(Activation('relu'))

    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(64,(3,3,3), use_bias = False, kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization()) #would perform the correct convolutional batch normalization
    model.add(Activation('relu'))

    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(64,(3,3,3), use_bias = False, kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization()) #would perform the correct convolutional batch normalization
    model.add(Activation('relu'))

    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(32,(3,3,3), use_bias = False, kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization()) #would perform the correct convolutional batch normalization
    model.add(Activation('relu'))

    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(32,(3,3,3), use_bias = False, kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization()) #would perform the correct convolutional batch normalization
    model.add(Activation('relu'))

    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(32,(3,3,3), use_bias = False, kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization()) #would perform the correct convolutional batch normalization
    model.add(Activation('relu'))

    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(1,(3,3,3), use_bias = False, kernel_initializer = 'glorot_normal'))

    return model
def model_FCN9_BNafterReLU():
    #create model similar to nobias
    model = Sequential()

    model.add(ZeroPadding3D((1,1,1),input_shape=(24,24,24,1)))
    model.add(Conv3D(32,(3,3,3), use_bias = False, kernel_initializer = 'he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization()) #would perform the correct convolutional batch normalization


    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(32,(3,3,3), use_bias = False, kernel_initializer = 'he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization()) #would perform the correct convolutional batch normalization


    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(32,(3,3,3), use_bias = False, kernel_initializer = 'he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(64,(3,3,3), use_bias = False, kernel_initializer = 'he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(64,(3,3,3), use_bias = False, kernel_initializer = 'he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(64,(3,3,3), use_bias = False, kernel_initializer = 'he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(32,(3,3,3), use_bias = False, kernel_initializer = 'he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(32,(3,3,3), use_bias = False, kernel_initializer = 'he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(32,(3,3,3), use_bias = False, kernel_initializer = 'he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(ZeroPadding3D((1,1,1)))
    model.add(Conv3D(1,(3,3,3), use_bias = False, kernel_initializer = 'glorot_normal'))

    return model

# 9 layers without zero-padding
def model_FCN9_24_4():
    model = Sequential()

    model.add(Conv3D(32,(3,3,3), use_bias = False, kernel_initializer = 'glorot_normal', input_shape=(24,24,24,1)))
    model.add(BatchNormalization()) #would perform the correct convolutional batch normalization
    model.add(Activation('relu'))

    model.add(Conv3D(32,(3,3,3), use_bias = False, kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization()) #would perform the correct convolutional batch normalization
    model.add(Activation('relu'))

    model.add(Conv3D(32,(3,3,3), use_bias = False, kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization()) #would perform the correct convolutional batch normalization
    model.add(Activation('relu'))

    model.add(Conv3D(64,(3,3,3), use_bias = False, kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization()) #would perform the correct convolutional batch normalization
    model.add(Activation('relu'))

    model.add(Conv3D(64,(3,3,3), use_bias = False, kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization()) #would perform the correct convolutional batch normalization
    model.add(Activation('relu'))

    model.add(Conv3D(64,(3,3,3), use_bias = False, kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization()) #would perform the correct convolutional batch normalization
    model.add(Activation('relu'))

    model.add(Conv3D(128,(3,3,3), use_bias = False, kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization()) #would perform the correct convolutional batch normalization
    model.add(Activation('relu'))

    model.add(Conv3D(128,(3,3,3), use_bias = False, kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization()) #would perform the correct convolutional batch normalization
    model.add(Activation('relu'))

    model.add(Conv3D(128,(3,3,3), use_bias = False, kernel_initializer = 'glorot_normal'))
    model.add(BatchNormalization()) #would perform the correct convolutional batch normalization
    model.add(Activation('relu'))

    model.add(Conv3D(1,(3,3,3), use_bias = False, kernel_initializer = 'glorot_normal'))

    return model
