from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation, Convolution2D
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras import backend as K


TF_WEIGHTS_PATH = 'vgg19_weights_tf_dim_ordering_tf_kernels.h5'
TF_WEIGHTS_PATH_NO_TOP = 'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
#сверху пути до моделей, но у нас без верхнего слоя, те без перцептрона

#тут описание каждого слоя нейронки с ее последовательностями
def VGG_19(include_top=True, weights='imagenet', input_tensor=None):
    input_shape = (None, None, 3)
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input) #сверточные слои
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x) #слой дискретизации
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(1000, activation='softmax', name='predictions')(x)

    model = Model(img_input, x)
    if include_top: #для верхнего слоя, но тк у нас его нет то мы идем сразу в 56 строку
        weights_path = TF_WEIGHTS_PATH
    else:
        weights_path = TF_WEIGHTS_PATH_NO_TOP
    model.load_weights(weights_path) #подгружаем веса из папки, а именно фаил vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5
    return model
from tensorflow.python.client import device_lib

if __name__=="__main__":
    print(device_lib.list_local_devices())
    # model = VGG_19(include_top=False, weights='imagenet')
    # img=image.load_img('test.png', target_size=(224, 224))
    # x=image.img_to_array(img)
    # x=np.expand_dims(x, axis=0)
    # x=preprocess_input(x)
    # features=model.predict(x)
    # print(distance.cosine(features.flatten(),features.flatten()))