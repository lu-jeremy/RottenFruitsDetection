from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
# import tensorflow as tf

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.1,
                                   zoom_range=0.2,
                                   featurewise_center=False,
                                   featurewise_std_normalization=False,
                                   rotation_range=10,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)


train_data = train_datagen.flow_from_directory(directory=r'dataset\train',
                                               target_size=(64, 64),
                                               batch_size=32,
                                               class_mode='categorical'
                                               )
test_data = test_datagen.flow_from_directory(directory=r'dataset\test',
                                             target_size=(64, 64),
                                             batch_size=32,
                                             class_mode='categorical'
                                             )
def upload_model():
    classifier = Sequential()

    classifier.add(Convolution2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))

    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    classifier.add(Convolution2D(64, (3, 3), activation='relu'))

    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    classifier.add(Convolution2D(128, (3, 3), activation='relu'))

    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    classifier.add(Dropout(0.25))

    classifier.add(Flatten())

    classifier.add(Dense(units=128, activation='relu'))

    classifier.add(Dropout(rate=0.5))

    classifier.add(Dense(units=6, activation='softmax'))

    # padding = 'valid'
    # img_input = keras.layers.Input(shape=(64, 64, 3))
    #
    # # START MODEL
    # conv_1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding=padding, activation='relu', name='conv_1')(
    #     img_input)
    # maxpool_1 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv_1)
    # x = tf.keras.layers.BatchNormalization()(maxpool_1)
    #
    # # FEAT-EX1
    # conv_2a = tf.keras.layers.Conv2D(96, (1, 1), strides=(1, 1), activation='relu', padding=padding,
    #                                  name='conv_2a')(x)
    # conv_2b = tf.keras.layers.Conv2D(208, (3, 3), strides=(1, 1), activation='relu', padding=padding,
    #                                  name='conv_2b')(conv_2a)
    # maxpool_2a = keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding=padding, name='maxpool_2a')(x)
    # conv_2c = tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), name='conv_2c')(maxpool_2a)
    # concat_1 = tf.keras.layers.concatenate(inputs=[conv_2b, conv_2c], axis=3, name='concat2')
    # maxpool_2b = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding=padding, name='maxpool_2b')(concat_1)
    #
    # # FEAT-EX2
    # conv_3a = tf.keras.layers.Conv2D(96, (1, 1), strides=(1, 1), activation='relu', padding=padding,
    #                                  name='conv_3a')(maxpool_2b)
    # conv_3b = tf.keras.layers.Conv2D(208, (3, 3), strides=(1, 1), activation='relu', padding=padding,
    #                                  name='conv_3b')(conv_3a)
    # maxpool_3a = keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding=padding, name='maxpool_3a')(maxpool_2b)
    # conv_3c = tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), name='conv_3c')(maxpool_3a)
    # concat_3 = tf.keras.layers.concatenate(inputs=[conv_3b, conv_3c], axis=3, name='concat3')
    # maxpool_3b = keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding=padding, name='maxpool_3b')(concat_3)
    #
    # # FINAL LAYERS
    # net = tf.keras.layers.Flatten()(maxpool_3b)
    # net = tf.keras.layers.Dense(6, activation='softmax', name='predictions')(net)
    #
    # # Create model.
    # model = tf.keras.Model(img_input, net, name='deXpression')
    return classifier

    # return classifier


classifier = upload_model()

optimizer = Adam(lr=1e-3)

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy', 'accuracy'])

classifier.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['categorical_accuracy', 'accuracy'])

classifier.summary()

classifier.fit_generator(train_data,
                         samples_per_epoch=2000,
                         nb_epoch=30,
                         validation_data=test_data,
                         nb_val_samples=200)

results = classifier.predict(test_data)

print(results)

classifier.save('RottenFruitsModel')

