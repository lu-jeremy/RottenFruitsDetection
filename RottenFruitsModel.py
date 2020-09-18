# Binary classification of fresh and rotten frutis

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator

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


train_data = train_datagen.flow_from_directory(directory=r'BinaryClassification\train',
                                               target_size=(64, 64),
                                               batch_size=32,
                                               class_mode='categorical'
                                               )
test_data = test_datagen.flow_from_directory(directory=r'BinaryClassification\test',
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

    classifier.add(Dense(units=2, activation='softmax'))

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return classifier


classifier = upload_model()

classifier.summary()

classifier.fit_generator(train_data,
                         samples_per_epoch=2000,
                         nb_epoch=15,
                         validation_data=test_data,
                         nb_val_samples=200)

results = classifier.predict(test_data)

classifier.save('RottenFruitsModel')  # change save
