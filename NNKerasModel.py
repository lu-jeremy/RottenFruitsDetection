<<<<<<< HEAD
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

from imutils import paths 
import sys
import pandas as pd

# generating data

def preprocessing(folder_name, mode):
    """
    Generates csv file
    :param folder_name: fruit directory
    """
    if mode == 0:
        training_file = open('train.csv', 'w')
        training_file.write('ID,class_id,fruit_id\n')

        for image_path in paths.list_images(folder_name):
            indiv_image = image_path.split('\\')[-1]
            fruit_name = image_path.split('\\')[-2]
            class_name = image_path.split('\\')[-3]

            if 'apple' in fruit_name:
                fruit_id = '1'
            elif 'banana' in fruit_name:
                fruit_id = '0'

            if 'rotten' in class_name:
                class_id = '1'
            elif 'fresh' in class_name:
                class_id = '0'

            training_file.write(indiv_image + ',' + class_id + ',' + fruit_id + '\n')

        training_file.close()

    elif mode == 1:
        test_file = open('test.csv', 'w')
        test_file.write('ID\n')

        for image_path in paths.list_images(folder_name):
            indiv_image = image_path.split('\\')[-1]

            test_file.write(indiv_image + '\n')

        test_file.close()

# preprocessing(r'C:\Users\bluet\PycharmProjects\RottenFruits\src\keras_dataset\train', 0)
preprocessing(r'C:\Users\bluet\PycharmProjects\RottenFruits\src\keras_dataset\test', 1)

training_set = pd.read_csv('train.csv')

training_images = list(training_set['ID'])
training_class = list(training_set['class_id'])
training_fruit = list(training_set['fruit_id'])

training_set = pd.DataFrame({'images': training_images, 'class_name': training_class, 'fruit_name': training_fruit})

training_set.class_name = training_set.class_name.astype(str)
training_set.fruit_name = training_set.fruit_name.astype(str)

training_set['new_class'] = training_set['class_name'] + training_set['fruit_name']

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.1,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

training_set = train_datagen.flow_from_dataframe(dataframe=training_set,
                                                 directory=r'C:\Users\bluet\PycharmProjects\RottenFruits\src\fruit_dataset\train',
                                                 x_col='images',
                                                 y_col='new_class',
                                                 class_mode='categorical',
                                                 target_size=(64, 64),
                                                 batch_size=32)

# test_set = test_datagen.flow_from_directory(r'C:\Users\bluet\PycharmProjects\RottenFruits\src\fruit_dataset\test',
#                                             target_size=(64, 64),
#                                             batch_size=32,
#                                             class_mode='binary')

# generating model

model = Sequential()
model.add(Convolution2D(filters=56,
                        kernel_size=(3, 3),
                        input_shape=(64, 64, 3),
                        activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3, activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(units=64, activation='relu'))

model.add(Dense(units=4, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy','accuracy'])

model.summary()

# model.fit_generator(testing_set,
#                     samples_per_epoch=2000,
#                     nb_epoch=15,
#                     validation_data=test_set,
#                     nb_val_samples=200)

model.fit_generator(training_set,
                    epochs=50,
                    steps_per_epoch=50)

=======
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

from imutils import paths 
import sys
import pandas as pd

# generating data

def preprocessing(folder_name, mode):
    """
    Generates csv file
    :param folder_name: fruit directory
    """
    if mode == 0:
        training_file = open('train.csv', 'w')
        training_file.write('ID,class_id,fruit_id\n')

        for image_path in paths.list_images(folder_name):
            indiv_image = image_path.split('\\')[-1]
            fruit_name = image_path.split('\\')[-2]
            class_name = image_path.split('\\')[-3]

            if 'apple' in fruit_name:
                fruit_id = '1'
            elif 'banana' in fruit_name:
                fruit_id = '0'

            if 'rotten' in class_name:
                class_id = '1'
            elif 'fresh' in class_name:
                class_id = '0'

            training_file.write(indiv_image + ',' + class_id + ',' + fruit_id + '\n')

        training_file.close()

    elif mode == 1:
        test_file = open('test.csv', 'w')
        test_file.write('ID\n')

        for image_path in paths.list_images(folder_name):
            indiv_image = image_path.split('\\')[-1]

            test_file.write(indiv_image + '\n')

        test_file.close()

# preprocessing(r'C:\Users\bluet\PycharmProjects\RottenFruits\src\keras_dataset\train', 0)
preprocessing(r'C:\Users\bluet\PycharmProjects\RottenFruits\src\keras_dataset\test', 1)

training_set = pd.read_csv('train.csv')

training_images = list(training_set['ID'])
training_class = list(training_set['class_id'])
training_fruit = list(training_set['fruit_id'])

training_set = pd.DataFrame({'images': training_images, 'class_name': training_class, 'fruit_name': training_fruit})

training_set.class_name = training_set.class_name.astype(str)
training_set.fruit_name = training_set.fruit_name.astype(str)

training_set['new_class'] = training_set['class_name'] + training_set['fruit_name']

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.1,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

training_set = train_datagen.flow_from_dataframe(dataframe=training_set,
                                                 directory=r'C:\Users\bluet\PycharmProjects\RottenFruits\src\fruit_dataset\train',
                                                 x_col='images',
                                                 y_col='new_class',
                                                 class_mode='categorical',
                                                 target_size=(64, 64),
                                                 batch_size=32)

# test_set = test_datagen.flow_from_directory(r'C:\Users\bluet\PycharmProjects\RottenFruits\src\fruit_dataset\test',
#                                             target_size=(64, 64),
#                                             batch_size=32,
#                                             class_mode='binary')

# generating model

model = Sequential()
model.add(Convolution2D(filters=56,
                        kernel_size=(3, 3),
                        input_shape=(64, 64, 3),
                        activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3, activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(units=64, activation='relu'))

model.add(Dense(units=4, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy','accuracy'])

model.summary()

# model.fit_generator(testing_set,
#                     samples_per_epoch=2000,
#                     nb_epoch=15,
#                     validation_data=test_set,
#                     nb_val_samples=200)

model.fit_generator(training_set,
                    epochs=50,
                    steps_per_epoch=50)

>>>>>>> 4e22822301509d4267a16e7ea3b472f9d8a325b9
model.save('NNKerasModel')