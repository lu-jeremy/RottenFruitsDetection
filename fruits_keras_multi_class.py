<<<<<<< HEAD
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

classifier = Sequential()
#First Convolutional layer
classifier.add(Convolution2D(filters = 56,kernel_size = (3,3), activation = 'relu', input_shape = (64,64,3)))
classifier.add(MaxPooling2D(pool_size = (2,2)))
#second Convolutional layer
classifier.add(Convolution2D(32,(3,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
#Flattening
classifier.add(Flatten())
#Hidden Layer
classifier.add(Dense(units = 128, activation = 'relu'))
#Output Layer
classifier.add(Dense(units = 6 , activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy','accuracy'])

train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.1, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('dataset/train', target_size=(64, 64), batch_size=32,class_mode='categorical')
test_set = test_datagen.flow_from_directory('dataset/test', target_size=(64, 64), batch_size=32, class_mode='categorical')

classifier.summary()
classifier.fit_generator(training_set, samples_per_epoch=2000, nb_epoch=15, validation_data=test_set, nb_val_samples=200)
classifier.predict_classes(test_set)


=======
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

classifier = Sequential()
#First Convolutional layer
classifier.add(Convolution2D(filters = 56,kernel_size = (3,3), activation = 'relu', input_shape = (64,64,3)))
classifier.add(MaxPooling2D(pool_size = (2,2)))
#second Convolutional layer
classifier.add(Convolution2D(32,(3,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
#Flattening
classifier.add(Flatten())
#Hidden Layer
classifier.add(Dense(units = 128, activation = 'relu'))
#Output Layer
classifier.add(Dense(units = 6 , activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy','accuracy'])

train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.1, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('dataset/train', target_size=(64, 64), batch_size=32,class_mode='categorical')
test_set = test_datagen.flow_from_directory('dataset/test', target_size=(64, 64), batch_size=32, class_mode='categorical')

classifier.summary()
classifier.fit_generator(training_set, samples_per_epoch=2000, nb_epoch=15, validation_data=test_set, nb_val_samples=200)
classifier.predict_classes(test_set)


>>>>>>> 4e22822301509d4267a16e7ea3b472f9d8a325b9
