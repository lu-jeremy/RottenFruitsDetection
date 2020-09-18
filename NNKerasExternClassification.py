<<<<<<< HEAD
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

model = load_model('NNKerasModel')

test_datagen = ImageDataGenerator(rescale=1./255)

external_set = test_datagen.flow_from_directory(r'C:\Users\bluet\PycharmProjects\RottenFruits\src\extern_images',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')
model.predict(external_set)
=======
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

model = load_model('NNKerasModel')

test_datagen = ImageDataGenerator(rescale=1./255)

external_set = test_datagen.flow_from_directory(r'C:\Users\bluet\PycharmProjects\RottenFruits\src\extern_images',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')
model.predict(external_set)
>>>>>>> 4e22822301509d4267a16e7ea3b472f9d8a325b9
