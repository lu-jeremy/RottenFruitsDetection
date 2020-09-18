from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

model = load_model('NNKerasModel')

test_datagen = ImageDataGenerator(rescale=1./255)

external_set = test_datagen.flow_from_directory(r'C:\Users\bluet\PycharmProjects\RottenFruits\src\extern_images',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')
model.predict(external_set)
