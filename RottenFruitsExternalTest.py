from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

external_datagen = ImageDataGenerator(rescale=1. / 255)

external_data = external_datagen.flow_from_directory(directory=r'extern_images',
                                             target_size=(64, 64),
                                             batch_size=32,
                                             class_mode='categorical'
                                             )

model = load_model('RottenFruitsModel')

model.summary()

results = model.predict(external_data)

print(results)

print(external_data.classes)

count = 0

for index, prediction in enumerate(results):
    classification = np.argmax(prediction)

    if classification == external_data.classes[index]:
        print('correct', classification, external_data.classes[index])
        count += 1

    if classification == 0:
        print('freshapples')
    elif classification == 1:
        print('freshbanana')
    elif classification == 2:
        print('freshoranges')
    elif classification == 3:
        print('rottenapples')
    elif classification == 4:
        print('rottenbanana')
    elif classification == 5:
        print('rottenoranges')

accuracy = (count / 27) * 100

print(accuracy)


