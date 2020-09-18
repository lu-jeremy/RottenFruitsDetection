<<<<<<< HEAD
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

external_datagen = ImageDataGenerator(rescale=1. / 255)

external_data = external_datagen.flow_from_directory(directory=r'BinaryClassification\external',
                                             target_size=(64, 64),
                                             batch_size=32,
                                             class_mode='categorical'
                                             )

model = load_model('RottenFruitsModel')

model.summary()

results = model.predict(external_data)

# print(results)

count = 0

# print(external_data.classes)

for index, prediction in enumerate(results):
    classification = np.argmax(prediction)

    if classification == external_data.classes[index]:
        count += 1
        print('correct', classification)
    else:
        print('wrong', classification)

    if classification == 0:
        print('fresh\n')
    elif classification == 1:
        print('rotten\n')

accuracy = (count / len(external_data.classes)) * 100

print(accuracy, '%')


=======
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


>>>>>>> 4e22822301509d4267a16e7ea3b472f9d8a325b9
