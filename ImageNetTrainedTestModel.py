# from keras.applications.resnet50 import ResNet50
# from keras.preprocessing import image
# from keras.applications.resnet50 import preprocess_input, decode_predictions
# import numpy as np
#
# model = ResNet50(weights='imagenet')
#
# img_path = r'dataset\train\freshapples\rotated_by_15_Screen Shot 2018-06-08 at 4.59.36 PM.png'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
#
# preds = model.predict(x)
# # decode the results into a list of tuples (class, description, probability)
# # (one such list for each sample in the batch)
# print('Predicted:', decode_predictions(preds, top=3)[0])

# from keras.applications.vgg19 import VGG19
# from keras.preprocessing import image
# from keras.applications.vgg19 import preprocess_input
# from keras.models import Model
# import numpy as np
#
# base_model = VGG19(weights='imagenet')
# model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
#
# img_path = r'dataset\train\freshapples\rotated_by_15_Screen Shot 2018-06-08 at 4.59.36 PM.png'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
#
# block4_pool_features = model.predict(x)

from keras.applications import MobileNetV2
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D

#Load the VGG model
vgg_conv = MobileNetV2(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)

from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

# Create the model
model = models.Sequential()

# Add the vgg convolutional base model
model.add(vgg_conv)

# Add new layers
model.add(GlobalAveragePooling2D())
model.add(Dense(units=2, activation='softmax'))

# Show a summary of the model. Check the number of trainable parameters
model.summary()

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1. / 255)

# Change the batchsize according to your system RAM
train_batchsize = 10
val_batchsize = 10

train_generator = train_datagen.flow_from_directory(
    r'BinaryClassification\train',
    target_size=(64, 64),
    batch_size=train_batchsize,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    r'BinaryClassification\test',
    target_size=(64, 64),
    batch_size=val_batchsize,
    class_mode='categorical',
    shuffle=False)

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
# Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples / train_generator.batch_size,
    epochs=1,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples / validation_generator.batch_size,
    verbose=1)

# Save the model
model.save('ImageNetModel.h5')