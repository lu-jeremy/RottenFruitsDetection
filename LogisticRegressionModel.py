<<<<<<< HEAD
import tensorflow as tf
import cv2
from imutils import paths
import numpy as np

import random

train_images = []
train_labels = []

test_images = []
test_labels = []


def path_images(folder_name, image_array, label_array):
    for image_path in paths.list_images(folder_name):

        image = image_path.split('/')[-1]

        image = cv2.imread(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, (28, 28))

        image = np.reshape(image, (784,))

        rand_index = random.randint(0, len(image_array))

        image_array.insert(rand_index, image)

        if 'freshapples' in folder_name:
            label_array.insert(rand_index, [0, 1])
        elif 'rottenapples' in folder_name:
            label_array.insert(rand_index, [1, 0])


path_images(r'C:\Users\bluet\PycharmProjects\RottenFruits\src\dataset\train\freshapples', train_images, train_labels)
path_images(r'C:\Users\bluet\PycharmProjects\RottenFruits\src\dataset\train\rottenapples', train_images, train_labels)

path_images(r'C:\Users\bluet\PycharmProjects\RottenFruits\src\dataset\test\freshapples', test_images, test_labels)
path_images(r'C:\Users\bluet\PycharmProjects\RottenFruits\src\dataset\test\rottenapples', test_images, test_labels)

trainX = tf.placeholder(tf.float32, shape=[None, 784])

trainY = tf.placeholder(tf.float32, shape=[None, 2])

w = tf.Variable(np.zeros((784, 2)), dtype=tf.float32)

b = tf.Variable(np.zeros(2), dtype=tf.float32)

prediction = tf.nn.softmax(tf.add(tf.matmul(trainX, w), b))

loss = tf.reduce_mean(-tf.reduce_sum(trainY * tf.log(prediction),
                                     reduction_indices=1))

learning_rate = 1e-10000

optimizer = tf.train.GradientDescentOptimizer(learning_rate)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:

    sess.run(init)

    for epoch in range(100):

        sess.run(train, feed_dict={trainX: train_images, trainY: train_labels})

        loss_value = sess.run(loss, feed_dict={trainX: train_images,
                                               trainY: train_labels})

        print(loss_value)

    saver.save(sess, 'train_rotten_model')

    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(trainY, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    accuracy_value = sess.run(accuracy, feed_dict={trainX: train_images,
                                                   trainY: train_labels})

    print(accuracy_value * 100, '%')
=======
import tensorflow as tf
import cv2
from imutils import paths
import numpy as np

import random

train_images = []
train_labels = []

test_images = []
test_labels = []


def path_images(folder_name, image_array, label_array):
    for image_path in paths.list_images(folder_name):

        image = image_path.split('/')[-1]

        image = cv2.imread(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, (28, 28))

        image = np.reshape(image, (784,))

        rand_index = random.randint(0, len(image_array))

        image_array.insert(rand_index, image)

        if 'freshapples' in folder_name:
            label_array.insert(rand_index, [0, 1])
        elif 'rottenapples' in folder_name:
            label_array.insert(rand_index, [1, 0])


path_images(r'C:\Users\bluet\PycharmProjects\RottenFruits\src\dataset\train\freshapples', train_images, train_labels)
path_images(r'C:\Users\bluet\PycharmProjects\RottenFruits\src\dataset\train\rottenapples', train_images, train_labels)

path_images(r'C:\Users\bluet\PycharmProjects\RottenFruits\src\dataset\test\freshapples', test_images, test_labels)
path_images(r'C:\Users\bluet\PycharmProjects\RottenFruits\src\dataset\test\rottenapples', test_images, test_labels)

trainX = tf.placeholder(tf.float32, shape=[None, 784])

trainY = tf.placeholder(tf.float32, shape=[None, 2])

w = tf.Variable(np.zeros((784, 2)), dtype=tf.float32)

b = tf.Variable(np.zeros(2), dtype=tf.float32)

prediction = tf.nn.softmax(tf.add(tf.matmul(trainX, w), b))

loss = tf.reduce_mean(-tf.reduce_sum(trainY * tf.log(prediction),
                                     reduction_indices=1))

learning_rate = 1e-10000

optimizer = tf.train.GradientDescentOptimizer(learning_rate)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:

    sess.run(init)

    for epoch in range(100):

        sess.run(train, feed_dict={trainX: train_images, trainY: train_labels})

        loss_value = sess.run(loss, feed_dict={trainX: train_images,
                                               trainY: train_labels})

        print(loss_value)

    saver.save(sess, 'train_rotten_model.ckpt')

    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(trainY, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    accuracy_value = sess.run(accuracy, feed_dict={trainX: train_images,
                                                   trainY: train_labels})

    print(accuracy_value * 100, '%')
>>>>>>> 4e22822301509d4267a16e7ea3b472f9d8a325b9
