from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans

import cv2
import random
from imutils import paths

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

full_data_x = train_images

num_steps = 200
batch_size = 500
k = 2
num_classes = 2
num_features = 784

# Input images
X = tf.placeholder(tf.float32, shape=[None, num_features])
# Labels (for assigning a label to a centroid and testing)
Y = tf.placeholder(tf.float32, shape=[None, num_classes])

# K-Means Parameters
kmeans = KMeans(inputs=X, num_clusters=k, distance_metric='cosine',
                use_mini_batch=True)

# Build KMeans graph
training_graph = kmeans.training_graph()

if len(training_graph) > 6:
    (all_scores, cluster_idx, scores, cluster_centers_initialized,
     cluster_centers_var, init_op, train_op) = training_graph
else:
    (all_scores, cluster_idx, scores, cluster_centers_initialized,
     init_op, train_op) = training_graph

cluster_idx = cluster_idx[0] # fix for cluster_idx being a tuple
avg_distance = tf.reduce_mean(scores)

# Initialize the variables (i.e. assign their default value)
init_vars = tf.global_variables_initializer()

# Start TensorFlow session
sess = tf.Session()

# Run the initializer
sess.run(init_vars, feed_dict={X: full_data_x})
sess.run(init_op, feed_dict={X: full_data_x})

# Training
for i in range(1, num_steps + 1):
    _, d, idx = sess.run([train_op, avg_distance, cluster_idx],
                         feed_dict={X: full_data_x})
    if i % 10 == 0 or i == 1:
        print("Step %i, Avg Distance: %f" % (i, d))

# Assign a label to each centroid
# Count total number of labels per centroid, using the label of each training
# sample to their closest centroid (given by 'idx')
counts = np.zeros(shape=(k, num_classes))
for i in range(len(idx)):
    counts[idx[i]] += train_labels[i]
# Assign the most frequent label to the centroid
labels_map = [np.argmax(c) for c in counts]
labels_map = tf.convert_to_tensor(labels_map)

# Evaluation ops
# Lookup: centroid_id -> label
cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)
# Compute accuracy
correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(Y, 1), tf.int32))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Test Model
test_x, test_y = test_images, test_labels
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))