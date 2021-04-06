#!/usr/bin/env python
# coding: utf-8

# In[2]:
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
import os
import math
import pandas as pd
import matplotlib.image as img
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# In[5]:
tf.__version__
print(device_lib.list_local_devices())

# In[6]:
# return gray image
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# In[7]:
# training targets, you can use your custome csv file if you already created it with "training-sample.py"
training_targets_cv = pd.read_csv('data_csv/jester-v1-train.csv', header=None, sep=";", index_col=0, squeeze=True).to_dict()

# validation targets, you can use your custome csv file if you already created it with "validation-sample.py"
targets_validation_cv = pd.read_csv('data_csv/jester-v1-validation.csv', header=None, sep=";", index_col=0, squeeze=True).to_dict()

# In[8]:
print("training_targets_cv ====>> " + str(len(training_targets_cv)))
print("targets_validation_cv ====>> " + str(len(targets_validation_cv)))

# In[9]:
# classes label you want to use all labels 
label = pd.read_csv('data_csv/jester-v1-labels.csv', header=None, usecols=[0])
label.head()
targets_name = label[0].tolist()
len(targets_name)

# In[11]:
# Get the data directories
path = "training_samples/"
path_cv = "validation_samples/"

dirs = os.listdir(path)
dirs_cv = os.listdir(path_cv)

# In[12]:
# number of samples for training and validation
print("len(dirs)" + str(len(dirs)))
print("len(dirs_cv)" + str(len(dirs_cv)))

# In[14]:
'''
The videos do not have the same number of frames, here we try to unify.
'''
hm_frames = 30 # number of frames
# unify number of frames for each training
def get_unify_frames(path):
    offset = 0
    # pick frames
    frames = os.listdir(path)
    frames_count = len(frames)
    # unify number of frames 
    if hm_frames > frames_count:
        # duplicate last frame if video is shorter than necessary
        frames += [frames[-1]] * (hm_frames - frames_count)
    elif hm_frames < frames_count:
        # If there are more frames, then sample starting offset
        #diff = (frames_count - hm_frames)
        #offset = diff-1 
        frames = frames[0:hm_frames]
    return frames

# In[15]:
# Resize frames
def resize_frame(frame):
    frame = img.imread(frame)
    frame = cv2.resize(frame, (64, 64))
    return frame

# In[16]:

class Conv3DModel(tf.keras.Model):
  def __init__(self):
    super(Conv3DModel, self).__init__()
    # Convolutions
    self.conv1 = tf.compat.v2.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', name="conv1", data_format='channels_last')
    self.pool1 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), data_format='channels_last')
    self.conv2 = tf.compat.v2.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', name="conv1", data_format='channels_last')
    self.pool2 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), data_format='channels_last')

    # LSTM & Flatten
    self.convLSTM = tf.keras.layers.ConvLSTM2D(40, (3, 3))
    self.flatten = tf.keras.layers.Flatten(name="flatten")

    # Dense layers
    self.d1 = tf.keras.layers.Dense(128, activation='relu', name="d1")
    self.out = tf.keras.layers.Dense(27, activation='softmax', name="output")

  def call(self, x):
    x = self.conv1(x)
    x = self.pool1(x)
    x = self.conv2(x)
    x = self.pool2(x)
    x = self.convLSTM(x)
    #x = self.pool2(x)
    #x = self.conv3(x)
    #x = self.pool3(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.out(x)

#Model Definitions
model = Conv3DModel()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Loss
train_loss = tf.keras.metrics.Mean(name='train_loss')
valid_loss = tf.keras.metrics.Mean(name='valid_loss')

# Accuracy
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

# Checkpoint
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(ckpt, 'training_checkpoints/tf_ckpts', max_to_keep=60)
ckpt.restore(manager.latest_checkpoint)

# train_step & valid_step

def train_step(image, targets):
    with tf.GradientTape() as tape:
        # Make a prediction on all the batch
        predictions = model(image)
        # Get the error/loss on these predictions
        loss = loss_fn(targets, predictions)
    # Compute the gradient which respect to the loss
    grads = tape.gradient(loss, model.trainable_variables)
    # Change the weights of the model
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # The metrics are accumulate over time. You don't need to average it yourself.
    train_loss(loss)
    train_accuracy(targets, predictions)


def valid_step(image, targets):
    predictions = model(image)
    t_loss = loss_fn(targets, predictions)
    # Set the metrics for the test
    valid_loss(t_loss)
    valid_accuracy(targets, predictions)

# General vars
training_data = []
training_targets = []
counter_training = 0

validation_data = []
validation_targets = []
counter_validation = 0

# Adjust training data
new_frames = [] # training data after resize & unify
new_frames_cv = []

# In[23]:
# Function to empty the RAM
def release_list(a):
    del a[:]
    del a

def getImagesPartitionByPartition():

    global dirs
    global dirs_cv

    global counter_training
    global training_data
    global training_targets

    global counter_validation
    global validation_data
    global validation_targets

    global new_frames
    global new_frames_cv

    dirs = np.array_split(dirs, 6)
    #dirs_cv = np.array_split(dirs_cv, 8)
    for i in range(0, 5):
        for directory in dirs[i]:
            new_frame = []  # one training
            # Frames in each folder
            frames = get_unify_frames(path + directory)
            if len(frames) == hm_frames:  # just to be sure
                for frame in frames:
                    frame = resize_frame(path + directory + '/' + frame)
                    # print(path+directory+'/'+frame)
                    new_frame.append(rgb2gray(frame))
                    if len(new_frame) == 15:  # partition each training on two trainings.
                        new_frames.append(new_frame)  # append each partition to training data
                        training_targets.append(targets_name.index(training_targets_cv[int(directory)]))
                        counter_training += 1
                        new_frame = []

        #if (len(training_data) > 0):
        #    training_data = np.append(training_data, np.array(new_frames[0:counter_training], dtype=np.float32))
        #else:
        training_data = np.array(new_frames[0:counter_training], dtype=np.float32)
        counter_training = 0
        release_list(new_frames)

        for directory in dirs_cv:
            new_frame = []
            # Frames in each folder
            frames = get_unify_frames(path_cv + directory)
            if len(frames) == hm_frames:
                for frame in frames:
                    frame = resize_frame(path_cv + directory + '/' + frame)
                    new_frame.append(rgb2gray(frame))
                    if len(new_frame) == 15:
                        new_frames_cv.append(new_frame)
                        validation_targets.append(targets_name.index(targets_validation_cv[int(directory)]))
                        counter_validation += 1
                        new_frame = []

        #if (len(validation_data) > 0):
        #    validation_data = np.append(validation_data, np.array(new_frames_cv[0:counter_validation], dtype=np.float32))
        #else:
        validation_data = np.array(new_frames_cv[0:counter_validation], dtype=np.float32)
        counter_validation = 0
        release_list(new_frames_cv)

        # Normalisation: training
        print('old mean', training_data.mean())
        scaler = StandardScaler()
        scaled_images = scaler.fit_transform(training_data.reshape(-1, 15 * 64 * 64))
        print('new mean', scaled_images.mean())
        scaled_images = scaled_images.reshape(-1, 15, 64, 64, 1)
        print(scaled_images.shape)

        # Normalisation: validation
        print('old mean', validation_data.mean())
        scaler = StandardScaler()
        scaled_images_cv = scaler.fit_transform(validation_data.reshape(-1, 15 * 64 * 64))
        print('new mean', scaled_images_cv.mean())
        scaled_images_cv = scaled_images_cv.reshape(-1, 15, 64, 64, 1)
        print(scaled_images_cv.shape)

        # use tensorflow Dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((scaled_images, training_targets))
        validation_dataset = tf.data.Dataset.from_tensor_slices((scaled_images_cv, validation_targets))

        # In[45]:
        model(scaled_images[0:2])

        # In[46]:
        model.summary()

        # #### here I use the checkpoints
        # read more:
        # https://www.tensorflow.org/beta/guide/checkpoints

        epoch = 10
        batch_size = 32
        b = 0
        training_acc = []
        validation_acc = []

        for epoch in range(epoch):
            # Training set
            for images_batch, targets_batch in train_dataset.batch(batch_size):
                train_step(images_batch, targets_batch)
                template = '\r Batch {}/{}, Loss: {}, Accuracy: {}'
                print(template.format(
                    b, len(training_targets), train_loss.result(),
                    train_accuracy.result() * 100
                ), end="")
                b += batch_size
            # Validation set
            for images_batch, targets_batch in validation_dataset.batch(batch_size):
                valid_step(images_batch, targets_batch)

            template = '\nEpoch {}, Valid Loss: {}, Valid Accuracy: {}'

            print(template.format(
                epoch + 1,
                valid_loss.result(),
                valid_accuracy.result() * 100)
            )

            training_acc.append(float(train_accuracy.result() * 100))
            validation_acc.append(float(valid_accuracy.result() * 100))
            ckpt.step.assign_add(1)
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
            valid_loss.reset_states()
            valid_accuracy.reset_states()
            train_accuracy.reset_states()
            train_loss.reset_states()

        # Erease Training Data & Validation Data
        training_data = []
        validation_data = []
        training_targets = []
        validation_targets = []
        scaled_images = []
        scaled_images_cv = []


getImagesPartitionByPartition()

# In[31]:

## 2. use a more complex syntax, for experts
# My model

# In[209]:
print(manager.checkpoints)

# In[217]:
# save the model for use in the application
model.save_weights('weights/model', save_format='tf')