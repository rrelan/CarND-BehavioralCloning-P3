#  File Name : model.py
#  Written by : Ranjan Relan 
# Project     : P3 - Behavioral cloning  
###########################################


import csv
import cv2
import numpy as np


# pull metadata from csv file 
lines = []
with open('/home/carnd/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# define image path from where we need to read images  		
IMG_PATH = '/home/carnd/data/IMG/'

# define generator function so as to read images in batches of 32 
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                center = batch_sample[0].split('/')[-1]
                left = batch_sample[1].split('/')[-1]
                right = batch_sample[2].split('/')[-1]

                # Add center , left and right images
                img_center = cv2.imread(IMG_PATH + center)
                img_left = cv2.imread(IMG_PATH + left)
                img_right = cv2.imread(IMG_PATH + right)

                # Flip images from - center , left and right for generalization of model 
                augmented_center = cv2.flip(img_center, 1)
                augmented_left = cv2.flip(img_left, 1)
                augmented_right = cv2.flip(img_right, 1)

                # Take steering angle and create steering left and steering right measures 
                steering_center = float(batch_sample[3])
                steering_left = steering_center + 0.25
                steering_right = steering_center - 0.25

                # Add images 
                images.extend([img_center, img_left, img_right])
                images.extend([augmented_center, augmented_left, augmented_right])

                # Add steering measurements
                measurements.extend([steering_center, steering_left, steering_right])
                measurements.extend([-steering_center, -steering_left, -steering_right])

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)
    





# Libraries for splitting data and keras library for implementing CNN (simple way as it has higher level methods)			
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D

# split the data into train and test with 80% data for training and 20 % for testing by using train_test_split method 
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# use batch size of 32 to implement generator otherwise we shall get memory error as so much data cannot be handled in numpy array 			
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


model = Sequential()

# Normalize the data 
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))

# Crop the images - this avoid GIGO (Garbage In Garbage out strategt) - that is we get better trained models 
model.add(Cropping2D(cropping=((70,25), (0,0))))

# Implement the Nvidia model using sub sampling , introducing non linearity (relu) 
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,5,5, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Use adam optimizer for training and loss type as "mean square error"
model.compile(loss='mse', optimizer='adam')

# model training 
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=10)

# save the model 
model.save('model.h5')

