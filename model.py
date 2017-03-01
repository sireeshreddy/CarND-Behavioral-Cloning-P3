import os
import csv
import cv2
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, ELU, Dropout, Cropping2D, normalization
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.utils import shuffle


samples = []
with open('./data/driving_log.csv') as csvfile: # using the udacity supplied test data
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples) 
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                # append center camera image as well as flipped image
                center_name = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(center_name)
                center_angle = float(batch_sample[3])
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2YUV)
                images.append(center_image)
                measurements.append(center_angle)
                images.append(cv2.flip(center_image,1))
                measurements.append(center_angle*-1.0)
                # append left camera image as well as flipped image
                left_name = './data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(left_name)
                left_angle = float(batch_sample[3]) + 0.23
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2YUV)
                images.append(left_image)
                measurements.append(left_angle)
                images.append(cv2.flip(left_image,1))
                measurements.append(left_angle*-1.0)
                # append right camera image as well as flipped image
                right_name = './data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(right_name)
                right_angle = float(batch_sample[3]) - 0.23
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2YUV)
                images.append(right_image)
                measurements.append(right_angle)
                images.append(cv2.flip(right_image,1))
                measurements.append(right_angle*-1.0)

                # borrowed from https://github.com/upul/Behavioral-Cloning/blob/master/helper.py
                # apply random gamma correction to augment data
                gamma = np.random.uniform(0.4, 1.5)
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255
                                    for i in np.arange(0, 256)]).astype("uint8")
                # apply gamma correction using the lookup table
                images.append(cv2.LUT(center_image, table))
                measurements.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# create training model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24, 5, 5, subsample=(2,2)))
model.add(ELU())
model.add(Conv2D(36, 5, 5, subsample=(2,2)))
model.add(ELU())
model.add(Conv2D(48, 5, 5, subsample=(2,2)))
model.add(ELU())
model.add(Conv2D(64, 3, 3))
model.add(ELU())
model.add(Conv2D(64, 3, 3))
model.add(ELU())
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(ELU())
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(ELU())
model.add(Dense(10))
model.add(Dense(1))
model.summary()

# compile model using mean standard error and Adam optimizer
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, 
                    samples_per_epoch = len(train_samples) * 8, 
                    nb_epoch = 15,
                    validation_data = (validation_generator),              
                    nb_val_samples = len(validation_samples))
# save model to model.h5
model.save('model.h5')
