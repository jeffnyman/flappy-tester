#!/usr/bin/env python3

import os
import sys
import random
import numpy as np

from collections import deque
from flappy_dqn import State

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Activation, Flatten
from keras.optimizers import Adam

import skimage

from skimage import transform, color, exposure

num_actions = 2
discount = 0.99
observe = 3200
explore = 3000000

FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.1

replay_memory = 50000


def build_network():
    print("START: build model network.")
    model_network = Sequential()
    model_network.add(Conv2D(32, (8, 8), padding='same', strides=(4, 4), input_shape=(80, 80, 4)))
    model_network.add(Activation('relu'))
    model_network.add(Conv2D(64, (4, 4), padding='same', strides=(2, 2)))
    model_network.add(Activation('relu'))
    model_network.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1)))
    model_network.add(Activation('relu'))
    model_network.add(Flatten())
    model_network.add(Dense(512))
    model_network.add(Activation('relu'))
    model_network.add(Dense(num_actions))

    if os.path.exists("model.h5"):
        print("Loading weights from model.h5 file.")
        model_network.load_weights("model.h5")
        print("Weights loaded successfully.")

    adam = Adam(lr=1e-4)
    model_network.compile(loss='mse', optimizer=adam)

    print("END: model network built.")

    return model_network


def process(observation):
    """
    The process converts the input from rgb to grey and resizes the image
    from 288x404 to 80x80. The intensity levels of the image are also
    stretched or shrunk and the pixel values are scaled down to (0, 1).
    """
    image = skimage.color.rgb2gray(observation)
    image = skimage.transform.resize(image, (80, 80), mode='constant')
    image = skimage.exposure.rescale_intensity(image, out_range=(0, 255))
    image = image / 255.0

    return image


def train_network(model_network, mode):
    training_mode = None

    if mode == 'run':
        training_mode = False
    elif mode == 'train':
        training_mode = True

    if training_mode:
        epsilon = INITIAL_EPSILON
    else:
        epsilon = FINAL_EPSILON

    s_file = open("scores_dqn.txt", "a+")

    episode = 1
    time_step = 0
    loss = 0

    game = State()

    # Store the previous observations in replay memory.

    replay = deque()

    # Take action 0 and get the resultant state.

    image, score, reward, alive = game.next(0)

    # The image observation is preprocessed and stacked to 80x80x4 pixels.

    image = process(image)
    input_image = np.stack((image, image, image, image), axis=2)
    input_image = input_image.reshape(1, input_image.shape[0], input_image.shape[1], input_image.shape[2])
    
    while True:
        # Get an action from an epsilon greedy policy.
        if random.random() <= epsilon:
            action = random.randrange(num_actions)
        else:
            q = model_network.predict(input_image)
            action = np.argmax(q)

        # Decay epsilon linearly.

        if epsilon > FINAL_EPSILON and time_step > observe:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / explore

        # Take the selected action and get the resultant state.

        image1, score, reward, alive = game.next(action)

        # Each image has to be preprocessed and stacked to 80x80x4 pixels.

        image1 = process(image1)
        image1 = image1.reshape(1, image1.shape[0], image1.shape[1], 1)
        input_image1 = np.append(image1, input_image[:, :, :, :3], axis=3)

        if training_mode:
            # Add the current transition to the replay buffer.
            replay.append((input_image, action, reward, input_image1, alive))

            if len(replay) > replay_memory:
                replay.popleft()

            if time_step > observe:
                # Sample a mini-batch of size 32 from replay memory.
                mini_batch = random.sample(replay, 32)

                s, a, r, s1, alive = zip(*mini_batch)
                s = np.concatenate(s)
                s1 = np.concatenate(s1)

                targets = model_network.predict(s)
                targets[range(32), a] = r + discount * np.max(model_network.predict(s1), axis=1) * alive

                loss += model_network.train_on_batch(s, targets)

        input_image = input_image1
        time_step = time_step + 1

        if training_mode:
            # Save the weights after every 1000 time steps.
            if time_step % 1000 == 0:
                model_network.save_weights("model.h5", overwrite=True)

            print("TIME STEP: " + str(time_step) + ", EPSILON: " + str(epsilon) + ", ACTION: " +
                  str(action) + ", REWARD: " + str(reward) + ", Loss: " + str(loss))

            loss = 0
        elif not alive:
            print("EPISODE: " + str(episode) + ", SCORE: " + str(score))

            s_file.write(str(score)+"\n")
            episode += 1


if __name__ == "__main__":
    model = build_network()
    train_network(model, sys.argv[1])
