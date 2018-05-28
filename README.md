# Flappy Tester

This repository is used to hold an implementation of Flappy Bird that is capable of having a artificial intelligence learning alogrithms applied against it 

There are two approaches used here: Q-Learning and DQN. In the case of Q-Learning, a variant is provided for a so-called "greedy" version.

## Q-Learning

To train a model, use this:

    python3 flappy_ql.py train

This will create a **model.txt** file. If you want to train a fresh model, delete that file.

To run against the model, use this:

    python3 flappy_ql.py run

## Q-Learning (ε-greedy):

To train a model, use this:

    python3 flappy_ql.py train greedy

This will create a **model_greedy.txt** file. If you want to train a fresh model, delete that file.

To run against the model, use this

    python3 flappy_ql.py run greedy

## Deep Q Network

To train a model, use this:

    python3 dqn.py train

That will create a **model.h5** file. If you want to train a fresh model, delete that file.

To run against a model, use this

    python3 dqn.py run
