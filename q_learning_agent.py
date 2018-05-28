import os
import numpy as np


class QLearningAgent:
    def __init__(self, is_training):
        self.training = is_training

        self.episode = 0
        self.discount_factor = 0.95
        self.learning_rate = 0.7

        self.previous_state = [96, 47, 0]
        self.previous_action = 0

        self.moves = []
        self.scores = []
        self.max_score = 0

        self.x_dimension = 130
        self.y_dimension = 130
        self.v_dimension = 20

        self.q_values = np.zeros((self.x_dimension, self.y_dimension, self.v_dimension, 2))

        self.initialize_model()

    def initialize_model(self):
        if os.path.exists("model.txt"):
            q_file = open("model.txt", "r")
            line = q_file.readline()

            if self.training:
                self.episode = int(line)
            line = q_file.readline()

            while len(line) != 0:
                state = line.split(',')
                self.q_values[int(state[0]), int(state[1]), int(state[2]), int(state[3])] = float(state[4])
                line = q_file.readline()

            q_file.close()

    def action(self, x_distance, y_distance, velocity):
        """
        The action stores the transition from the previous state to the
        current state. That transition is the action that led from the
        previous to the current.
        """
        if self.training:
            state = [x_distance, y_distance, velocity]
            self.moves.append([self.previous_state, self.previous_action, state, 0])
            self.previous_state = state

        # In order for the agent to act based on learning, it needs to get
        # the action with the maximum q value for the current state.

        if self.q_values[x_distance, y_distance, velocity][0] >= self.q_values[x_distance, y_distance, velocity][1]:
            self.previous_action = 0
        else:
            self.previous_action = 1

        return self.previous_action

    def record_reward(self, reward):
        """
        The reward is being applied to the last transition that led to the
        reward.
        """
        self.moves[-1][3] = reward

    def update_q_values(self, score):
        self.episode += 1
        self.max_score = max(self.max_score, score)

        print("Episode: " + str(self.episode) +
              " Score: " + str(score) +
              " Max Score: " + str(self.max_score))

        self.scores.append(score)

        if self.training:
            history = list(reversed(self.moves))
            first = True
            second = True
            jump = True

            if history[0][1] < 69:
                jump = False

            for move in history:
                [x, y, v] = move[0]
                action = move[1]

                [x1, y1, z1] = move[2]
                reward = move[3]

                # Penalize the last two states before a collision.

                if first or second:
                    reward = -1000000
                    if first:
                        first = False
                    else:
                        second = False

                # Penalize the last jump before a collision.

                if jump and action:
                    reward = -1000000
                    jump = False

                self.q_values[x, y, v, action] = (1 - self.learning_rate) *\
                                                 (self.q_values[x, y, v, action]) + self.learning_rate *\
                                                 (reward + self.discount_factor
                                                  * max(self.q_values[x1, y1, z1, 0],
                                                        self.q_values[x1, y1, z1, 1]))

            self.moves = []

    def save_model(self):
        data = str(self.episode) + "\n"

        for x in range(self.x_dimension):
            for y in range(self.y_dimension):
                for v in range(self.v_dimension):
                    for a in range(2):
                        data += str(x) + ", " + str(y) +\
                                ", " + str(v) + ", " +\
                                str(a) + ", " + str(self.q_values[x, y, v, a]) + "\n"

        q_file = open("model.txt", "w")
        q_file.write(data)
        q_file.close()

        data1 = ''

        for i in range(len(self.scores)):
            data1 += str(self.scores[i]) + "\n"

        s_file = open("model_scores.txt", "a+")
        s_file.write(data1)
        s_file.close()
