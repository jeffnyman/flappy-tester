#!/usr/bin/env python3

import sys
import random
import pygame

from itertools import cycle
from pygame.locals import *

from q_learning_agent import QLearningAgent
from q_learning_agent_greedy import QLearningAgentGreedy

training_mode = None
operation = sys.argv[1]

if operation == 'train':
    training_mode = True
elif operation == 'run':
    training_mode = False

if len(sys.argv) == 2:
    Agent = QLearningAgent(training_mode)
elif sys.argv[2] == 'greedy':
    Agent = QLearningAgentGreedy(training_mode)

FPS = 30
FPS_CLOCK = None

SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
SCREEN = None

PIPE_GAP_SIZE = 100
BASE_Y = SCREEN_HEIGHT * 0.79

IMAGES = {}
HITMASKS = {}

AGENTS_LIST = (
    (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    ),
    (
        'assets/sprites/bluebird-upflap.png',
        'assets/sprites/bluebird-midflap.png',
        'assets/sprites/bluebird-downflap.png',
    ),
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
)

BACKGROUNDS_LIST = (
    'assets/sprites/background-day.png',
    'assets/sprites/background-night.png',
)

PIPES_LIST = (
    'assets/sprites/pipe-green.png',
    'assets/sprites/pipe-red.png',
)


def environment():
    global SCREEN, FPS_CLOCK

    pygame.init()

    FPS_CLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    pygame.display.set_caption('Flappy Tester')

    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    while True:
        background = random.randint(0, len(BACKGROUNDS_LIST) - 1)
        IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[background]).convert()

        agent = random.randint(0, len(AGENTS_LIST) - 1)
        IMAGES['agent'] = (
            pygame.image.load(AGENTS_LIST[agent][0]).convert_alpha(),
            pygame.image.load(AGENTS_LIST[agent][1]).convert_alpha(),
            pygame.image.load(AGENTS_LIST[agent][2]).convert_alpha(),
        )

        pipe = random.randint(0, len(PIPES_LIST) - 1)
        IMAGES['pipe'] = (
            pygame.transform.flip(pygame.image.load(PIPES_LIST[pipe]).convert_alpha(), False, True),
            pygame.image.load(PIPES_LIST[pipe]).convert_alpha(),
        )

        HITMASKS['pipe'] = (
            get_hitmask(IMAGES['pipe'][0]),
            get_hitmask(IMAGES['pipe'][1]),
        )

        HITMASKS['agent'] = (
            get_hitmask(IMAGES['agent'][0]),
            get_hitmask(IMAGES['agent'][1]),
            get_hitmask(IMAGES['agent'][2]),
        )

        starting = setup_starting_animation()
        observation = environment_loop(starting)


def setup_starting_animation():
    agent_y = int((SCREEN_HEIGHT - IMAGES['agent'][0].get_height()) / 2)
    agent_index_cycle = cycle([0, 1, 2, 1])

    return {
        'agent_y': agent_y,
        'base_x': 0,
        'agent_index_cycle': agent_index_cycle,
    }


def environment_loop(starting):
    score = 0
    agent_index = 0
    loop_iteration = 0

    agent_index_cycle = starting['agent_index_cycle']
    agent_x, agent_y = int(SCREEN_WIDTH * 0.2), starting['agent_y']

    base_x = starting['base_x']
    base_shift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    new_pipe1 = generate_pipe()
    new_pipe2 = generate_pipe()

    upper_pipes = [
        {'x': SCREEN_WIDTH + 200, 'y': new_pipe1[0]['y']},
        {'x': SCREEN_WIDTH + 200 + (SCREEN_WIDTH / 2), 'y': new_pipe2[0]['y']},
    ]

    lower_pipes = [
        {'x': SCREEN_WIDTH + 200, 'y': new_pipe1[1]['y']},
        {'x': SCREEN_WIDTH + 200 + (SCREEN_WIDTH / 2), 'y': new_pipe2[1]['y']},
    ]

    pipe_velocity_x = -4

    agent_velocity_y = -9
    agent_max_velocity_y = 10

    agent_accelerate_y = 1
    agent_flap_accelerate = -9
    agent_flapped = False

    reward = 0

    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                if training_mode:
                    Agent.save_model()
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                if agent_y > -2 * IMAGES['agent'][0].get_height():
                    agent_velocity_y = agent_flap_accelerate
                    agent_flapped = True
        
        x_dist_pipe = lower_pipes[0]['x'] - agent_x + 30

        if x_dist_pipe > 0:
            pipe_number = 0
        else:
            pipe_number = 1

        x_dist_lower_pipe = lower_pipes[pipe_number]['x'] - agent_x
        y_dist_lower_pipe = lower_pipes[pipe_number]['y'] - agent_y

        # Agent must perform an action in the environment.

        if Agent.action(int((x_dist_lower_pipe + 60) / 5),
                        int((y_dist_lower_pipe + 225) / 5),
                        int(agent_velocity_y + 9)):
            if agent_y > -2 * IMAGES['agent'][0].get_height():
                    agent_velocity_y = agent_flap_accelerate
                    agent_flapped = True

        # A check is made to determine if the agent has collided with
        # any of the pipes or has collided with the ground.

        collision = check_for_collision({'x': agent_x, 'y': agent_y,
                                         'index': agent_index},
                                        upper_pipes, lower_pipes)
        if collision[0]:
            Agent.update_q_values(score)

            return {
                'y': agent_y,
                'groundCrash': collision[1],
                'base_x': base_x,
                'upper_pipes': upper_pipes,
                'lower_pipes': lower_pipes,
                'score': score,
                'agent_velocity_y': agent_velocity_y,
            }

        # A minimal reward is provided for the lack of a collision.

        reward = 1

        # Apply score and reward based on whether the agent has navigated
        # between the pipes.

        agent_middle_position = agent_x + IMAGES['agent'][0].get_width() / 2

        for pipe in upper_pipes:
            pipe_middle_position = pipe['x'] + IMAGES['pipe'][0].get_width() / 2

            if pipe_middle_position <= agent_middle_position < pipe_middle_position + 4:
                score += 1
                reward = 5

        # When training, it's important to record observations that lead to
        # rewards.

        if training_mode:
            Agent.record_reward(reward)

        # The agent has had a base_x change; meaning movement along the x
        # direction. That movement has to be reflected and the observation
        # point has to be shifted.

        if (loop_iteration + 1) % 3 == 0:
            agent_index = next(agent_index_cycle)

        loop_iteration = (loop_iteration + 1) % 30

        base_x = -((-base_x + 100) % base_shift)

        # Agent movement has to be reflected based on the velocity and
        # acceleration.

        if agent_velocity_y < agent_max_velocity_y and not agent_flapped:
            agent_velocity_y += agent_accelerate_y

        if agent_flapped:
            agent_flapped = False

        agent_height = IMAGES['agent'][agent_index].get_height()
        agent_y += min(agent_velocity_y, BASE_Y - agent_y - agent_height)

        # The pipes will be shifted to the left after an action.

        for upper_pipe, lower_pipe in zip(upper_pipes, lower_pipes):
            upper_pipe['x'] += pipe_velocity_x
            lower_pipe['x'] += pipe_velocity_x

        # A new pipe has to be added to the environment when the first pipe
        # is about to touch the leftmost area of the environment.

        if 0 < upper_pipes[0]['x'] < 5:
            new_pipe = generate_pipe()
            upper_pipes.append(new_pipe[0])
            lower_pipes.append(new_pipe[1])

        # A pipe must be removed if it is out of the environment as it no
        # longer contributes to observations.

        if upper_pipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upper_pipes.pop(0)
            lower_pipes.pop(0)

        # What follows is drawing all of the sprites to the environment.
        # This is providing the agent with the observation of the new
        # state after an action has been taken.

        SCREEN.blit(IMAGES['background'], (0, 0))

        for upper_pipe, lower_pipe in zip(upper_pipes, lower_pipes):
            SCREEN.blit(IMAGES['pipe'][0], (upper_pipe['x'], upper_pipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lower_pipe['x'], lower_pipe['y']))

        SCREEN.blit(IMAGES['base'], (base_x, BASE_Y))

        display_score(score)

        agent_surface = IMAGES['agent'][agent_index]
        SCREEN.blit(agent_surface, (agent_x, agent_y))

        pygame.display.update()
        FPS_CLOCK.tick(FPS)


def generate_pipe():
    """
    Generates a random pipe. Note that the PIPE_GAP_SIZE is the default gap
    between the upper and lower part of a given pipe. In this function, the
    gap_y value is is the actual gap between the upper and lower pipe for
    each generated pipe.
    """
    gap_y = random.randrange(0, int(BASE_Y * 0.6 - PIPE_GAP_SIZE))
    gap_y += int(BASE_Y * 0.2)
    pipe_height = IMAGES['pipe'][0].get_height()
    pipe_x = SCREEN_WIDTH + 10

    return [
        {'x': pipe_x, 'y': gap_y - pipe_height},
        {'x': pipe_x, 'y': gap_y + PIPE_GAP_SIZE},
    ]


def display_score(score):
    score_digits = [int(x) for x in list(str(score))]
    total_width = 0

    for digit in score_digits:
        total_width += IMAGES['numbers'][digit].get_width()

    x_offset = (SCREEN_WIDTH - total_width) / 2

    for digit in score_digits:
        SCREEN.blit(IMAGES['numbers'][digit], (x_offset, SCREEN_HEIGHT * 0.1))
        x_offset += IMAGES['numbers'][digit].get_width()


def check_for_collision(agent, upper_pipes, lower_pipes):
    agent_index = agent['index']
    agent['w'] = IMAGES['agent'][0].get_width()
    agent['h'] = IMAGES['agent'][0].get_height()

    if agent['y'] + agent['h'] >= BASE_Y - 1:
        return [True, True]
    else:
        agent_rect = pygame.Rect(agent['x'], agent['y'], agent['w'], agent['h'])

        pipe_width = IMAGES['pipe'][0].get_width()
        pipe_height = IMAGES['pipe'][0].get_height()

        for upper_pipe, lower_pipe in zip(upper_pipes, lower_pipes):
            upper_pipe_rect = pygame.Rect(upper_pipe['x'], upper_pipe['y'], pipe_width, pipe_height)
            lower_pipe_rect = pygame.Rect(lower_pipe['x'], lower_pipe['y'], pipe_width, pipe_height)

            agent_hitmask = HITMASKS['agent'][agent_index]
            upper_hitmask = HITMASKS['pipe'][0]
            lower_hitmask = HITMASKS['pipe'][1]

            upper_collide = pixel_collision(agent_rect, upper_pipe_rect, agent_hitmask, upper_hitmask)
            lower_collide = pixel_collision(agent_rect, lower_pipe_rect, agent_hitmask, lower_hitmask)

            if upper_collide or lower_collide:
                return [True, False]

    return [False, False]


def pixel_collision(rect1, rect2, hitmask1, hitmask2):
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True

    return False


def get_hitmask(image):
    """
    Provides a hitmask using the alpha of the passed in image.
    """
    mask = []

    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x, y))[3]))

    return mask


if __name__ == '__main__':
    environment()
