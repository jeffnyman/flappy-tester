from itertools import cycle
import random
import sys
import pygame
from pygame.locals import *

FPS = 30
SCREEN_WIDTH = 288
SCREEN_HEIGHT = 404

PIPE_GAP_SIZE = 100
BASE_Y = SCREEN_HEIGHT
PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])
IMAGES, HITMASKS = {}, {}

pygame.init()

FPS_CLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

pygame.display.set_caption('Flappy Tester')

AGENT = (
    'assets/sprites/redbird-upflap.png',
    'assets/sprites/redbird-midflap.png',
    'assets/sprites/redbird-downflap.png',
)

IMAGES['agent'] = (
    pygame.image.load(AGENT[0]).convert_alpha(),
    pygame.image.load(AGENT[1]).convert_alpha(),
    pygame.image.load(AGENT[2]).convert_alpha(),
)

agent_height = IMAGES['agent'][0].get_height()

BACKGROUND = 'assets/sprites/background-black.png'

IMAGES['background'] = pygame.image.load(BACKGROUND).convert()

PIPE = 'assets/sprites/pipe-green.png'

IMAGES['pipe'] = (
    pygame.transform.rotate(pygame.image.load(PIPE).convert_alpha(), 180),
    pygame.image.load(PIPE).convert_alpha(),
)


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


HITMASKS['pipe'] = (
    get_hitmask(IMAGES['pipe'][0]),
    get_hitmask(IMAGES['pipe'][1]),
)

HITMASKS['agent'] = (
    get_hitmask(IMAGES['agent'][0]),
    get_hitmask(IMAGES['agent'][1]),
    get_hitmask(IMAGES['agent'][2]),
)


class State:
    def __init__(self):
        self.score = 0
        self.agent_index = 0
        self.loop_iteration = 0

        self.agent_x = int(SCREEN_WIDTH * 0.2)
        self.agent_y = int((SCREEN_HEIGHT - IMAGES['agent'][0].get_height()) / 2)

        new_pipe1 = generate_pipe()
        new_pipe2 = generate_pipe()

        self.upper_pipes = [
            {'x': SCREEN_WIDTH + 200, 'y': new_pipe1[0]['y']},
            {'x': SCREEN_WIDTH + 200 + (SCREEN_WIDTH / 2), 'y': new_pipe2[0]['y']},
        ]

        self.lower_pipes = [
            {'x': SCREEN_WIDTH + 200, 'y': new_pipe1[1]['y']},
            {'x': SCREEN_WIDTH + 200 + (SCREEN_WIDTH / 2), 'y': new_pipe2[1]['y']},
        ]

        self.pipe_velocity_x = -4

        self.agent_velocity_y = -9
        self.agent_max_velocity_y = 10
        self.agent_min_velocity_y = -8

        self.agent_accelerate_y = 1
        self.agent_flap_accelerate = -9
        self.agent_flapped = False

    def next(self, action):
        pygame.event.pump()

        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()

        # Reward for staying alive.

        reward = 0.1
        alive = 1

        if action == 1:
            if self.agent_y > -2 * IMAGES['agent'][0].get_height():
                self.agent_velocity_y = self.agent_flap_accelerate
                self.agent_flapped = True
            
        agent_middle_position = self.agent_x + IMAGES['agent'][0].get_width() / 2

        for pipe in self.upper_pipes:
            pipe_middle_position = pipe['x'] + IMAGES['pipe'][0].get_width() / 2

            if pipe_middle_position <= agent_middle_position < pipe_middle_position + 4:
                self.score += 1
                # Reward for crossing the pipe gap.
                reward = 1

        score = self.score

        if (self.loop_iteration + 1) % 3 == 0:
            self.agent_index = next(PLAYER_INDEX_GEN)

        self.loop_iteration = (self.loop_iteration + 1) % 30
        
        if self.agent_velocity_y < self.agent_max_velocity_y and not self.agent_flapped:
            self.agent_velocity_y += self.agent_accelerate_y

        if self.agent_flapped:
            self.agent_flapped = False

        self.agent_y += min(self.agent_velocity_y, BASE_Y - self.agent_y - agent_height)

        if self.agent_y < 0:
            self.agent_y = 0

        for upper_pipe, lower_pipe in zip(self.upper_pipes, self.lower_pipes):
            upper_pipe['x'] += self.pipe_velocity_x
            lower_pipe['x'] += self.pipe_velocity_x

        if 0 < self.upper_pipes[0]['x'] < 5:
            new_pipe = generate_pipe()
            self.upper_pipes.append(new_pipe[0])
            self.lower_pipes.append(new_pipe[1])

        if self.upper_pipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            self.upper_pipes.pop(0)
            self.lower_pipes.pop(0)

        collision = check_for_collision({'x': self.agent_x, 'y': self.agent_y, 'index': self.agent_index},
                                        self.upper_pipes, self.lower_pipes)
        if collision[0]:
            alive = 0
            self.__init__()
            reward = -1

        SCREEN.blit(IMAGES['background'], (0, 0))

        for upper_pipe, lower_pipe in zip(self.upper_pipes, self.lower_pipes):
            SCREEN.blit(IMAGES['pipe'][0], (upper_pipe['x'], upper_pipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lower_pipe['x'], lower_pipe['y']))

        SCREEN.blit(IMAGES['agent'][self.agent_index], (self.agent_x, self.agent_y))
        image = pygame.surfarray.array3d(pygame.display.get_surface())

        pygame.display.update()
        FPS_CLOCK.tick(FPS)

        return [image, score, reward, alive]


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
