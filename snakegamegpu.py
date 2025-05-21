import random
import torch
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import matplotlib.pyplot as plt
import time
import threading
import matplotlib.backends.backend_agg as agg
import pylab
import pygame
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(11, 256, 3).to(device)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game._is_collision(point_r)) or
            (dir_l and game._is_collision(point_l)) or
            (dir_u and game._is_collision(point_u)) or
            (dir_d and game._is_collision(point_d)),

            # Danger right
            (dir_u and game._is_collision(point_r)) or
            (dir_d and game._is_collision(point_l)) or
            (dir_l and game._is_collision(point_u)) or
            (dir_r and game._is_collision(point_d)),

            # Danger left
            (dir_d and game._is_collision(point_r)) or
            (dir_u and game._is_collision(point_l)) or
            (dir_r and game._is_collision(point_u)) or
            (dir_l and game._is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # if max memory is reached, pops left

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(torch.tensor(states, dtype=torch.float).to(device),
                                torch.tensor(actions, dtype=torch.long).to(device),
                                torch.tensor(rewards, dtype=torch.float).to(device),
                                torch.tensor(next_states, dtype=torch.float).to(device),
                                torch.tensor(dones, dtype=torch.bool).to(device))

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device),
                                torch.tensor(action, dtype=torch.long).unsqueeze(0).to(device),
                                torch.tensor(reward, dtype=torch.float).unsqueeze(0).to(device),
                                torch.tensor(next_state, dtype=torch.float).unsqueeze(0).to(device),
                                torch.tensor(done, dtype=torch.bool).unsqueeze(0).to(device))

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def save_model(self):
        model_folder_path = './'  # Ensure the directory path is correct
        os.makedirs(model_folder_path, exist_ok=True)  # Create the directory if it doesn't exist
        file_name = os.path.join(model_folder_path, 'snake_model.keras')
        self.model.save(file_name)
        print(f"Model saved to {file_name}")


# Update the MODEL_PATH to include the correct directory and file name
MODEL_PATH = './snake_model.keras'
# Load the saved model if it exists
agent = Agent()
# Update the model loading logic to correctly load the state dictionary
if os.path.exists(MODEL_PATH):
    agent.model.load_state_dict(torch.load(MODEL_PATH))
    print(f"Model loaded from {MODEL_PATH}")
else:
    print("No saved model found. Starting with a new model.")


# Update the graph rendering logic
def update_graph_inline(attempts, scores, game_display, game_width):
    fig = pylab.figure(figsize=[4, 4], dpi=100)
    ax = fig.gca()
    ax.plot(attempts, scores, marker='o')
    ax.set_title('Attempts vs. Score')
    ax.set_xlabel('Attempt')
    ax.set_ylabel('Score')
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()
    graph_surface = pygame.image.fromstring(raw_data, size, "RGB")
    game_display.blit(graph_surface, (game_width, 0))
    pylab.close(fig)  # Close the figure to prevent memory issues


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    game = SnakeGameAI()
    attempts = []

    while True:
        start_time = time.time()

        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f'Game {agent.n_games}, Score: {score}, Record: {record}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            elapsed_time = time.time() - start_time
            attempts.append(agent.n_games)

            # Update the graph inline beside the game
            update_graph_inline(attempts, plot_scores, game.display, game.w)

            plot(plot_scores, plot_mean_scores)

            # Save the model after every 100 games
            if agent.n_games % 10 == 0:
                agent.save_model()

    agent.save_model()


if __name__ == '__main__':
    train()
