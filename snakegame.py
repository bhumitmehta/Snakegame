import pygame
import random
import time
import sys
# // import a nureal network library for feed forward neural network
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#import a feed forward neural network library for reinforcement learning
# // import a reinforcement learning library
# Importing the necessary libraries
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import pylab
import os
import atexit


# Initialize Pygame
pygame.init()
# Set up display
WIDTH, HEIGHT = 400, 400  # Increase the width to make space for the graph
WINDOW = pygame.display.set_mode((WIDTH + 400, HEIGHT))  # Add extra width for the graph
pygame.display.set_caption("Snake Game")
# Set up colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Set up game variables
SNAKE_SIZE = 10
SNAKE_SPEED = 35
FPS = pygame.time.Clock()
# Set up fonts

FONT = pygame.font.SysFont("Arial", 25)
# Set up game over flag
game_over = False
# Set up snake and food positions
snake_pos = [[50, 50], [60, 50], [70, 50]]
snake_direction = "RIGHT"
food_pos = [random.randrange(1, (WIDTH // SNAKE_SIZE)) * SNAKE_SIZE,
             random.randrange(1, (HEIGHT // SNAKE_SIZE)) * SNAKE_SIZE]
food_spawn = True
# Set up score
score = 0
# Initialize variables to track attempts and scores
attempts = []
scores = []
attempt_count = 0
# Function to display score
def show_score(choice,color,size):
    score_surface = FONT.render('Score : ' + str(score), True, color)
    score_rect = score_surface.get_rect()
    if choice == 1:
        score_rect.midtop = (50, 15)
    else:
        score_rect.midtop = (WIDTH // 2, HEIGHT // 4)
    WINDOW.blit(score_surface, score_rect)

# Update the model path to use the recommended .keras format
MODEL_PATH = "snake_model.keras"

# Check if a saved model exists and load it
if os.path.exists(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded from", MODEL_PATH)
else:
    # Define the neural network model
    model = keras.Sequential([
        layers.Dense(128, input_dim=11, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='linear')  # 3 outputs for [Straight, Left, Right]
    ])
    # Update the loss function to use the full path
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=keras.losses.MeanSquaredError())
    print("New model created")

# Update the save_model function to reflect the new format
def save_model():
    model.save(MODEL_PATH)
    print("Model saved to", MODEL_PATH)

# Function to display game over message
def game_over_message():
    global snake_pos, snake_direction, food_pos, food_spawn, score, attempts, scores, attempt_count

    # Log the score for the current attempt
    attempts.append(attempt_count)
    scores.append(score)

    # Plot the graph of attempts vs. scores
    plt.plot(attempts, scores, marker='o')
    plt.title('Attempts vs. Score')
    plt.xlabel('Attempt')
    plt.ylabel('Score')
    plt.show()

    # Save the model
    save_model()

    # Reset game variables
    attempt_count += 1
    snake_pos = [[50, 50], [60, 50], [70, 50]]
    snake_direction = "RIGHT"
    food_pos = [random.randrange(1, (WIDTH // SNAKE_SIZE)) * SNAKE_SIZE,
                 random.randrange(1, (HEIGHT // SNAKE_SIZE)) * SNAKE_SIZE]
    food_spawn = True
    score = 0
# Main game loop

def reward(var):
    if var =="food":
        return 10
    if var == "game_over":
        return -10
    else:
        return 0
    
def action(var):
    if var == "Straight":
        #upadte snake position
        return [1,0,0]
        
    if var == "Left":
        return [0,1,0]
    if var == "Right":
        return [0,0,1]
    
def state(snake_pos, food_pos):
    if snake_pos[0][0] < food_pos[0]:
        if snake_pos[0][1] < food_pos[1]:
            return "Straight"
        else:
            return "Left"
    elif snake_pos[0][0] > food_pos[0]:
        if snake_pos[0][1] < food_pos[1]:
            return "Right"
        else:
            return "Straight"
    else:
        if snake_pos[0][1] < food_pos[1]:
            return "Straight"
        else:
            return "Left"

# Function to get the state of the game
def get_state(snake_pos, food_pos, snake_direction):
    head_x, head_y = snake_pos[0]
    food_x, food_y = food_pos

    state = [
        # Danger straight
        (snake_direction == "UP" and [head_x, head_y - SNAKE_SIZE] in snake_pos) or
        (snake_direction == "DOWN" and [head_x, head_y + SNAKE_SIZE] in snake_pos) or
        (snake_direction == "LEFT" and [head_x - SNAKE_SIZE, head_y] in snake_pos) or
        (snake_direction == "RIGHT" and [head_x + SNAKE_SIZE, head_y] in snake_pos),

        # Danger right
        (snake_direction == "UP" and [head_x + SNAKE_SIZE, head_y] in snake_pos) or
        (snake_direction == "DOWN" and [head_x - SNAKE_SIZE, head_y] in snake_pos) or
        (snake_direction == "LEFT" and [head_x, head_y - SNAKE_SIZE] in snake_pos) or
        (snake_direction == "RIGHT" and [head_x, head_y + SNAKE_SIZE] in snake_pos),

        # Danger left
        (snake_direction == "UP" and [head_x - SNAKE_SIZE, head_y] in snake_pos) or
        (snake_direction == "DOWN" and [head_x + SNAKE_SIZE, head_y] in snake_pos) or
        (snake_direction == "LEFT" and [head_x, head_y + SNAKE_SIZE] in snake_pos) or
        (snake_direction == "RIGHT" and [head_x, head_y - SNAKE_SIZE] in snake_pos),

        # Move direction
        snake_direction == "UP",
        snake_direction == "DOWN",
        snake_direction == "LEFT",
        snake_direction == "RIGHT",

        # Food location
        food_x < head_x,  # Food left
        food_x > head_x,  # Food right
        food_y < head_y,  # Food up
        food_y > head_y   # Food down
    ]

    return np.array(state, dtype=int)

# Function to train the model
def train_short_memory(state, action, reward, next_state, done):
    target = reward
    if not done:
        target = reward + 0.9 * np.amax(model.predict(next_state.reshape((1, 11))))

    target_f = model.predict(state.reshape((1, 11)))
    target_f[0][np.argmax(action)] = target
    model.fit(state.reshape((1, 11)), target_f, epochs=1, verbose=0)

# Function to get the action
def get_action(state):
    epsilon = 80 - score  # Decreasing epsilon with increasing score
    if random.randint(0, 200) < epsilon:
        return np.eye(3)[random.randint(0, 2)]  # Random action
    else:
        prediction = model.predict(state.reshape((1, 11)))
        return np.eye(3)[np.argmax(prediction)]

# Update the graph rendering logic
def update_graph():
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
    return graph_surface

# Ensure the model is saved when the program exits
def on_exit():
    save_model()
    print("Model saved on exit.")

atexit.register(on_exit)

# Replace manual controls with AI decision-making
while True:
    state_old = get_state(snake_pos, food_pos, snake_direction)
    action = get_action(state_old)

    if np.array_equal(action, [1, 0, 0]):
        pass  # Keep moving straight
    elif np.array_equal(action, [0, 1, 0]):
        if snake_direction == "UP":
            snake_direction = "LEFT"
        elif snake_direction == "DOWN":
            snake_direction = "RIGHT"
        elif snake_direction == "LEFT":
            snake_direction = "DOWN"
        elif snake_direction == "RIGHT":
            snake_direction = "UP"
    elif np.array_equal(action, [0, 0, 1]):
        if snake_direction == "UP":
            snake_direction = "RIGHT"
        elif snake_direction == "DOWN":
            snake_direction = "LEFT"
        elif snake_direction == "LEFT":
            snake_direction = "UP"
        elif snake_direction == "RIGHT":
            snake_direction = "DOWN"

    # Move the snake
    new_head = snake_pos[0].copy()
    if snake_direction == "UP":
        new_head[1] -= SNAKE_SIZE
    if snake_direction == "DOWN":
        new_head[1] += SNAKE_SIZE
    if snake_direction == "LEFT":
        new_head[0] -= SNAKE_SIZE
    if snake_direction == "RIGHT":
        new_head[0] += SNAKE_SIZE

    snake_pos.insert(0, new_head)

    # Check for collisions and rewards
    reward_value = 0
    if snake_pos[0] == food_pos:
        score += 10
        reward_value = 10
        food_spawn = False
    else:
        snake_pos.pop()

    if not food_spawn:
        food_pos = [random.randrange(1, (WIDTH // SNAKE_SIZE)) * SNAKE_SIZE,
                     random.randrange(1, (HEIGHT // SNAKE_SIZE)) * SNAKE_SIZE]
        food_spawn = True

    if (snake_pos[0][0] < 0 or snake_pos[0][0] > (WIDTH - SNAKE_SIZE) or
            snake_pos[0][1] < 0 or snake_pos[0][1] > (HEIGHT - SNAKE_SIZE)):
        reward_value = -10
        game_over_message()

    for block in snake_pos[1:]:
        if snake_pos[0] == block:
            reward_value = -10
            game_over_message()

    state_new = get_state(snake_pos, food_pos, snake_direction)
    train_short_memory(state_old, action, reward_value, state_new, game_over)

    # Fill the background color
    WINDOW.fill(BLACK)
    for pos in snake_pos:
        pygame.draw.rect(WINDOW, GREEN, pygame.Rect(pos[0], pos[1], SNAKE_SIZE, SNAKE_SIZE))
    pygame.draw.rect(WINDOW, WHITE, pygame.Rect(food_pos[0], food_pos[1], SNAKE_SIZE, SNAKE_SIZE))

    # Display the graph next to the game
    graph_surface = update_graph()
    WINDOW.blit(graph_surface, (WIDTH, 0))

    show_score(1, WHITE, 20)
    pygame.display.update()
    FPS.tick(SNAKE_SPEED)
