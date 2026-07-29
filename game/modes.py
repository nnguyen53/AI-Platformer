import pygame
import random
import sys
from utils.consts import *
from game.game_env import FloorIsLavaEnv
from collections import deque
from copy import deepcopy
from utils.helpers import *
from game.menu import draw_pause_overlay

def run_ai_mode(screen, clock, model_path, level):
    """Runs the game in AI mode using the provided UI objects, model path, and level"""  
    env = FloorIsLavaEnv()

    env.game_config.NETWORK_LOAD_PATH = model_path
    env.current_level = level

    env.load_network()
    env.reset()

    peak = MAPS[level]["peak_win_rate"] if model_path else 0.0

    running = True
    paused = False  # tracks whether the game is currently paused
    state = None
    # epsilon - the probability that the agent's action gets picked randomly
    epsilon = max(0.15, 1.0 - peak)
    epsilon_floor = max(0.05, 0.3 * (1 - peak))
    epsilon_decay = 0.999 + (0.0008 * (1 - peak))
    replay_buffer = deque(maxlen=BUFFER_CAP) # saved experiences that the agent can learn from
    priorities = deque(maxlen=BUFFER_CAP) # priorities of experiences - those where the neural network predicted more incorrectly should be trained on more
    frame_count = 0
    total_frames = 0
    total_updates = 0

    while running:
        clock.tick(60)

        # event handling moved to the top of the loop and now
        # supports pausing (ESC) and quitting back to the menu (Q while paused)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:  
                if event.key == pygame.K_ESCAPE:
                    paused = not paused  # first press pauses, second press resumes
                elif event.key == pygame.K_q and paused:
                    return
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_d and not paused:
                    env.game_config.DEBUG_MODE = not env.game_config.DEBUG_MODE

        if paused:  
            draw_pause_overlay(screen)
            continue  # skip all game logic and training while paused

        move_action = 0
        jump_action = False

        agent_action = 0

        if state is not None:
            if random.random() < epsilon:
                agent_action = random.randint(0, 5)

            else:
                prediction, _ = env.network.run(state)
                agent_action = np.argmax(prediction)
        
        # agent actions 
        # 0 - idle, 1 - left, 2 - right, 3 - left + jump, 4 - right + jump, 5 - jump
        if agent_action in [1, 3]:
            move_action = 1
        elif agent_action in [2, 4]:
            move_action = 2

        if agent_action in [3, 4, 5]:
            jump_action = True

        next_state, reward, done = env.step((move_action, jump_action))

        # save every decision and its outcomes in the replay buffer for training
        if state is not None:
            replay_buffer.append([
                state,
                agent_action,
                reward,
                next_state,
                done,
            ])
            priorities.append(max(priorities, default=1.0)) # give new experiences max priority automatically

        state = next_state

        if total_frames % 1000 == 0:
            env.target_network = deepcopy(env.network) # target network outputs are copied periodically, and is what we use to determine target output and compute error 
            # we need a target network so the we're not constantly training the network we're trying to match the outputs of

        if total_frames >= 5000 and total_frames % 4 == 0:
            batch = sample_batch(total_frames, replay_buffer, priorities, env.agent_config)

            for idx, is_weight in batch:
                _state, _agent_action, _reward, _next_state, _done = replay_buffer[idx]

                prediction, caches = env.network.run(_state)
                online_next, _ = env.network.run(_next_state)
                target_next, _ = env.target_network.run(_next_state)

                target = prediction.copy()
                if _done:
                    target[_agent_action] = _reward
                else:
                    best_action = np.argmax(online_next)
                    target[_agent_action] = _reward + 0.99 * target_next[best_action] # update the target (correct value for chosen action) based on outcomes from the next couple frames

                td_error = abs(float(target[_agent_action][0]) - float(prediction[_agent_action][0]))
                priorities[idx] = td_error + 1e-5  # avoid a priority of exactly 0, which would make it unsamplable

                # train the network
                env.network.backpropagate(prediction, target, caches, weight=is_weight)
                total_updates += 1

        env.render(screen)

        if done:
            env.current_episode["epsilon"] = epsilon
            env.current_episode["frames"] = frame_count

            env.episodes.append(env.current_episode)

            # slowly decay epsilon, but always keep a little randomness in to encourage exploration
            epsilon *= epsilon_decay
            epsilon = max(epsilon_floor, epsilon)

            frame_count = 0
            state = env.reset()

        frame_count += 1
        total_frames += 1


def run_manual_mode(screen, clock):  
    """Runs the game with the user controlling the player, no AI involved"""
    env = FloorIsLavaEnv()

    env.game_config.DEBUG_MODE = False
    env.game_config.AI_MODE = False
    
    running = True
    paused = False  # tracks whether the game is currently paused
    while running:
        clock.tick(60)

        for event in pygame.event.get():  # supports pausing (ESC) and quitting back to the menu (Q while paused)
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    paused = not paused  # first press pauses, second press resumes
                elif event.key == pygame.K_q and paused:
                    return  # quit back to the main menu while paused

        if paused:  
            draw_pause_overlay(screen)
            continue  # skip all game logic while paused

        move_action = 0
        jump_action = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]:
            move_action = 1
        elif keys[pygame.K_d]:
            move_action = 2

        if keys[pygame.K_w] or keys[pygame.K_SPACE]:
            jump_action = True

        env.step((move_action, jump_action))
        env.render(screen)

        if env.done:
            if env.current_episode["won"]:  # advance to the next level on a win, wrapping back to level 1 after the last one
                env.current_level = env.current_level + 1 if env.current_level < NUM_LEVELS else 1
            env.reset()