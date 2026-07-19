import pygame
import sys
import numpy as np
import math
import random
from collections import deque
from copy import deepcopy

from raycasts import Raycasts
from neural_network import NeuralNetwork
from layer import Layer
from helpers import listToColumn

OBJECT_TYPES = {
    "none": 0,
    "platform": 1,
    "wall": 2,
    "lava": 3,
    "star": 4
}

REWARDS = {
    "DISTANCE": 0.001,
    "NEW_PLATFORM_BASE": 1.5,
    "NEW_PLATFORM_INCREMENT": 0.25,
    "NEW_CELL": 0,
    "DEATH": -10,
    "WIN": 10,
    "LIVING_PENALTY": -0.015,
    "FALSE_JUMP": -0.01,
    "STALL": -8,
}

DRAW_RAYCASTS = True
MAX_EPISODE_STEPS = 900 # 15 seconds per episode
STALL_LIMIT = 420 # 7 seconds with no new platforms discovered - reset and give STALL penalty

NETWORK_SAVE_FILE_NAME = "checkpoints/overnight_run"
SAVE_FREQUENCY = 50
NETWORK_LOAD_PATH = None  # set to None to start training from scratch, otherwise load a pre-trained network

ALPHA = 0.6  # constants to determine how significant priorities are in training
BETA_START = 0.4
BETA_END = 1.0
BETA_FRAMES = 200_000  

def get_beta(frame): 
    return min(BETA_END, BETA_START + (BETA_END - BETA_START) * frame / BETA_FRAMES)

class FloorIsLavaEnv:
    def __init__(self):
        self.width = 800
        self.height = 600
        self.current_level = 1  # Set to a certain level for current testing layout
        self.lava_y = 570       # Position of the red lava bar
        self.raycasts = Raycasts(self)
        self.raycast_data = []
        self.network = NeuralNetwork([
            Layer(39, 128),
            Layer(128, 128),
            Layer(128, 64),
            Layer(64, 6)
        ])
        if NETWORK_LOAD_PATH is not None:
            self.network.load(NETWORK_LOAD_PATH)
        self.target_network = NeuralNetwork([])
        self.visited_cells = None
        self.visited_platfoms = None
        self.episodes = []
        self.current_episode = {}
        self.next_episode_id = 1
        self.episode_steps = 0
        self.steps_since_progress = 0
        
        # Define the 3 maps (Collaborating with Preita's coordinates)
        self.level_maps = {
            1: {
                "platforms": [pygame.Rect(50, 500, 180, 20), pygame.Rect(280, 420, 150, 20), pygame.Rect(400, 340, 150, 20)],
                "player_start": (100, 450),
                "star": pygame.Rect(540, 290, 30, 30)
            },
            2: {
                "platforms": [pygame.Rect(50, 520, 120, 20), pygame.Rect(220, 440, 120, 20), pygame.Rect(400, 480, 100, 20), pygame.Rect(590, 360, 160, 20)],
                "player_start": (80, 470),
                "star": pygame.Rect(620, 310, 30, 30)
            },
            3: {
                # Level 3: hardest map
                "platforms": [
                    pygame.Rect(50, 550, 150, 20), #plat1 - lowest
                    pygame.Rect(250, 420, 100, 20), #plat2 - second to lowest
                    pygame.Rect(100, 250, 120, 20), #plat 3 - next to highest
                    pygame.Rect(300, 175, 200, 20), #highest plat
                    pygame.Rect(380, 350, 30, 20) # small plarform
                ],
                "player_start": (70, 500),
                "star": pygame.Rect(550, 120, 30, 30)
            },
            # Level 4: added for testing
            4: {
                "platforms": [pygame.Rect(50, 500, 180, 20), pygame.Rect(280, 420, 150, 20), pygame.Rect(400, 340, 150, 20), pygame.Rect(230, 280, 100, 20)],
                "player_start": (100, 450),
                "star": pygame.Rect(250, 220, 30, 30)
            },
        }
        self.reset()

    def distance_to_goal(self):
        return math.sqrt(math.pow(self.player_x - self.goal.x, 2) + math.pow(self.player_y - self.goal.y, 2))

    def reset(self):
        """Loads the specific map based on the current active level"""
        map_data = self.level_maps[self.current_level]

        # Spawn player at the designated lowest platform for this specific level
        self.player_x, self.player_y = map_data["player_start"]
        self.platforms = map_data["platforms"]
        self.goal = map_data["star"]
        
        self.vel_x = 0
        self.vel_y = 0
        self.is_grounded = False
        self.done = False
        self.episode_steps = 0
        self.steps_since_progress = 0

        self.previous_distance = self.distance_to_goal()
        self.visited_cells = []
        self.visited_platfoms = [self.platforms[0]]

        # log results of greedy episodes (where the AI agent is on its own with no random actions mixed in)
        if len(self.episodes) > 1 and self.current_episode["greedy"]:

            print("GREEDY EPISODE")
            print("REWARD: " + str(self.current_episode["reward"]))
            print("SECONDS: " + str(self.current_episode["frames"] / 60))
            print("WON? " + str(self.current_episode["won"]))

        self.current_episode = {
            "id": self.next_episode_id,
            "reward": 0,
            "epsilon": 1,
            "frames": 0,
            "greedy": False,
            "won": False
        }

        # log results of every batch of SAVE_FREQUENCY episodes and save the network
        if self.current_episode["id"] % SAVE_FREQUENCY == 0:
            self.network.save(f"{NETWORK_SAVE_FILE_NAME}-{self.current_episode["id"] // SAVE_FREQUENCY}.npz")
            print(f"EPISODES {self.current_episode["id"] - SAVE_FREQUENCY} TO {self.current_episode["id"]}")

            reward_sum = 0
            frames_sum = 0
            wins = 0

            for episode in self.episodes[self.current_episode["id"] - SAVE_FREQUENCY:self.current_episode["id"]]:
                reward_sum += episode["reward"]
                frames_sum += episode["frames"]
                wins += int(episode["won"])
            
            print("-------------------------------")
            print("REWARD: " + str(reward_sum / SAVE_FREQUENCY))
            print("SECONDS PER ATTEMPT: " + str(frames_sum / 60 / SAVE_FREQUENCY))
            print("WIN %: " + str(wins / SAVE_FREQUENCY * 100))

        self.next_episode_id += 1

        return self._get_state()
    
    def _get_state(self):
        """Returns raw sensor data vectors needs for pyTorch features"""
        state = [
            (self.player_x + 15) / self.width,
            (self.player_y + 15) / self.height,
            self.vel_x / 5,
            self.vel_y / 14,
            (self.goal.x - self.player_x) / self.width,
            (self.goal.y - self.player_y) / self.height,
            int(self.is_grounded),
        ]

        for dist, obj_type, _, _ in self.raycast_data:
            state.extend(
                [
                    dist / 300,
                    OBJECT_TYPES[obj_type] / 4
                ]
            )

        return listToColumn(tuple(state))
    
    def step(self, action):
        reward = REWARDS["LIVING_PENALTY"] # to prevent stalling and encourage action

        old_distance = self.previous_distance

        move_type = action[0]
        should_jump = action[1]

        # Handle Horizontal Movement State
        if move_type == 1:
            self.vel_x = -5
        elif move_type == 2:
            self.vel_x = 5
        else:
            self.vel_x = 0

        # Handle Jump State
        if should_jump: 
            if self.is_grounded:
                self.vel_y = -10.5   # Optimized physics float speed
                self.is_grounded = False
            else:
                reward += REWARDS["FALSE_JUMP"] # punish agent for jumping in air (to train against overjumping)

        # Gravity
        self.vel_y += 0.4
        if self.vel_y > 14: # Terminal velocity cap
            self.vel_y = 14

        # Update Position Coordinates 
        self.player_x += self.vel_x
        
        # Screen boundary constraints for X
        if self.player_x < 0: self.player_x = 0
        if self.player_x > self.width - 30: self.player_x = self.width - 30
        
        # create a temporary collision rect for horizontal checking
        player_rect = pygame.Rect(self.player_x, self.player_y, 30, 30)
        
        for platform in self.platforms:
            if player_rect.colliderect(platform):
                # stop the player from moving into the platform horizontally
                if self.vel_x > 0:
                    self.player_x = platform.left - 30
                elif self.vel_x < 0:
                    self.player_x = platform.right
                player_rect.x = self.player_x

        # vertical movement and collision checking
        self.player_y += self.vel_y
        player_rect.y = self.player_y  
        
        self.is_grounded = False

        for platform in self.platforms:
            if player_rect.colliderect(platform):
                # landing on the platform from above
                if self.vel_y > 0:
                    if (player_rect.bottom - self.vel_y) <= platform.top + 10:
                        self.player_y = platform.top - 30
                        self.vel_y = 0
                        self.is_grounded = True
                        player_rect.y = self.player_y

                        if platform not in self.visited_platfoms: # reward the player for visiting a new platform
                            reward += REWARDS["NEW_PLATFORM_BASE"]
                            reward += REWARDS["NEW_PLATFORM_INCREMENT"] * len(self.visited_platfoms)
                            self.visited_platfoms.append(platform)
                            self.steps_since_progress = 0 # reset the progress counter
                        
                # jump up -> hitting head on the ceiling
                elif self.vel_y < 0:
                    # Check if player head was below the bottom of the platform before moving
                    if (player_rect.top - self.vel_y) >= platform.bottom - 10:
                        self.player_y = platform.bottom     # Snap right below the ceiling
                        self.vel_y = 0.5                     # Instantly cancel upward force and start falling
                        player_rect.y = self.player_y

        # Check Lava Defeat Condition
        if player_rect.bottom >= self.lava_y:
            self.done = True
            reward += REWARDS["DEATH"]

        # Check Goal Star Win Condition
        if player_rect.colliderect(self.goal):
            self.done = True
            self.current_episode["won"] = True
            reward += REWARDS["WIN"]

            if self.current_level < len(self.level_maps):
                self.current_level += 1
            else:
                self.current_level = 1

        # reward the agent for visiting a new part of the map - THIS REWARD IS CURRENTLY 0, will likely remove
        cell = (int(self.player_x // 40), int(self.player_y // 40))
        if cell not in self.visited_cells:
            self.visited_cells.append(cell)
            reward += REWARDS["NEW_CELL"]
                
        new_distance = self.distance_to_goal()

        # reward the agent for moving closer to the target
        reward += (old_distance - new_distance) * REWARDS["DISTANCE"]

        self.previous_distance = new_distance

        self.raycast_data = self.raycasts.cast_all_rays()
        self.episode_steps += 1
        self.steps_since_progress += 1

        # end every episode after 15 seconds
        if self.episode_steps > MAX_EPISODE_STEPS:
            self.done = True
        
        # punish agent for stalling
        if self.steps_since_progress > STALL_LIMIT:
            reward += REWARDS["STALL"]
            self.done = True

        self.current_episode["reward"] += reward

        return self._get_state(), reward, self.done

    def render(self, surface):
        """Draws the assets layout to the screen canvas"""
        surface.fill((20, 20, 20)) # Dark background matrix
        
        # Draw Platforms (blue rec)
        for platform in self.platforms:
            pygame.draw.rect(surface, (31, 78, 120), platform)
            
        # Draw Target Goal Zone (Yellow Star)
        pygame.draw.rect(surface, (255, 192, 0), self.goal)
        
        # Draw Danger Zone Hazard Line (Red Lava)
        pygame.draw.rect(surface, (192, 0, 0), (0, self.lava_y, self.width, self.height - self.lava_y))
        
        # Draw Player State Node (Green Circle)
        pygame.draw.circle(surface, (112, 173, 71), (int(self.player_x) + 15, int(self.player_y) + 15), 15)

        # Draw raycasts

        if DRAW_RAYCASTS:
            for ray in self.raycast_data:
                x = self.player_x + 15
                y = self.player_y + 15
                dist, _, dx, dy = ray

                end_x = x + dx * dist
                end_y = y + dy * dist

                pygame.draw.line(surface, (255, 255, 255), (x, y), (end_x, end_y))
        
        # Render Level Title Display Text
        font = pygame.font.SysFont("Calibri", 24, bold=True)
        text_surf = font.render(f"CURRENT MAP LEVEL: {self.current_level}", True, (255, 255, 255))
        surface.blit(text_surf, (20, 20))
        episode_text = font.render(f"Episode: {self.current_episode["id"]}", True, (255, 255, 255))
        surface.blit(episode_text, (20, 50))
        reward_text = font.render(f"Reward: {self.current_episode["reward"]}", True, (255, 255, 255))
        surface.blit(reward_text, (20, 80))
        
        pygame.display.flip()

    
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Game Env Testing Room")
    clock = pygame.time.Clock()
    
    env = FloorIsLavaEnv()
    
    running = True
    state = None
    # epsilon - the probability that the agent's action gets picked randomly
    epsilon = 0.3 # exploring random actions first helps the agent learn what works and what doesn't
    replay_buffer = deque(maxlen=50000) # saved experiences that the agent can learn from
    priorities = deque(maxlen=50000) # priorities of experiences - those where the neural network predicted more incorrectly should be trained on more
    frame_count = 0
    total_frames = 0
    total_updates = 0
    greedy = False # every SAVE FREQUENCY episodes run a couple fully greedy runs to evaluate model

    while running:
        clock.tick(60)

        move_action = 0
        jump_action = False

        agent_action = 0

        if env.next_episode_id % SAVE_FREQUENCY > 0 and env.next_episode_id % SAVE_FREQUENCY < 5 and env.next_episode_id > 5:
            greedy = True
        else:
            greedy = False

        if state is not None:
            if random.random() < epsilon and not greedy:
                agent_action = random.randint(0, 5)

            else:
                prediction, _ = env.network.run(state)
                agent_action = np.argmax(prediction)
    
        keys = pygame.key.get_pressed()
        
        # agent actions 
        # 0 - idle, 1 - left, 2 - right, 3 - left + jump, 4 - right + jump, 5 - jump
        if keys[pygame.K_a] or agent_action in [1, 3]:
            move_action = 1
        elif keys[pygame.K_d] or agent_action in [2, 4]:
            move_action = 2

        if keys[pygame.K_w] or keys[pygame.K_SPACE] or agent_action in [3, 4, 5]:
            jump_action = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        next_state, reward, done = env.step((move_action, jump_action))

        # save every decision and its outcomes in the replay buffer for training
        if state is not None and not greedy:
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

        if len(replay_buffer) >= 5000 and total_frames % 4 == 0 and not greedy:
            priority_arr = np.array(priorities, dtype=np.float64) ** ALPHA
            probs = priority_arr / priority_arr.sum()

            indices = np.random.choice(len(replay_buffer), size=32, p=probs, replace=True) # pick experiences to learn from based on priorities

            beta = get_beta(total_frames)
            weights = (len(replay_buffer) * probs[indices]) ** (-beta) # progressively weigh updates more evenly as time goes on
            weights /= weights.max()  # normalize so max weight = 1, keeps updates stable

            buf_snapshot = list(replay_buffer)  # O(n) once per batch, avoids O(n) per-index deque access

            for idx, is_weight in zip(indices, weights):
                _state, _agent_action, _reward, _next_state, _done = buf_snapshot[idx]

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
            env.current_episode["greedy"] = greedy
            env.episodes.append(env.current_episode)

            # slowly decay epsilon, but always keep a little randomness in to encourage exploration
            epsilon *= 0.995
            epsilon = max(0.05, epsilon)

            frame_count = 0
            state = env.reset()

        frame_count += 1
        total_frames += 1
            
    pygame.quit()
    sys.exit()