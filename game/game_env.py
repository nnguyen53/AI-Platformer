import pygame
import sys
import numpy as np
import math

from agent.raycasts import Raycasts
from agent.neural_network import NeuralNetwork
from agent.layer import Layer
from agent.config import AgentConfig
from utils.helpers import listToColumn
from game.maps import MAPS
from game.config import GameConfig
from utils.consts import *

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
        self.game_config = GameConfig()
        self.agent_config = AgentConfig()
        self.target_network = NeuralNetwork([])
        self.visited_cells = None
        self.visited_platfoms = None
        self.episodes = []
        self.current_episode = {}
        self.next_episode_id = 0
        self.episode_steps = 0
        self.steps_since_progress = 0
        self.level_maps = MAPS
        self.reset()

    def load_network(self):
        if self.game_config.NETWORK_LOAD_PATH is not None:
            self.network.load(self.game_config.NETWORK_LOAD_PATH)

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

        self.current_episode = {
            "id": self.next_episode_id,
            "reward": 0,
            "epsilon": 1,
            "frames": 0,
            "won": False
        }

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
        reward = self.agent_config.REWARDS["LIVING_PENALTY"] # to prevent stalling and encourage action

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
                reward += self.agent_config.REWARDS["FALSE_JUMP"] # punish agent for jumping in air (to train against overjumping)

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
                            reward += self.agent_config.REWARDS["NEW_PLATFORM_BASE"]
                            reward += self.agent_config.REWARDS["NEW_PLATFORM_INCREMENT"] * len(self.visited_platfoms)
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
            reward += self.agent_config.REWARDS["DEATH"]

        # Check Goal Star Win Condition
        if player_rect.colliderect(self.goal):
            self.done = True
            self.current_episode["won"] = True
            reward += self.agent_config.REWARDS["WIN"]
                
        new_distance = self.distance_to_goal()

        # reward the agent for moving closer to the target
        reward += (old_distance - new_distance) * self.agent_config.REWARDS["DISTANCE"]

        self.previous_distance = new_distance

        self.raycast_data = self.raycasts.cast_all_rays()
        self.episode_steps += 1
        self.steps_since_progress += 1

        # end every episode after 15 seconds

        if self.game_config.AI_MODE:
            if self.episode_steps > self.agent_config.MAX_EPISODE_STEPS:
                self.done = True
            
            # punish agent for stalling
            if self.steps_since_progress > self.agent_config.STALL_LIMIT:
                reward += self.agent_config.REWARDS["STALL"]
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
        if self.game_config.DRAW_RAYCASTS:
            for ray in self.raycast_data:
                x = self.player_x + 15
                y = self.player_y + 15
                dist, _, dx, dy = ray

                end_x = x + dx * dist
                end_y = y + dy * dist

                pygame.draw.line(surface, (255, 255, 255), (x, y), (end_x, end_y))
        
        # Render Level Title Display Text
        font = pygame.font.SysFont("Calibri", 24)
        text_surf = font.render(f"Current Level: {self.current_level}", True, (255, 255, 255))
        surface.blit(text_surf, (20, 20))

        if self.game_config.AI_MODE:
            episode_text = font.render(f"Attempt: {self.current_episode["id"]}", True, (255, 255, 255))
            surface.blit(episode_text, (20, 50))
            reward_text = font.render(f"Reward: {self.current_episode["reward"]}", True, (255, 255, 255))
            surface.blit(reward_text, (20, 80))

        tip_surf = font.render("Press ESC to pause", True, (255, 255, 255))
        surface.blit(tip_surf, (605, 20))
        
        pygame.display.flip()

