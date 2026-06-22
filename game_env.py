import pygame
import sys

class FloorIsLavaEnv:
    def __init__(self):
        self.width = 800
        self.height = 600
        self.current_level = 1  # Set to a certain level for current testing layout
        self.lava_y = 570       # Position of the red lava bar
        
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
            }
        }
        self.reset()

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
        return self._get_state()
    
    def _get_state(self):
        """Returns raw sensor data vectors needs for pyTorch features"""
        return (self.player_x, self.player_y, self.vel_x, self.vel_y)
    
    def step(self, action):

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
        if should_jump and self.is_grounded:
            self.vel_y = -10.5   # Optimized physics float speed
            self.is_grounded = False

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
                        
                # jump up -> hitting head on the ceiling
                elif self.vel_y < 0:
                    # Check if player head was below the bottom of the platform before moving
                    if (player_rect.top - self.vel_y) >= platform.bottom - 10:
                        self.player_y = platform.bottom     # Snap right below the ceiling
                        self.vel_y = 0.5                     # Instantly cancel upward force and start falling
                        player_rect.y = self.player_y

        # Check Lava Defeat Condition
        if player_rect.bottom >= self.lava_y:
            print(f"PLAYER DIED IN THE LAVA ON LEVEL {self.current_level}!")
            self.done = True
            return self._get_state(), self.done

        # Check Goal Star Win Condition
        if player_rect.colliderect(self.goal):
            print(f"LEVEL {self.current_level} COMPLETE!")
            self.done = True
            
            if self.current_level < 3:
                self.current_level += 1
            else:
                print("GAME OVER!")
                self.current_level = 1
                
            return self._get_state(), self.done

        return self._get_state(), self.done

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
        
        # Render Level Title Display Text
        font = pygame.font.SysFont("Calibri", 24, bold=True)
        text_surf = font.render(f"CURRENT MAP LEVEL: {self.current_level}", True, (255, 255, 255))
        surface.blit(text_surf, (20, 20))
        
        pygame.display.flip()

    
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Game Env Testing Room")
    clock = pygame.time.Clock()
    
    env = FloorIsLavaEnv()
    
    running = True
    while running:
        clock.tick(60)
        
        # Track frame movement states independently
        move_action = 0  
        jump_action = False
        
        # Check keystrokes simultaneously to support diagonal running jumps
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]:
            move_action = 1
        elif keys[pygame.K_d]:
            move_action = 2
            
        if keys[pygame.K_w] or keys[pygame.K_SPACE]:
            jump_action = True
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        state, done = env.step((move_action, jump_action))
        
        # Render update maps
        env.render(screen)
        
        if done:
            env.reset()
            
    pygame.quit()
    sys.exit()