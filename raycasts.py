import math
import pygame

class Raycasts:
    def __init__(self, game_state, num_rays = 16):
        self.game_state = game_state
        self.RAY_DIRECTIONS = []

        for ray in range(num_rays):
            angle = math.radians(ray * (360 / num_rays))
            self.RAY_DIRECTIONS.append((math.cos(angle), math.sin(angle)))

    def cast_ray(self, x, y, dx, dy):
        """
        Cast a ray starting at (x,y) and increasing x and y by (dx, dy) each step
        Returns the distance and the type of object hit
        """

        max_distance = 300
        step = 4
        distance = 0

        while distance < max_distance:
            rx = x + dx * distance
            ry = y + dy * distance
            
            # Hit lava
            if ry > self.game_state.lava_y:
                return (distance, "lava")

            # Hit the wall
            if rx < 0 or rx > self.game_state.width or ry < 0:
                return (distance, "wall")
            
            point = pygame.Rect(rx, ry, 1, 1)

            # Hit platform
            for platform in self.game_state.platforms:
                if platform.colliderect(point):
                    return (distance, "platform")
                
            # Hit star
            if self.game_state.goal.colliderect(point):
                return (distance, "star")
            
            distance += step

        return (max_distance, "none")
    
    def cast_all_rays(self):
        "Casts all rays in self.RAY_DIRECTIONS and returns data (distance, object type hit, dx, dy) for each"

        ray_data = []
        for ray in self.RAY_DIRECTIONS:
            x = self.game_state.player_x + 15
            y = self.game_state.player_y + 15
            dx, dy = ray
            
            dist, obj_type = self.cast_ray(x, y, dx, dy)

            ray_data.append((dist, obj_type, dx, dy))
        
        return ray_data
