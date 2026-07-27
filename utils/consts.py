from game.maps import MAPS
from utils.helpers import resource_path

OBJECT_TYPES = {
    "none": 0,
    "platform": 1,
    "wall": 2,
    "lava": 3,
    "star": 4
}

DRAW_RAYCASTS = True
MAX_EPISODE_STEPS = 900 
STALL_LIMIT = 420 

NETWORK_SAVE_FILE_NAME = "checkpoints/overnight_run"
SAVE_FREQUENCY = 50

NUM_LEVELS = len(MAPS)
BUFFER_CAP = 50000 

GAME_FONT = resource_path("assets/Fonts/Calibri.ttf")
GAME_FONT_BOLD = resource_path("assets/Fonts/Calibribold.ttf")