from maps import MAPS

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
MAX_EPISODE_STEPS = 900 
STALL_LIMIT = 420 

NETWORK_SAVE_FILE_NAME = "checkpoints/overnight_run"
SAVE_FREQUENCY = 50
NETWORK_LOAD_PATH = "checkpoints/overnight_run-50.npz"

NUM_LEVELS = len(MAPS)
PER_LEVEL_CAP = 50000 // NUM_LEVELS

ALPHA = 0.6  
BETA_START = 0.4
BETA_END = 1.0
BETA_FRAMES = 2_000_000  

SAMPLES_PER_LEVEL = 2