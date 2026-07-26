class AgentConfig:
    def __init__(self):
        self.REWARDS = {
            "DISTANCE": 0.001,
            "NEW_PLATFORM_BASE": 1.5,
            "NEW_PLATFORM_INCREMENT": 0.25,
            "DEATH": -10,
            "WIN": 10,
            "LIVING_PENALTY": -0.015,
            "FALSE_JUMP": -0.01,
            "STALL": -8,
        }

        self.MAX_EPISODE_STEPS = 900
        self.STALL_LIMIT = 420

        self.ALPHA = 0.6
        self.BETA_START = 0.4
        self.BETA_END = 1.0
        self.BETA_FRAMES = 2_000_000
        self.SAMPLES_PER_BATCH = 32

        self.EPSILON_START = 0.4
        self.EPSILON_END = 0.2
        self.EPSILON_DELAY = 0.9985