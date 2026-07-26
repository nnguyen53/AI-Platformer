MODELS = {
    "Untrained": [None, "A completely untrained model. \nNetwork parameters will be initialized randomly."],
    "Basic": ["checkpoints/basic.npz", "A basic model. \nTrained for around 9000 attempts and can solve about 3 levels."],
    "Medium": ["checkpoints/medium.npz", "A medium-level model. \nTrained for around 11500 attempts and can solve about 5 levels."],
    "Advanced": ["checkpoints/advanced.npz", "A more advanced model. \nTrained for around 14000 attempts and can solve about 6 levels"],
    "Best": ["checkpoints/advanced.npz", "The best model we were able to train. \nTrained for around 19000 attempts total and can solve about 11 levels."]
}