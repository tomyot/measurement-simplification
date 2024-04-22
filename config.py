import numpy as np
import gtsam

# Default parameters
SCENARIO = 'uniform' 
PRIOR_MAPPING = 'line' # 'line' or 'boxes'
NUM_LANDMARKS = 5000
MAP_SIZE = 40
GOAL = (32, 0)
NUM_PATHS = 20
NUM_FIGURES = 1
ITERATIONS = 3
# Prior information
PRIOR = np.array([20, 0, 0])  # x, y, heading

# Random actions
NUM_RANDOM_ACTIONS = 100
ACTIONS_RANDOM = [np.random.uniform(-5, 5, size=(1, 3))[0] for _ in range(NUM_RANDOM_ACTIONS)]

# Predefined actions
ACTIONS = [
    np.array([0, 1, 0]),  #  0 forward
    np.array([0, -1, 0]),  # 1 backward
    np.array([1, 0, 0]),  # 2 step right
    np.array([-1, 0, 0]),  # 3 step left
    np.array([0, 0, np.radians(-90)]),  # 4 turn right
    np.array([0, 0, np.radians(90)]),  # 5 turn left
    np.array([5, 0, 0]),  # 6 observe
    np.array([3, 0, 0]),  # 7 observe small step
    np.array([0, 5, 0]),  # 8 explore
    np.array([0, 2, 0]),  # 9
    np.array([0, 0, np.radians(-45)])  # 10
]

# Other parameters
PRIOR_NOISE = np.array([5e-1, 5e-1, 1e-3], dtype=float)
MOTION_MODEL_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-0, 1e-0, 1e-1], dtype=float))
OBSERVATION_MODEL_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([2e-0, 2e-0], dtype=float))
