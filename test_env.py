# Importing libraries
import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from matplotlib import animation

# Defining constants
MAP_SIZE = 12  # Map size in meters
OBSTACLES = [(6, 4), (6, 5), (6, 6), (6, 7), (6, 8)]  # Obstacle coordinates in meters
GOAL = (11, 11)  # Goal coordinates in meters
GOAL_RADIUS = 0.5  # Goal radius in meters
ROBOT_RADIUS = 0.25  # Robot radius in meters
MAX_LINEAR_VELOCITY = 0.3  # Maximum linear velocity in meters per second
MIN_LINEAR_VELOCITY = -0.1  # Minimum linear velocity in meters per second
MAX_ANGLE_CHANGE = 0.38  # Maximum angle change in radians per second
TIME_STEP = 0.1  # Time step in seconds

# Defining rewards and penalties
REWARD_GOAL = 100  # Reward for reaching the goal
PENALTY_OBSTACLE = -100  # Penalty for hitting an obstacle
PENALTY_BOUNDARY = -100  # Penalty for going outside the map
REWARD_STEP = -1  # Reward for each step


# Defining the environment class
class SquareMapEnv(gym.Env):
    def __init__(self):
        # Defining the action space as a box with two dimensions: linear velocity and angle change
        self.action_space = spaces.Box(low=np.array([MIN_LINEAR_VELOCITY, -MAX_ANGLE_CHANGE]),
                                       high=np.array([MAX_LINEAR_VELOCITY, MAX_ANGLE_CHANGE]), dtype=np.float32)

        # Defining the observation space as a box with four dimensions: x position, y position, angle and distance to goal
        self.observation_space = spaces.Box(low=np.array([0, 0, -np.pi, 0]),
                                            high=np.array([MAP_SIZE, MAP_SIZE, np.pi, np.sqrt(2) * MAP_SIZE]),
                                            dtype=np.float32)

        # Initializing the state variables
        self.x = None  # x position of the robot in meters
        self.y = None  # y position of the robot in meters
        self.angle = None  # angle of the robot in radians
        self.distance = None  # distance to the goal in meters

        # Initializing the renderer
        self.renderer = None

    def reset(self):
        # Resetting the state variables to random values within the map boundaries and away from obstacles and goal
        self.x = np.random.uniform(ROBOT_RADIUS, MAP_SIZE - ROBOT_RADIUS)
        self.y = np.random.uniform(ROBOT_RADIUS, MAP_SIZE - ROBOT_RADIUS)
        self.angle = np.random.uniform(-np.pi, np.pi)
        self.distance = np.sqrt((self.x - GOAL[0]) ** 2 + (self.y - GOAL[1]) ** 2)

        while self._is_collision() or self._is_goal():
            self.x = np.random.uniform(ROBOT_RADIUS, MAP_SIZE - ROBOT_RADIUS)
            self.y = np.random.uniform(ROBOT_RADIUS, MAP_SIZE - ROBOT_RADIUS)
            self.angle = np.random.uniform(-np.pi, np.pi)
            self.distance = np.sqrt((self.x - GOAL[0]) ** 2 + (self.y - GOAL[1]) ** 2)

        # Returning the observation as a numpy array
        return np.array([self.x, self.y, self.angle, self.distance])

    def step(self, action):
        # Clipping the action values to the action space boundaries
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Updating the state variables according to the action and the kinematics model of the robot
        linear_velocity = action[0]  # Linear velocity in meters per second
        angle_change = action[1]  # Angle change in radians per second

        self.x += linear_velocity * np.cos(self.angle) * TIME_STEP  # Updating x position in meters
        self.y += linear_velocity * np.sin(self.angle) * TIME_STEP  # Updating y position in meters
        self.angle += angle_change * TIME_STEP  # Updating angle in radians

        # Normalizing the angle to be between -pi and pi
        self.angle = (self.angle + np.pi) % (2 * np.pi) - np.pi

        # Updating the distance to the goal in meters
        self.distance = np.sqrt((self.x - GOAL[0]) ** 2 + (self.y - GOAL[1]) ** 2)

        # Initializing the reward and done variables
        reward = None  # Reward for the current step
        done = None  # Whether the episode is over or not

        # Checking if the robot has reached the goal
        if self._is_goal():
            # Setting the reward to the positive reward and done to True
            reward = REWARD_GOAL
            done = True
        # Checking if the robot has hit an obstacle or gone outside the map
        elif self._is_collision() or self._is_outside():
            # Setting the reward to the negative penalty and done to True
            reward = PENALTY_OBSTACLE if self._is_collision() else PENALTY_BOUNDARY
            done = True
        # Otherwise, the robot is still moving in the map
        else:
            # Setting the reward to the step reward and done to False
            reward = REWARD_STEP
            done = False

        # Returning the observation, reward, done and info as a tuple
        return (np.array([self.x, self.y, self.angle, self.distance]), reward, done, {})

    def render(self, mode='human'):
        # Creating a renderer if it does not exist
        if self.renderer is None:
            self.renderer = plt.figure()

        # Clearing the previous plot
        plt.clf()

        # Plotting the map boundaries as a black rectangle
        plt.plot([0, 0, MAP_SIZE, MAP_SIZE, 0], [0, MAP_SIZE, MAP_SIZE, 0, 0], 'k-')

        # Plotting the obstacles as red circles
        for obstacle in OBSTACLES:
            plt.plot(obstacle[0], obstacle[1], 'ro', markersize=20)

        # Plotting the goal as a green circle
        plt.plot(GOAL[0], GOAL[1], 'go', markersize=10)

        # Plotting the robot as a blue circle with an arrow indicating its angle
        plt.plot(self.x, self.y, 'bo', markersize=5)
        plt.arrow(self.x, self.y, ROBOT_RADIUS * np.cos(self.angle), ROBOT_RADIUS * np.sin(self.angle), color='b',
                  head_width=0.1)

        # Setting the axis limits and labels
        plt.xlim(0, MAP_SIZE)
        plt.ylim(0, MAP_SIZE)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')

        # Showing the plot
        plt.show()

    def close(self):
        # Closing the renderer if it exists
        if self.renderer is not None:
            plt.close(self.renderer)

    def _is_collision(self):
        # Checking if the robot is colliding with any obstacle by comparing their distances with their radii
        for obstacle in OBSTACLES:
            distance = np.sqrt((self.x - obstacle[0]) ** 2 + (self.y - obstacle[1]) ** 2)
            if distance < ROBOT_RADIUS + 0.5:
                return True

        return False

    def _is_outside(self):
        # Checking if the robot is outside the map boundaries by comparing its position with the map size and its radius
        if self.x < ROBOT_RADIUS or self.x > MAP_SIZE - ROBOT_RADIUS or self.y < ROBOT_RADIUS or self.y > MAP_SIZE - ROBOT_RADIUS:
            return True

        return False

    def _is_goal(self):
        # Checking if the robot is within the goal radius by comparing its distance to the goal with the goal radius
        if self.distance < GOAL_RADIUS:
            return True

        return False


# Testing the environment with random actions

# Creating an instance of the environment
env = SquareMapEnv()

# Resetting the environment and getting the initial observation
observation = env.reset()

# Rendering the initial state of the environment
env.render()

# Initializing a variable to store the total reward
total_reward = 0
frames = []

# Looping until the episode is over
while True:
    # Choosing a random action from the action space
    action = env.action_space.sample()

    # Taking a step in the environment and getting the observation, reward, done and info
    observation, reward, done, info = env.step(action)

    # Rendering the current state of the environment
    frame = env.render()
    frames.append(frame)

    # Updating the total reward
    total_reward += reward

    # Printing the observation, action and reward for debugging purposes
    print(f"Observation: {observation}")
    print(f"Action: {action}")
    print(f"Reward:{reward}")

    # Checking if the episode is over
    if done:
        # Printing the final reward and breaking the loop
        print(f"Final reward: {total_reward}")
        break

# Closing the environment
env.close()

# the time step and the frames per second
ani = animation.ArtistAnimation(plt.gcf(), frames, interval=TIME_STEP * 1000, blit=True, repeat_delay=1000)

# Saving the animation as a video file
ani.save('video.mp4')