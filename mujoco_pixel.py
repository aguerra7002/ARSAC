import gym
import mujoco_py
import matplotlib.pyplot as plt

# Creates the environment
env = gym.make("HalfCheetah-v2")

# Sets up the viewer
env.env.viewer = mujoco_py.MjViewer(env.env.sim)

# gets the image data of dimension (64, 64, 3)
img = env.env.sim.render(camera_name='track', width=64, height=64, depth=False)
plt.imshow(img, origin='lower')
plt.show()